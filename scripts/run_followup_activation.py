"""F6 bounded activation audit: frozen native model, matched interventions, family replication."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch,joblib
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.readouts import decode,effects,errors
from zj_wam.io import resolve,write_json,sha256


def choose_cases(out,split,resolution,count):
    pool=[]
    for path in sorted((out/'lewm_audit/effects').glob(f'{split}_*_h05.npz')):
        with np.load(path) as a:
            truth=a['true_effect'][:,0];real=effects(a['projected_cls_ridge_real_encoded_pose'])[:,0]
            imagined=effects(a['projected_cls_ridge_predicted_pose'])[:,0]
        meta=json.loads((out/'interventions'/path.with_suffix('.json').name).read_text())
        pe,re=errors(real,truth);ppe,pre=errors(imagined,truth)
        for axis,true,error,pred_error,threshold in [('position',np.linalg.norm(truth[:,:2],axis=1),pe,ppe,resolution['position']),
                                                   ('rotation',np.abs(truth[:,2]),re,pre,resolution['rotation'])]:
            mask=(true>threshold)&(error<=.5*true)
            for idx in np.flatnonzero(mask):
                pool.append({'state_id':meta['state_id'],'horizon':5,'index':int(idx),'split':split,
                             'family':meta['initial_pose_family'],'axis':axis,'response_class':'success' if pred_error[idx]<.5*true[idx] else 'failure',
                             'actual_effect':float(true[idx]),'real_error':float(error[idx]),'predicted_error':float(pred_error[idx])})
    rng=np.random.default_rng(20261011);rng.shuffle(pool);used=set();chosen=[];shortfalls=[]
    for status in ['failure','success']:
      for axis in ['position','rotation']:
        candidates=[p for p in pool if p['response_class']==status and p['axis']==axis];selected=[]
        for item in candidates:
            if item['family'] in used:continue
            selected.append(item);used.add(item['family'])
            if len(selected)==count:break
        chosen.extend(selected)
        if len(selected)<count:shortfalls.append({'split':split,'class':status,'axis':axis,'requested':count,'available_distinct':len(selected)})
    return chosen,shortfalls


def capture(model,out,spec):
    key=spec['state_id']+'_h05'
    with np.load(out/'interventions'/f'{key}.npz') as a:
        images=a['history_images'];past=a['past_actions'];raw=a['actions'][[0,spec['index']]]
    with np.load(out/'lewm_audit/effects'/f'{key}.npz') as a:
        real=a['real_projected_cls'][[0,spec['index']],0];truth=a['true_effect'][spec['index'],0]
    predict,_=model.predictor(images,past);acts={};layers=model.model.predictor.transformer.layers
    handles=[layer.register_forward_hook(lambda module,inputs,output,i=i:acts.__setitem__(i,output.detach().clone())) for i,layer in enumerate(layers)]
    inputs=[torch.as_tensor(model.normalize(a).reshape(-1),device=model.device) for a in raw]
    try:
        with torch.no_grad():
            z0=predict(inputs[0]);a0={k:v.clone() for k,v in acts.items()}
            z1=predict(inputs[1]);a1={k:v.clone() for k,v in acts.items()}
    finally:
        for h in handles:h.remove()
    return predict,inputs,[z0,z1],[a0,a1],real,truth


def main():
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());out=resolve(cfg['output']);dest=out/'activation';dest.mkdir(exist_ok=True)
    resolution=json.loads((out/'lewm_audit/effects/resolution.json').read_text())['thresholds']['projected_cls_ridge']
    torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
    probe=joblib.load(out/'lewm_audit/readouts/projected_cls_ridge.joblib');lock=dest/'selection.json'
    if lock.exists():design=json.loads(lock.read_text())
    else:
        dev,short1=choose_cases(out,'development',resolution,3);cal,short2=choose_cases(out,'calibration',resolution,2)
        if not dev:raise ValueError('No development cases satisfy real-effect resolution prerequisites')
        scores=[]
        for spec in dev:
            _,_,_,acts,_,_=capture(model,out,spec)
            scores.append(np.stack([(acts[1][i]-acts[0][i])[0,-1].abs().cpu().numpy() for i in range(6)]))
        ranking=np.mean(scores,axis=0);channels={str(i):np.argsort(ranking[i])[-16:].tolist() for i in range(6)}
        design={'development':dev,'calibration':cal,'shortfalls':short1+short2,'channels':channels,'component_count':16,
                'doses':[.5,1.0],'random_repeats':3,'resolution':resolution,
                'selection':'Largest mean nominal-to-intervention activation difference, development cases only; fixed channels reused in calibration families',
                'interpretation':'Internal computation influence only; predictor remains intact for deployment; diagnostic case selection is not prevalence estimation',
                'readout_sha256':sha256(out/'lewm_audit/readouts/projected_cls_ridge.joblib')}
        write_json(lock,design)
    layers=model.model.predictor.transformer.layers
    for ci,spec in enumerate(design['development']+design['calibration']):
        key=f"{spec['state_id']}_i{spec['index']:03d}";done=dest/f'{key}.json'
        if done.exists():continue
        started=time.perf_counter();predict,inputs,z,acts,real,truth=capture(model,out,spec)
        nominal_pose=decode(probe,z[0].cpu().numpy()[None])[0]
        rows=[];rng=np.random.default_rng(cfg['seed']+ci)
        for layer in range(6):
          targeted=np.array(design['channels'][str(layer)])
          for receiver in [0,1]:
            donor=1-receiver;base=acts[receiver][layer];other=acts[donor][layer]
            donor_delta=other-base;target_norm=float(torch.linalg.vector_norm(donor_delta[0,-1,targeted]))
            ablate_norm=float(torch.linalg.vector_norm(base[0,-1,targeted]))
            for dose in design['doses']:
              variants=[('target_patch',targeted,0),('target_ablation',targeted,0),('context_mean',targeted,0),('sham',targeted,0)]
              for repeat in range(design['random_repeats']):
                  indices=rng.choice(192,16,replace=False)
                  variants.extend([('random_patch',indices,repeat),('random_ablation',indices,repeat)])
              for kind,indices,repeat in variants:
                delta=torch.zeros_like(base)
                if kind in ['target_patch','random_patch']:
                    values=donor_delta[0,-1,indices].clone()
                    if kind=='random_patch':values*=target_norm/torch.linalg.vector_norm(values).clamp_min(1e-12)
                elif kind in ['target_ablation','random_ablation']:
                    values=-base[0,-1,indices].clone()
                    if kind=='random_ablation':values*=ablate_norm/torch.linalg.vector_norm(values).clamp_min(1e-12)
                elif kind=='context_mean':values=base[0,:-1,indices].mean(0)-base[0,-1,indices]
                else:values=torch.zeros(16,device=model.device)
                delta[0,-1,indices]=dose*values
                def hook(module,inputs,output):return output+delta
                handle=layers[layer].register_forward_hook(hook)
                try:
                    with torch.no_grad():patched=predict(inputs[receiver])
                finally:handle.remove()
                if kind=='sham':torch.testing.assert_close(patched,z[receiver],rtol=0,atol=0)
                pose=decode(probe,patched.cpu().numpy()[None])[0]
                response=pose-nominal_pose;response[2]=np.arctan2(np.sin(response[2]),np.cos(response[2]))
                expected=truth if receiver else np.zeros(3)
                pe,re=errors(response,expected)
                shift=patched-z[receiver];toward=z[donor]-z[receiver]
                rows.append({**spec,'layer':layer,'receiver':'modified' if receiver else 'nominal','kind':kind,'dose':dose,'repeat':repeat,
                             'intervention_norm':float(torch.linalg.vector_norm(delta)),
                             'latent_shift':float(torch.linalg.vector_norm(shift)),
                             'donor_projection_fraction':float(torch.dot(shift,toward)/torch.dot(toward,toward).clamp_min(1e-12)),
                             'latent_prediction_error':float(np.linalg.norm(patched.cpu().numpy()-real[receiver])),
                             'baseline_latent_error':float(np.linalg.norm(z[receiver].cpu().numpy()-real[receiver])),
                             'object_effect_position_error':float(pe),'object_effect_rotation_error_deg':float(np.rad2deg(re))})
        pd.DataFrame(rows).to_csv(dest/f'{key}.csv',index=False)
        write_json(done,{'case':spec,'rows':len(rows),'duration_seconds':time.perf_counter()-started,'source_sha256':sha256(Path(__file__))})
        print(key,flush=True)
    allrows=pd.concat([pd.read_csv(p.with_suffix('.csv')) for p in dest.glob('*_i[0-9][0-9][0-9].json')],ignore_index=True)
    allrows.to_csv(dest/'candidates.csv',index=False)
    allrows.groupby(['split','layer','receiver','kind','dose']).agg(n=('index','size'),latent_shift=('latent_shift','mean'),donor_projection=('donor_projection_fraction','mean'),position_error=('object_effect_position_error','mean'),rotation_error=('object_effect_rotation_error_deg','mean'),global_error=('latent_prediction_error','mean')).reset_index().to_csv(dest/'summary.csv',index=False)
    write_json(dest/'completion.json',{'development_cases':len(design['development']),'calibration_cases':len(design['calibration']),'rows':len(allrows),'selection_sha256':sha256(lock),'shortfalls':design['shortfalls']})

if __name__=='__main__':main()
