"""F4/F5 native-block long-horizon diagnostics with causal context refresh."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch,joblib
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.model_audit import encode_representations,predict_candidates
from zj_wam.readouts import decode,effects,errors,history_features
from zj_wam.io import resolve,write_json,sha256

@torch.no_grad()
def rollout(adapter,context,past,actions,real,refresh=None,repeat_history=False,batch_size=64):
    result=[];histories=[]
    for start in range(0,len(actions),batch_size):
        raw=actions[start:start+batch_size];n,h,_=raw.shape
        act=torch.as_tensor(adapter.normalize(raw.reshape(-1,2)).reshape(n,-1,10),device=adapter.device)
        ae=adapter.model.action_encoder(torch.cat([past.expand(n,-1,-1),act],1))
        initial=context.expand(n,-1,-1)
        if repeat_history:initial=initial[:,-1:].expand(n,3,-1)
        emb=initial;pred=[];windows=[]
        observed=torch.cat([context.expand(n,-1,-1),torch.as_tensor(real[start:start+n],device=adapter.device)],1)
        for i in range(h//5):
            if refresh and i and i%refresh==0:
                # At block i, only observations through physical step 5*i exist.
                emb=observed[:,:3+i]
            end=3+i;begin=max(0,end-3)
            window=emb[:,begin:end]
            z=adapter.model.predict(window,ae[:,begin:end])[:,-1:]
            windows.append(torch.cat([window[:,-2:],z],1).reshape(n,1,-1))
            pred.append(z);emb=torch.cat([emb,z],1)
        result.append(torch.cat(pred,1).cpu().numpy());histories.append(torch.cat(windows,1).cpu().numpy())
    return np.concatenate(result),np.concatenate(histories)


def process(path,model,out,readouts,cfg):
    dest=out/'lewm_audit/long_horizon';dest.mkdir(parents=True,exist_ok=True)
    done=dest/path.with_suffix('.json').name
    if done.exists():return
    start=time.perf_counter();a=dict(np.load(path));f=pd.read_csv(path.with_suffix('.csv'));n,blocks=a['endpoint_images'].shape[:2];h=blocks*5
    _,context=model.predictor(a['history_images'],a['past_actions']);hist=context['embedding'].cpu().numpy()
    reps=encode_representations(model,a['endpoint_images'].reshape(-1,224,224,3));real=reps['projected_cls'].reshape(n,blocks,-1)
    prediction={};history={}
    for name,refresh,repeat in [('free',None,False),('refresh_5',1,False),('refresh_10',2,False),('repeat_initial_history',None,True)]:
        prediction[name],history[name]=rollout(model,context['embedding'],context['past'],a['actions'],real,refresh,repeat)
    reference=predict_candidates(model,context['embedding'],context['past'],a['actions'])
    equal=float(np.max(np.abs(reference-prediction['free'])))
    if equal>1e-5:raise ValueError(f'Free rollout mismatch {equal}')
    first_equal=float(np.max(np.abs(prediction['free'][:,0]-prediction['refresh_5'][:,0])))
    if first_equal>1e-5:raise ValueError('Refresh used future information at first block')
    prediction['real_encoded']=real;history['real_encoded']=history_features(hist,real)
    true=a['control'][:,3+np.arange(4,h,5),2:5];delta=effects(true);idx=f['index'].to_numpy(int)
    rows=[];saved={'truth':true,'real_native':real,'history_native':hist}
    for representation in ['projected_cls','history_projected']:
        probe=readouts[representation];features=prediction if representation=='projected_cls' else history
        for method,feature in features.items():
            pose=decode(probe,feature);effect=effects(pose);pe,re=errors(effect,delta);ape,are=errors(pose,true)
            saved[representation+'_'+method+'_pose']=pose
            for block in range(blocks):
                rows.append(pd.DataFrame({'state_id':f.state_id,'initial_pose_family':f.initial_pose_family,'horizon':h,'endpoint_step':(block+1)*5,
                                         'index':idx,'method':method,'representation':representation,'candidate_method':f.method,'relative_radius':f.relative_radius,
                                         'nominal_contact':f.nominal_contact,'preserved':f.preserved,'position_effect_error':pe[idx,block],'rotation_effect_error':re[idx,block],
                                         'position_absolute_error':ape[idx,block],'rotation_absolute_error':are[idx,block],
                                         'position_effect':np.linalg.norm(delta[idx,block,:2],axis=-1),'rotation_effect':np.abs(delta[idx,block,2]),
                                         'nominal_position_error':ape[0,block],'nominal_rotation_error':are[0,block]}))
        for block in range(blocks):
            rows.append(pd.DataFrame({'state_id':f.state_id,'initial_pose_family':f.initial_pose_family,'horizon':h,'endpoint_step':(block+1)*5,
                                     'index':idx,'method':'zero_effect','representation':representation,'candidate_method':f.method,'relative_radius':f.relative_radius,
                                     'nominal_contact':f.nominal_contact,'preserved':f.preserved,'position_effect_error':np.linalg.norm(delta[idx,block,:2],axis=-1),
                                     'rotation_effect_error':np.abs(delta[idx,block,2]),'position_effect':np.linalg.norm(delta[idx,block,:2],axis=-1),'rotation_effect':np.abs(delta[idx,block,2])}))
    pd.concat(rows,ignore_index=True).to_csv(done.with_suffix('.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**saved)
    write_json(done,{'state_id':str(f.state_id.iloc[0]),'split':str(f.split.iloc[0]),'horizon':h,'candidates':len(f),'blocks':blocks,
                     'free_reference_max_error':equal,'first_block_refresh_max_error':first_equal,'seconds':time.perf_counter()-start,
                     'scope':'Context refresh uses already executed real endpoints. Diagnostic upper-information condition, not open-loop action selection or a smoothing relinearization experiment.',
                     'source_sha256':sha256(Path(__file__))})
    print(done.stem,round(time.perf_counter()-start,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--horizon',type=int,choices=[5,10,20],required=True);p.add_argument('--limit',type=int);args=p.parse_args()
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());out=resolve(cfg['output']);torch.set_num_threads(cfg['threads'])
    data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
    readouts={name:joblib.load(out/f'lewm_audit/readouts/{name}_ridge.joblib') for name in ['projected_cls','history_projected']}
    paths=sorted((out/'f1').glob(f'development*_h{args.horizon:02d}.npz'))
    if len(paths)!=105:raise ValueError('Incomplete base bank')
    if args.limit:paths=paths[:args.limit]
    for path in paths:process(path,model,out,readouts,cfg)
if __name__=='__main__':main()
