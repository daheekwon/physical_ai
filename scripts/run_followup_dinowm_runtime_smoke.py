"""Validate fresh CPU candidates/scores against the common verifier bank."""
import json,sys,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits,threadpool_info
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from scripts.run_followup_cube import Model
from scripts.model_runtime_imports import load_dino
from scripts.dinowm_runtime_core import run
from scripts.lewm_runtime_reference import native_scores
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_runtime_smoke_v3';read=lambda p:json.loads(p.read_text())

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);maincfg=read(resolve('configs/followup_v1.json'));ccfg=read(resolve('configs/followup_cube.json'));rcfg=read(resolve('configs/followup_response_models.json'));data=ExpertDataset(maincfg['dataset']);source=read(BASE/'dinowm_interventions_smoke/manifest.json');rulemap={(x['environment'],x['method']):x['rule'] for x in read(BASE/'dinowm_selection/rules.json')['rules']};parent=pd.read_csv(BASE/'dinowm_selection/candidates.csv');selected=pd.read_csv(BASE/'dinowm_selection/selected.csv');rows=[];checks=[];inputs={p:sha256(resolve(p)) for p in ['scripts/dinowm_runtime_core.py','scripts/run_followup_dinowm_runtime_smoke.py','scripts/model_runtime_imports.py','scripts/lewm_runtime_reference.py']}
    for env,splits in source['states'].items():
        cfg=maincfg if env=='pusht' else {**ccfg,'device':'cpu'};norm=read(BASE/'cube/manifest.json')['normalization'];mean=data.scaler.mean_ if env=='pusht' else np.array(norm['mean']);scale=data.scaler.scale_ if env=='pusht' else np.array(norm['scale']);lewm=FrozenLeWM(maincfg['checkpoint'],data.scaler,'cpu') if env=='pusht' else Model(cfg,mean,scale);dino=load_dino(env)
        for split,names in splits.items():
            src=resolve(names[0]);meta=read(src);f=pd.read_csv(src.with_suffix('.csv'))
            with np.load(src.with_suffix('.npz'),allow_pickle=False) as z:a=dict(z)
            c={'episode':meta['episode'],'timestep':meta['timestep'],'images':a['history_images'],'past':a['past_actions'],'scale':scale,'mean':mean}
            if env=='pusht':
                original,recorded=data.episode(meta['episode']);t=meta['timestep'];root=np.zeros(10);root[[0,1,5,6]]=a['root'][[0,1,5,6]];prefix=np.zeros_like(a['physics'][0,:30]);prefix[:,[0,1,5,6]]=a['physics'][0,:30,[0,1,5,6]].T;proprio=np.zeros((3,10));proprio[:,:7]=recorded[[t-10,t-5,t]];proprio[-1,:2]=root[:2];proprio[-1,5:7]=root[5:7];proprio[:,[2,3,4,7,8,9]]=0;c.update(nominal=original[t:t+5],suffix=original[t+5:t+8],root=root,prefix=prefix,proprio=proprio)
                lookup={('nominal',0.,0):0}
                for r in f.itertuples():
                    label=r.method;sign=0
                    if '_sign' in label:label,sg=label.split('_sign');sign=int(sg)
                    lookup[(label,float(r.relative_radius),sign)]=int(r.index)
            else:
                c.update(nominal=a['nominal_actions'],suffix=a['suffix_actions'],points=a['past_points'],targets=a['past_targets'],bounds=a['workspace_bounds'],gripper_contact=float(a['root_gripper_contact']));lookup={(r.method,float(r.radius),int(r.sign)):int(r.candidate) for r in f.itertuples()}
            for method in ['controller','contact','lewm_global','dino_patch_rms','dino_patch_mean']:
                result=run(env,c,cfg,method,rulemap[(env,method)],lewm if method=='lewm_global' else None,dino if method.startswith('dino') else None,rcfg);ids=np.array([lookup[tuple(label)] for label in result['labels']]);parentactions=a['actions'][ids];error=float(abs(result['actions']-parentactions).max());exact=bool(np.array_equal(result['actions'],parentactions));normerror=float(abs(result['norms']-np.linalg.norm(((parentactions-parentactions[:1])/scale).reshape(len(ids),-1),axis=1)).max());subset=parent[(parent.environment==env)&(parent.state_id==meta['state_id'])];indexmap={int(k):i for i,k in enumerate(ids)};usable=subset.candidate.to_numpy(int);scores=result['scores'][[indexmap[int(k)] for k in usable]];scoreerr=float(abs(scores-subset[method].to_numpy()).max()) if len(usable) else 0.;picked=int(ids[result['selected']]);old=int(selected[(selected.environment==env)&(selected.state_id==meta['state_id'])&(selected.method==method)].candidate.iloc[0]);check={'environment':env,'split':split,'state_id':meta['state_id'],'method':method,'actions_exact':exact,'max_action_error':error,'max_norm_error':normerror,'max_score_error':scoreerr,'selected_candidate':picked,'parent_candidate':old,'selected_matches_parent':picked==old};checks.append(check);assert error<1e-9,(check,'freshcandidate mismatch');assert method=='lewm_global' or scoreerr<1e-8,(check,'score mismatch')
                if method=='lewm_global':
                    reference=native_scores(env,lewm,c,result['actions']);cpuerror=float(abs(reference-result['scores']).max());check['max_native_cpu_score_error']=cpuerror;assert cpuerror<2e-5,(check,'native CPU score mismatch')
                row={**check,**{k:v for k,v in result.items() if k.endswith('_seconds') or k.endswith('_calls')},'candidates':len(ids),'optimizer_calls':len(result['solver']),'solver_failures':sum(not x['success'] for x in result['solver'])};rows.append(row);stem=env+'__'+src.stem+'__'+method;np.savez_compressed(OUT/(stem+'.npz'),candidate_ids=ids,actions=result['actions'],scores=result['scores'],costs=result['costs'],norms=result['norms']);write_json(OUT/(stem+'.json'),{'row':row,'solver':result['solver'],'labels':result['labels']});print('SMOKE',env,split,method,'seconds',round(result['total_seconds'],3),'exact',exact,'choice',picked,old,flush=True)
        del lewm,dino
    pd.DataFrame(rows).to_csv(OUT/'timings.csv',index=False);write_json(OUT/'execution_complete.json',{'states':4,'variants':20,'inputs':inputs,'checks':checks,'threads':2,'threadpools':threadpool_info(),'device':'cpu','platform':platform.platform(),'scope':'First technical smoke only; timings include firstcall overhead, not warmed cost estimates. Fresh generation/encoding/prediction/scoring/selection from causalRGB/proprio buffers, excludesrender/acquisition/model loading/physics. No fresh physical outcomes yet for changed choices/actions. Fullruntime protocol and independent audit pending.'});print('RUNTIME SMOKE COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2,user_api='blas'):main()
