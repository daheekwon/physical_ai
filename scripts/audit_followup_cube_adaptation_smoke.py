"""Audit actual staged selector against the same frozen Cube candidate fixtures."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from scripts.cube_adaptation_factorial_core import decide,past_error
from scripts.output_replacement_core import physical
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main():
    acfg=read(resolve('configs/followup_cube_adaptation_factorial.json'));parent=BASE/'cube_policy_calibration_bank_smoke';pa=read(parent/'raw_audit.json');assert pa['states']==8 and pa['candidates']==163;manifest=read(parent/'manifest.json');paths=[parent/f'{seed}_{phase}_t{step:03d}.json' for seed,phases in manifest['phases'].items() for phase,step in phases.items() if step is not None];cfg={**read(BASE/'cube_policy_refinement/manifest.json')['cube'],'device':'cpu','yaw_length_scale_m':0.,'radii':manifest['config']['radii']};rule=next(r['rule'] for r in read(BASE/'cube_policy_selection/rules.json')['rules'] if r['method']==acfg['verifier']);out=BASE/'cube_adaptation_smoke';out.mkdir(exist_ok=True);files=[resolve(p) for p in ['configs/followup_cube_adaptation_factorial.json','scripts/cube_adaptation_factorial_core.py','scripts/audit_followup_cube_adaptation_smoke.py','scripts/output_replacement_core.py','zj_wam/cube_experiment.py']]+[parent/'manifest.json',parent/'raw_audit.json',BASE/'cube_policy_selection/rules.json',BASE/'cube_policy_selection/raw_audit.json']
    for p in paths:files.extend([p,p.with_suffix('.npz')])
    protocol={'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'config':acfg,'scope':'Technical actual-staging smoke on8already auditedcal/dev states. Fixed and non-early arms reproduceoriginalfullbank; cheap-only isoptimizer subset. Reuse existing physicalcandidateoutcomes, no newphysics or calibratedadaptationeffect claim. Alpha0.5/radius0.1 are technical smoke settings only.'}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write_json(out/'protocol.json',protocol)
    errors={};rows=[];outputs={};calls=0;full=0;early=0
    def check(name,x,y,tol=0,rtol=0):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,(name,x.shape,y.shape);np.testing.assert_allclose(x,y,atol=tol,rtol=rtol,err_msg=name);errors[name]=max(errors.get(name,0.),float(abs(x.astype(float)-y.astype(float)).max()) if x.size else 0.)
    for path in paths:
        m=read(path);raw=dict(np.load(path.with_suffix('.npz')));context={k[6:]:v for k,v in raw.items() if k.startswith('input_')};truth=physical('cube',raw);error,components,pred,gains=past_error(context);p=context['past_points'];targets=context['past_targets'];state=p[125].copy();d=p[125]-p[124];ind=[]
        for target in targets[125:]:
            e=target-state;e[3]=np.angle(np.exp(1j*e[3]));d=(1-.002*gains[:,1])*d+.002**2*gains[:,0]*e;state=state+d;ind.append(state.copy())
        check('independent_past_prediction',pred,np.array(ind),1e-10);delta=np.array(ind)-p[126:];parts=np.array([np.sqrt(np.sum(delta[:,:3]**2,axis=1)).max()/.002,np.abs(np.angle(np.exp(1j*delta[:,3]))).max()/np.deg2rad(2)]);check('independent_past_residual',parts,components,1e-8);assert abs(error-parts.max())<1e-8;labels={tuple(x):i for i,x in enumerate(raw['labels'])};assert len(labels)==len(raw['actions'])
        for mode in acfg['methods']:
            bank,decision=decide(context,{'episode':m['seed'],'timestep':m['step']},cfg,acfg,rule,raw['mean'],raw['scale'],mode,alpha=.5,constant_radius=.1);ids=np.array([labels[tuple(x)] for x in bank['labels']]);assert ids[0]==0 and len(set(ids))==len(ids)
            for key,tol,rtol in [('actions',0,0),('coefficients',1e-9,0),('costs',1e-2,1e-8),('norm',1e-8,0),('predicted_robot',1e-10,0)]:check('original_'+key,bank[key],raw[key][ids],tol,rtol)
            check('original_scores',bank['scores'],raw['proprio_contact_scores'][ids],1e-8)
            if decision['stages']==2:np.testing.assert_array_equal(ids,np.arange(len(raw['actions'])));full+=1
            else:assert all(x[0] in ['nominal','past_pd_optimizer'] for x in bank['labels'])
            expected_radius=.4/(1+.5*min(error,10)) if mode in ['allowance','both'] else (.1 if mode=='constant_shrink' else .4);check('radius',decision['radius'],expected_radius)
            eligible=[i for i in range(len(ids)) if bank['norm'][i]>1e-6 and bank['norm'][i]<=expected_radius+1e-6 and bank['scores'][i]<=rule['tau'] and (bank['costs'][0]-bank['costs'][i])/max(bank['costs'][0],1e-12)>1e-8] if rule['enabled'] else [];chosen=min(eligible,key=lambda i:(bank['costs'][i],bank['norm'][i],int(ids[i]))) if eligible else 0;assert chosen==decision['selected'];assert decision['scored_candidates']==len(ids) and decision['solver_calls']==5;early+=decision['early_stop'];j=int(ids[chosen]);row={'state_id':path.stem,'initial_pose_family':m['seed'],'phase':m['phase'],'mode':mode,'candidate':j,'radius':expected_radius,'past_error':error,'object_error':float(truth['dense_full'][j]),'jerk_gain':float(truth['jerk_gain'][j]),'seconds':decision['seconds'],'scored_candidates':decision['scored_candidates'],'solver_calls':5,'stages':decision['stages'],'early_stop':decision['early_stop']};rows.append(row);dest=out/f'{path.stem}_{mode}.npz';np.savez_compressed(dest,**bank,original_indices=ids);mp=dest.with_suffix('.json');write_json(mp,{'decision':decision,'source':str(path.relative_to(resolve('.'))),'data_sha256':sha256(dest)});outputs[str(dest.relative_to(resolve('.')))]=sha256(dest);outputs[str(mp.relative_to(resolve('.')))]=sha256(mp);calls+=1
        print('AUDIT CUBE ADAPTATION SMOKE',path.stem,calls,flush=True)
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    pd.DataFrame(rows).to_csv(out/'evaluation.csv',index=False);assert calls==48;write_json(out/'raw_audit.json',{'states':8,'actual_selector_calls':calls,'fullbank_exact_calls':full,'early_stop_calls':early,'max_errors':errors,'inputs':outputs,'protocol_sha256':sha256(out/'protocol.json'),'source_sha256':sha256(Path(__file__)),'scope':protocol['scope']});print('AUDITED CUBE ADAPTATION SMOKE',calls,full,early,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=2,user_api='blas'):main()
