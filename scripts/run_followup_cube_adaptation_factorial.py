"""Measure actual Cube factorial selectors; reuse only exactly matched physics labels."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from scripts.cube_adaptation_factorial_core import decide
from scripts.output_replacement_core import physical
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main(split):
    acfg=read(resolve('configs/followup_cube_adaptation_factorial.json'));cal=BASE/'cube_adaptation_calibration';settings=read(cal/'settings.json');ca=read(cal/'raw_audit.json');assert ca['settings_sha256']==sha256(cal/'settings.json') and settings['qualified'] and settings['constant_radius_qualified'];bank=BASE/('cube_policy_calibration_bank' if split=='nominal' else 'cube_shift_candidate_bank');ba=read(bank/'raw_audit.json');manifest=read(bank/'manifest.json')
    for p,h in {**read(cal/'protocol.json')['inputs'],**manifest['inputs'],**ba['inputs']}.items():assert sha256(resolve(p))==h,p
    if split=='nominal':
        assert ba['states']==138;paths=[bank/f'{seed}_{phase}_t{t:03d}.json' for seed,phases in manifest['phases'].items() for phase,t in phases.items() if t is not None]
    else:
        assert ba['contexts']==336 and read(bank/'pipeline_complete.json')['raw_audit_sha256']==sha256(bank/'raw_audit.json');paths=[bank/read(resolve(p))['condition']/Path(p).name for p in manifest['sources']]
    cfg={**read(BASE/'cube_policy_refinement/manifest.json')['cube'],'device':'cpu','yaw_length_scale_m':0.,'radii':read(resolve('configs/followup_cube_policy_calibration_bank.json'))['radii']};out=BASE/'cube_adaptation_factorial'/split;out.mkdir(parents=True,exist_ok=True);files=[resolve(p) for p in ['configs/followup_cube_adaptation_factorial.json','scripts/run_followup_cube_adaptation_factorial.py','scripts/cube_adaptation_factorial_core.py','scripts/output_replacement_core.py','zj_wam/cube_experiment.py']]+[cal/'protocol.json',cal/'settings.json',cal/'raw_audit.json',bank/'manifest.json',bank/'raw_audit.json',BASE/'cube_policy_refinement/manifest.json']
    protocol={'split':split,'settings':settings,'config':acfg,'sources':[str(p.relative_to(resolve('.'))) for p in paths],'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':'Actual per-state per-arm staged generation/scoring/timing after context available. Every generated candidate must exactlymatch audited sourcebank action, scores andexpectedmotion before its physical label is reused. No newphysics/RGB inthisselectorcost stage. SharedCPU walltime, one rotated-order run perstate/arm; not an actualdelay or isolateddeadline guarantee. Nominal calibration results resubstitution; development/shifted separate.'}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write_json(out/'protocol.json',protocol)
    count=0
    for state_index,path in enumerate(paths):
        meta=read(path);condition='nominal' if split=='nominal' else meta['condition'];labelsplit=meta.get('split','development');folder=out/condition;folder.mkdir(exist_ok=True);dest=folder/path.name
        if dest.exists():m=read(dest);assert m['protocol_sha256']==sha256(out/'protocol.json') and m['data_sha256']==sha256(dest.with_suffix('.npz'));count+=len(acfg['methods']);continue
        raw=dict(np.load(path.with_suffix('.npz')));assert meta['data_sha256']==sha256(path.with_suffix('.npz'));context={k[6:]:v for k,v in raw.items() if k.startswith('input_')};labels={tuple(x):i for i,x in enumerate(raw['labels'])};modes=acfg['methods'];offset=state_index%len(modes);order=modes[offset:]+modes[:offset];arrays={};decisions={};records=[]
        for mode in order:
            bank0,d=decide(context,{'episode':meta['seed'],'timestep':meta['step']},cfg,acfg,settings['base_rule'],raw['mean'],raw['scale'],mode,settings['alpha'],settings['constant_radius']);ids=np.array([labels[tuple(x)] for x in bank0['labels']]);assert len(set(ids))==len(ids) and ids[0]==0
            for key in ['actions','costs','norm','predicted_robot','coefficients']:np.testing.assert_array_equal(bank0[key],raw[key][ids])
            np.testing.assert_array_equal(bank0['scores'],raw['proprio_contact_scores'][ids])
            if d['stages']==2:np.testing.assert_array_equal(ids,np.arange(len(raw['actions'])))
            j=int(ids[d['selected']]);decisions[mode]={**d,'original_candidate':j,'source_candidate_indices':ids.tolist()};arrays.update({mode+'__'+k:v for k,v in bank0.items()});count+=1
        # Outcomes are consulted only after every actual arm decision is finalized.
        truth=physical('cube',raw)
        for mode,d in decisions.items():
            j=d['original_candidate'];active=bool(raw['norm'][j]>1e-6);E=float(truth['dense_full'][j]);gain=float(truth['jerk_gain'][j]);records.append({'state_id':path.stem,'initial_pose_family':meta['seed'],'phase':meta['phase'],'split':labelsplit,'condition':condition,'mode':mode,'candidate':j,'active':active,'violation':active and E>1,'joint':active and E<=1 and gain>=.1,'object_error':E,'jerk_gain':gain,**{k:v for k,v in d.items() if k in ['seconds','radius','alpha','past_error','stages','early_stop','scored_candidates','solver_calls','solver_iterations','predicted_gain']},**{'time_'+k:v for k,v in d['timing'].items()}})
        np.savez_compressed(dest.with_suffix('.npz'),**arrays);pd.DataFrame(records).to_csv(dest.with_suffix('.csv'),index=False);write_json(dest,{'source':str(path.relative_to(resolve('.'))),'source_data_sha256':sha256(path.with_suffix('.npz')),'execution_order':order,'decisions':decisions,'protocol_sha256':sha256(out/'protocol.json'),'data_sha256':sha256(dest.with_suffix('.npz'))});print('CUBE FACTORIAL',labelsplit,condition,path.stem,count,flush=True)
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    assert count==len(paths)*6;write_json(out/'execution_complete.json',{'split':split,'states':len(paths),'actual_selector_calls':count,'protocol_sha256':sha256(out/'protocol.json'),'scope':protocol['scope']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=['nominal','shifted']);a=p.parse_args()
    with threadpool_limits(limits=2,user_api='blas'):main(a.split)
