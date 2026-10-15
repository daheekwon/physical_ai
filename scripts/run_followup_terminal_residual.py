"""Extend existing boundary-constrained candidate physics without changing choices."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
from zj_wam.adapters import ExpertDataset
from zj_wam.followup import DenseSimulator,dense_metrics,optimize_controller

BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def prepare(smoke):
    cfg=read(resolve('configs/followup_v1.json'));tcfg=read(resolve('configs/followup_terminal_residual.json'));out=BASE/('terminal_residual_smoke' if smoke else 'terminal_residual');out.mkdir(exist_ok=True)
    files=[resolve(p) for p in ['configs/followup_terminal_residual.json','configs/followup_v1.json','configs/followup_family_ablation.json','scripts/run_followup_terminal_residual.py','scripts/analyze_followup_terminal_residual.py','scripts/run_followup_family_ablation.py','scripts/analyze_followup_final_expert.py','zj_wam/followup.py','zj_wam/adapters.py','zj_wam/correction_families.py','zj_wam/diagnostics.py']]
    manifest={'smoke':smoke,'config':tcfg,'main':cfg,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'created_for':'New long-suffix diagnostic, same old dev/cal candidate actions; no test alteration'}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    return cfg,tcfg,out,manifest


def process(path,cfg,tcfg,out,data):
    meta=read(path);folder=out/meta['split'];folder.mkdir(exist_ok=True);dest=folder/path.name
    if dest.exists():assert read(dest)['manifest_sha256']==sha256(out/'manifest.json');return
    start=time.perf_counter();a,s=data.episode(meta['episode']);t=meta['timestep'];h=tcfg['horizon'];assert len(a)>=t+tcfg['total_actions'];nominal=a[t:t+h];short_suffix=a[t+h:t+h+tcfg['original_suffix']];suffix=a[t+h:t+tcfg['total_actions']]
    source=pd.read_csv(path.with_suffix('.csv'));source=source[source.method.isin(tcfg['families'])].copy();assert len(source)==24 and set(source.relative_radius)==set(tcfg['relative_radii'])
    with np.load(path.with_suffix('.npz')) as raw:old={k:raw[k].copy() for k in raw.files if k in ['actions','physics','root'] or k.startswith('basis_')}
    sim=DenseSimulator(a,s,t,cfg['seed']+meta['episode']);reference=sim.execute_dense(nominal,suffix);prefix=sim.prefix_physics.copy();root=reference['root_full_state'];np.testing.assert_array_equal(reference['physics_trace'][:110],old['physics'][0]);np.testing.assert_array_equal(root,old['root'])
    records=[];traces=[reference['physics_trace']];actions=[nominal.copy()];coverage=[[reference['coverage_t'],reference['coverage_suffix']]];errors={};dtypes=[]
    for row in source.itertuples():
        basis=old['basis_'+row.method]
        if basis.shape[1]:chosen,info=optimize_controller(nominal,short_suffix,root,prefix,data.scaler.scale_,row.relative_radius*np.sqrt(2*h),basis=basis)
        else:chosen=nominal.copy();info={'solver_success':True}
        error=float(np.max(np.abs(chosen-old['actions'][row.index])));errors['regenerated_action']=max(errors.get('regenerated_action',0.),error);assert error<1e-9,(row.method,error)
        result=sim.execute_dense(chosen,suffix);err=float(np.max(np.abs(result['physics_trace'][:110]-old['physics'][row.index])));errors['old_short_physics']=max(errors.get('old_short_physics',0.),err);assert err<=1e-8,(meta['state_id'],row.method,err,chosen.dtype)
        metrics=dense_metrics(result,reference,cfg['tolerances']);nested=max(float(row.object_error),metrics['object_error']);d=result['physics']-reference['physics'];terminal=h*10-1;pq=float(np.linalg.norm(d[terminal,:2]));pv=float(np.linalg.norm(d[terminal,5:7]))
        if row.method.endswith('_terminal'):assert max(pq,pv)<=1e-3
        pose=np.maximum(np.linalg.norm(d[:,2:4],axis=1)/cfg['tolerances']['position'],np.abs(np.arctan2(np.sin(d[:,4]),np.cos(d[:,4])))/cfg['tolerances']['rotation']);violations=np.flatnonzero(pose>1)
        rec={'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'episode':meta['episode'],'split':meta['split'],'method':row.method,'relative_radius':row.relative_radius,'dimension':int(row.dimension),'index':len(traces),'parent_index':int(row.index),'norm':float(row.actual_norm),'short_object_error':float(row.object_error),'short_jerk_gain':float(row.jerk_gain),'long_object_error':nested,'long_jerk_gain':metrics['jerk_gain'],'active':bool(row.actual_norm>1e-6),'terminal_pusher_position':pq,'terminal_pusher_velocity':pv,'post_correction_pusher_max':float(np.linalg.norm(d[h*10:,:2],axis=1).max()),'first_pose_violation_seconds':float((violations[0]+1)*.01) if len(violations) else None,'coverage_long_loss':metrics['coverage_suffix_loss']}
        for step in [5,8,23]:
            x=d[step*10-1];rec[f'object_position_{step}']=float(np.linalg.norm(x[2:4]));rec[f'object_velocity_{step}']=float(np.linalg.norm(x[7:9]));rec[f'object_angular_velocity_{step}']=float(abs(x[9]))
        records.append(rec);traces.append(result['physics_trace']);actions.append(chosen);coverage.append([result['coverage_t'],result['coverage_suffix']]);dtypes.append(str(chosen.dtype))
    sim.close();pd.DataFrame(records).to_csv(dest.with_suffix('.csv'),index=False);np.savez_compressed(dest.with_suffix('.npz'),physics=traces,actions=actions,suffix=suffix,coverage=coverage,root=root)
    write_json(dest,{**{k:meta[k] for k in ['state_id','split','episode','timestep','initial_pose_family']},'candidates':len(records),'total_actions':tcfg['total_actions'],'action_dtypes':dtypes,'max_error':errors,'parent_sha256':{str(path.with_suffix(ext).relative_to(resolve('.'))):sha256(path.with_suffix(ext)) for ext in ['.json','.csv','.npz']},'manifest_sha256':sha256(out/'manifest.json'),'seconds':time.perf_counter()-start})
    print(meta['state_id'],'24longbranches','seconds',round(time.perf_counter()-start,2),'oldtraceerror',errors['old_short_physics'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();cfg,tcfg,out,m=prepare(args.smoke);data=ExpertDataset(cfg['dataset'])
    for split in tcfg['splits']:
        paths=sorted((BASE/'family_ablation').glob(split+'*_h05.json'));assert len(paths)==105
        for path in paths[:1] if args.smoke else paths:process(path,cfg,tcfg,out,data)
    for path,h in m['inputs'].items():assert sha256(resolve(path))==h,path
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'states':2 if args.smoke else 210,'manifest_sha256':sha256(out/'manifest.json')})
if __name__=='__main__':main()
