"""Measure complete Cube refinement pipelines; reuse only exactly matched physics."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from zj_wam.io import resolve,write_json,sha256
from scripts.run_followup_cube import Model
from scripts.cube_runtime_core import pipeline

BASE=resolve('artifacts/followup_v1/cube')
read=lambda p:json.loads(p.read_text())


def prepare(smoke):
    cfg=read(resolve('configs/followup_cube.json'));rcfg=read(resolve('configs/followup_cube_runtime.json'));parent=read(BASE/'manifest.json');out=BASE/('runtime_smoke' if smoke else 'runtime');out.mkdir(exist_ok=True)
    paths=['configs/followup_cube_runtime.json','configs/followup_cube.json','scripts/cube_runtime_core.py','scripts/run_followup_cube_runtime.py','scripts/analyze_followup_cube_runtime.py','scripts/run_followup_cube.py','scripts/analyze_followup_cube.py','scripts/analyze_followup_response_selection.py','zj_wam/cube_experiment.py','zj_wam/cube_execution.py','zj_wam/diagnostics.py','zj_wam/io.py']
    files=[resolve(p) for p in paths]+[BASE/p for p in ['manifest.json','raw_audit.json','calibrated_settings.json','proposal_ablation/analysis.json','evaluation.csv','proposal_ablation/evaluation.csv']]
    record={'config':rcfg,'cube_config':cfg,'smoke':smoke,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'checkpoint_sha256':sha256(resolve(cfg['checkpoint'])/'weights.pt'),'normalization':parent['normalization'],'splits':parent['splits']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==record,'Runtime sources changed: explicitly version results'
    else:write_json(out/'manifest.json',record)
    return cfg,rcfg,record,out


def process(path,cfg,rcfg,manifest,out,model,mean,scale,rules):
    meta=read(path);folder=out/meta['split'];folder.mkdir(exist_ok=True);dest=folder/path.name
    if dest.exists():assert read(dest)['manifest_sha256']==sha256(out/'manifest.json');return
    raw=pd.read_csv(path.with_suffix('.csv'));look={(r.method,float(r.radius),int(r.sign)):int(r.candidate) for r in raw.itertuples()}
    assert len(look)==len(raw)
    with np.load(path.with_suffix('.npz')) as a:
        context={key:a[key].copy() for key in ['nominal_actions','suffix_actions','past_points','past_targets','workspace_bounds','history_images','past_actions','root_gripper_contact']}
        parent={key:a[key].copy() for key in ['actions','predicted_robot','response_gains']}
    variants=[(profile,method) for profile in rcfg['profiles'] for method in rcfg['methods']];order=[(p,m,k) for k in range(rcfg['repetitions']) for p,m in variants];rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],meta['episode'],meta['timestep'],909]));rng.shuffle(order)
    rows=[];saved={};errors={};first={}
    def check(name,a,b,tol):
        e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));errors[name]=max(errors.get(name,0.),e);assert e<=tol,(meta['state_id'],name,e)
    for profile_name,method,repetition in order:
        profile=rcfg['profiles'][profile_name];rule=rules[profile['bank']][method]
        result=pipeline(context,meta,cfg,profile,method,rule,model,mean,scale)
        ids=np.array([look[(name,float(radius),int(sign))] for name,radius,sign in result['labels']]);assert len(set(ids))==len(ids)
        expected=raw[(raw.method=='nominal')|((raw.radius.isin(profile['radii'])) & (raw.method.isin(['random0','random1','past_pd_optimizer']) if profile['bank']=='non_wm_proposals' else raw.method!='nominal'))]
        np.testing.assert_array_equal(ids,expected.candidate.to_numpy())
        check('parent_actions',result['actions'],parent['actions'][ids],1e-9);check('parent_normalized_norm',result['norm'],raw.norm.to_numpy()[ids],1e-8);check('parent_response_fit',result['gains'],parent['response_gains'],1e-9)
        costs=raw.predicted_jerk_cost.to_numpy()[ids];check('parent_relative_cost',result['costs']/np.maximum(1,costs),costs/np.maximum(1,costs),1e-10)
        if method=='lewm_global':score=raw.lewm_score.to_numpy()[ids]
        else:
            pred=parent['predicted_robot'][ids];d=pred-pred[:1];pos=np.linalg.norm(d[:,:,:3],axis=-1);angle=np.abs(np.arctan2(np.sin(d[:,:,3]),np.cos(d[:,:,3])));gate=float(context['root_gripper_contact'])>.1 or np.any(np.r_[context['nominal_actions'][:,4],context['suffix_actions'][:,4]]>.1)
            score=np.maximum(pos.max(1)/cfg['position_tolerance_m'],angle.max(1)/cfg['rotation_tolerance_rad']) if method=='controller_displacement' else np.maximum((pos+np.sqrt(3)*.02*angle).max(1)/cfg['position_tolerance_m'],angle.max(1)/cfg['rotation_tolerance_rad']) if gate else np.zeros(len(ids))
        check('parent_selector_score',result['scores'],score,1e-6)
        nominal=raw.iloc[0];eligible=(raw.norm.to_numpy()[ids]>1e-6)&(raw.norm.to_numpy()[ids]<=rule.get('radius',.4)+1e-6)&(score<=rule.get('tau',-np.inf))&(costs<nominal.predicted_jerk_cost-1e-10)
        if rule.get('disabled',False):eligible[:]=False
        candidates=np.flatnonzero(eligible);old=int(ids[candidates[np.lexsort((ids[candidates],costs[candidates]))[0]]]) if len(candidates) else 0
        selected=int(ids[result['selected']]);assert selected==old,('changed choice',selected,old)
        if profile_name in ['nonwm_full','wm_full']:
            reference=BASE/('proposal_ablation/evaluation.csv' if profile_name=='nonwm_full' else 'evaluation.csv');f=pd.read_csv(reference);f=f[(f.split==meta['split'])&(f.state_id==meta['state_id'])&(f.method==method)]
            if profile_name=='nonwm_full':f=f[f.bank=='non_wm_proposals']
            assert len(f)==1 and int(f.iloc[0].candidate)==selected,('old full selection',profile_name,method,meta['state_id'])
        key=profile_name+'__'+method
        if key in first:assert first[key]==selected
        else:
            first[key]=selected;saved[key+'_candidate_ids']=ids;saved[key+'_actions']=result['actions'];saved[key+'_scores']=result['scores'];saved[key+'_costs']=result['costs'];saved[key+'_norm']=result['norm']
        chosen=raw.iloc[selected]
        rows.append({'state_id':meta['state_id'],'initial_pose_family':meta['episode'],'phase':meta['phase'],'split':meta['split'],'profile':profile_name,'method':method,'repetition':repetition,'seconds':result['seconds'],'candidate':selected,'candidate_count':len(ids),'encoder_calls':result['encoder_calls'],'jacobian_calls':result['jacobian_calls'],'finite_prediction_calls':result['finite_prediction_calls'],'norm':float(chosen.norm),'active':bool(chosen.norm>1e-6),'object_error':float(chosen.object_error),'jerk_gain':float(chosen.jerk_gain),'joint':bool(chosen.norm>1e-6 and chosen.object_error<=1 and chosen.jerk_gain>=cfg['jerk_gain_threshold'])})
    pd.DataFrame(rows).to_csv(dest.with_suffix('.csv'),index=False);np.savez_compressed(dest.with_suffix('.npz'),**saved)
    write_json(dest,{'state_id':meta['state_id'],'split':meta['split'],'timing_rows':len(rows),'variants':len(variants),'max_parent_error':errors,'parent_sha256':{str(path.with_suffix(ext).relative_to(resolve('.'))):sha256(path.with_suffix(ext)) for ext in ['.json','.csv','.npz']},'manifest_sha256':sha256(out/'manifest.json'),'scope':'All decisions made before reading physical outcome columns. Existing raw physical outcome reused only for matching delivered action; timings freshly executed.'})
    print(meta['split'],meta['state_id'],'timings',len(rows),'max_action_error',errors['parent_actions'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--dependency-pid',type=int);args=parser.parse_args()
    if args.dependency_pid:
        from scripts.run_followup_policy_final import complete_calibration
        start=time.monotonic();print('WAITING for completed policy calibration timing acquisition',flush=True)
        while not complete_calibration():
            try:os.kill(args.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError('Calibration dependency ended before full audit; inspect failure')
            if time.monotonic()-start>21600:raise RuntimeError('Observation window elapsed; revalidate parent')
            time.sleep(10)
    torch.set_num_threads(4);cfg,rcfg,manifest,out=prepare(args.smoke);mean=np.asarray(manifest['normalization']['mean']);scale=np.asarray(manifest['normalization']['scale']);model=Model(cfg,mean,scale)
    rules={'common':read(BASE/'calibrated_settings.json')['rules'],'non_wm_proposals':read(BASE/'proposal_ablation/analysis.json')['settings']['non_wm_proposals']['rules']}
    # Fixed warm-up before measurement, excluded and common to both splits.
    first=sorted((BASE/'calibration').glob('e*.npz'))[0]
    with np.load(first) as a:
        _,predict=model.predictor(a['history_images'],a['past_actions']);x=torch.tensor(((a['nominal_actions']-mean)/scale).reshape(-1),dtype=torch.float32,device=cfg['device']);torch.autograd.functional.jacobian(predict,x,vectorize=True)
    for split in ['calibration','development']:
        paths=sorted((BASE/split).glob('e*.json'));assert len(paths)==96
        for path in paths[:1] if args.smoke else paths:process(path,cfg,rcfg,manifest,out,model,mean,scale,rules)
    for path,h in manifest['inputs'].items():assert sha256(resolve(path))==h,path
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'states':2 if args.smoke else 192,'manifest_sha256':sha256(out/'manifest.json')})
if __name__=='__main__':main()
