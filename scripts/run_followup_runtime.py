"""Whole refinement pipeline timing and realized performance on paired expert states."""
import argparse,json,sys,time,platform,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from zj_wam.io import resolve,write_json,sha256
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import DenseSimulator,dense_metrics
from zj_wam.runtime_benchmark import measure,stages


def signature():return {p:sha256(resolve(p)) for p in ['configs/followup_runtime.json','zj_wam/runtime_benchmark.py','scripts/run_followup_runtime.py','zj_wam/policy_refinement.py','zj_wam/repeated_refinement.py','zj_wam/response_models.py','zj_wam/followup.py','artifacts/followup_v1/selection/selector_rules.json','artifacts/followup_v1/strong_baselines/calibrated_settings.json']}


def resources():
    try:return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).splitlines()
    except Exception as e:return [str(e)]


def variants(cfg):
    return [(m+'__'+profile,m,profile,radii) for m in cfg['methods'] for profile,radii in cfg['candidate_profiles'].items() if m!='gaussian_action' or profile=='full']


def run(path,cfg,rcfg,tcfg,data,model,rules,strong,out,sig,warmup):
    meta=json.loads(path.read_text());key=meta['state_id'];done=out/(key+'.json')
    if done.exists():assert json.loads(done.read_text())['signature']==sig;return
    original,recorded=data.episode(meta['episode']);t=meta['timestep'];sim=DenseSimulator(original,recorded,t,cfg['seed']+meta['episode']);images,past=sim.restore(capture=True);proprio=sim.context_control.copy();prefix=sim.prefix_physics.copy();nominal=original[t:t+8];seed=np.random.SeedSequence([cfg['seed'],meta['episode'],t,5]);settings=variants(tcfg);records={};selected={};snapshots={};start=time.perf_counter()
    if warmup:
        for name,method,profile,radii in settings:
            trial={**cfg,'radii':radii}
            for _ in range(tcfg['warmup_calls_per_variant']):measure(sim.env,model,method,images,past,proprio,prefix,nominal,data.scaler.scale_,seed,trial,rcfg,rules,strong)
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],meta['episode'],t,902]));orders=[]
    for repeat in range(tcfg['repetitions']):
        order=rng.permutation(len(settings));orders.append(order.tolist())
        for index in order:
            name,method,profile,radii=settings[index];trial={**cfg,'radii':radii};action,info,timing,current=measure(sim.env,model,method,images,past,proprio,prefix,nominal,data.scaler.scale_,seed,trial,rcfg,rules,strong)
            assert np.array_equal(current,images[-1]),'Fresh render changed from restored current image'
            if name in selected:assert np.max(np.abs(action-selected[name]))<1e-7,(key,name,'timedrepeat changedoutput')
            else:selected[name]=action.copy();snapshots[name]=info
            records.setdefault(name,[]).append({'repeat':repeat,**timing,'selected_index':info.get('selected_index'),'candidate_count':info['candidate_count']})
    profiles={}
    for name,method,profile,radii in settings:
        with stages(model) as (measured,counts):
            action,info,timing,_=measure(sim.env,model,method,images,past,proprio,prefix,nominal,data.scaler.scale_,seed,{**cfg,'radii':radii},rcfg,rules,strong)
        assert np.max(np.abs(action-selected[name]))<1e-7,(key,name,'instrumentationchangedoutput');profiles[name]={**measured,**timing,'call_counts':counts,'unattributed_seconds':timing['total_seconds']-timing['render_seconds']-sum(measured.values())}
    reference=sim.execute_dense(nominal[:5],nominal[5:]);raw={'nominal_actions':nominal,'nominal_physics':reference['physics_trace'],'nominal_coverage':np.array([reference['coverage_t'],reference['coverage_suffix']]),'images':images,'past_actions':past,'proprio':proprio,'prefix':prefix};rows=[]
    for name,method,profile,radii in settings:
        action=selected[name];actual=sim.execute_dense(action[:5],action[5:]);norm=float(np.linalg.norm((action[:5]-nominal[:5])/data.scaler.scale_));metrics=dense_metrics(actual,reference,cfg['tolerances']);raw[name+'_actions']=action;raw[name+'_physics']=actual['physics_trace'];raw[name+'_coverage']=np.array([actual['coverage_t'],actual['coverage_suffix']])
        rows.append({k:meta[k] for k in ['state_id','split','episode','timestep','initial_pose_family']});rows[-1].update({'variant':name,'method':method,'profile':profile,'norm':norm,'active':norm>1e-6,'nominal_contact':bool(np.any(reference['physics_contacts'])),'candidate_count':snapshots[name]['candidate_count'],'mean_seconds':float(np.mean([r['total_seconds'] for r in records[name]])),'median_seconds':float(np.median([r['total_seconds'] for r in records[name]])),**metrics})
    sim.close();pd.DataFrame(rows).to_csv(done.with_suffix('.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**raw);write_json(done,{'signature':sig,'state_id':key,'repetitions':records,'stage_profiles':profiles,'decisions':snapshots,'order':orders,'seconds':time.perf_counter()-start,'scope':tcfg['scope']});print(key,len(rows),'seconds',round(time.perf_counter()-start,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['calibration','development'],required=True);p.add_argument('--limit',type=int);args=p.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text());rcfg=json.loads(resolve('configs/followup_response_models.json').read_text());tcfg=json.loads(resolve('configs/followup_runtime.json').read_text());base=resolve(cfg['output']);out=base/'runtime';out.mkdir(exist_ok=True);sig=signature();mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())['signature']==sig
    else:write_json(mp,{'signature':sig,'config':tcfg,'hardware':{'platform':platform.platform(),'torch':torch.__version__,'device':cfg['device'],'GPU':torch.cuda.get_device_name(cfg['device']),'threads':cfg['threads']},'resources_at_start':resources()})
    directory=out/args.split;directory.mkdir(exist_ok=True);torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);started=time.perf_counter();model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);load_seconds=time.perf_counter()-started;write_json(directory/'run_environment.json',{'model_load_seconds_excluded':load_seconds,'resources':resources(),'unix':time.time()});rules=json.loads((base/'selection/selector_rules.json').read_text())['rules'];strong=json.loads((base/'strong_baselines/calibrated_settings.json').read_text())['rules']['gaussian_action'];paths=sorted((base/'f1').glob(args.split+'*h05.json'));assert len(paths)==105
    if args.limit:paths=paths[:args.limit]
    warmup=True
    for path in paths:
        exists=(directory/(json.loads(path.read_text())['state_id']+'.json')).exists();run(path,cfg,rcfg,tcfg,data,model,rules,strong,directory,sig,warmup)
        if not exists:warmup=False
    write_json(directory/'resources_at_end.json',{'resources':resources(),'unix':time.time()})
if __name__=='__main__':main()
