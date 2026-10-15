"""Full pipeline timing including uncached derivative and direct nonlinear solve."""
import argparse,json,sys,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import DenseSimulator,dense_metrics
from zj_wam.strong_runtime import refine
from zj_wam.io import resolve,write_json,sha256
from scripts.run_followup_runtime import resources


def signature():
    paths=['configs/followup_strong_runtime.json','scripts/run_followup_strong_runtime.py','zj_wam/strong_runtime.py','zj_wam/locked_geometric.py','zj_wam/strong_optimizers.py','zj_wam/sensitivity.py','zj_wam/adapters.py','zj_wam/followup.py','zj_wam/selection.py','artifacts/followup_v1/strong_baselines/calibrated_settings.json','artifacts/followup_v1/selection/selector_rules.json']
    return {p:sha256(resolve(p)) for p in paths}


def variants(cfg):
    return [(m+('__b'+str(b) if b else ''),m,b) for m in cfg['methods'] for b in (cfg['nonlinear_budgets'] if m=='nonlinear_wm' else [0])]


def process(path,cfg,tcfg,data,model,rules,strong,out,sig,warmup):
    meta=json.loads(path.read_text());done=out/path.name
    if done.exists():assert json.loads(done.read_text())['signature']==sig;return
    actions,states=data.episode(meta['episode']);t=meta['timestep'];nominal=actions[t:t+5];suffix=actions[t+5:t+8];sim=DenseSimulator(actions,states,t,cfg['seed']+meta['episode']);images,past=sim.restore(capture=True);root=sim.root_full_state.copy();prefix=sim.prefix_physics.copy();settings=variants(tcfg);saved={};decisions={};times={};started=time.perf_counter()
    def call(method,budget):
        torch.cuda.synchronize(model.device);resident=torch.cuda.memory_allocated(model.device);torch.cuda.reset_peak_memory_stats(model.device);start=time.perf_counter();current=sim.env.render();frames=np.stack([images[0],images[1],current]);result,info=refine(method,budget,model,frames,past,root,prefix,nominal,suffix,data.scaler.scale_,cfg,rules,strong);torch.cuda.synchronize(model.device)
        return result,info,{'seconds':time.perf_counter()-start,'additional_peak_gpu_bytes':max(0,torch.cuda.max_memory_allocated(model.device)-resident)}
    if warmup:
        for _,m,b in settings:call(m,b)
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],meta['episode'],t,1704]));orders=[]
    for repeat in range(tcfg['repetitions']):
        order=rng.permutation(len(settings));orders.append(order.tolist())
        for index in order:
            name,method,budget=settings[index];a,info,timing=call(method,budget)
            if name in saved:assert np.max(np.abs(a-saved[name]))<1e-7,(name,'repetition output drift')
            else:saved[name]=a;decisions[name]=info
            times.setdefault(name,[]).append({'repeat':repeat,**timing})
    reference=sim.execute_dense(nominal,suffix);archive={'nominal_actions':np.concatenate([nominal,suffix]),'nominal_physics':reference['physics_trace'],'nominal_coverage':np.array([reference['coverage_t'],reference['coverage_suffix']]),'images':images,'past':past,'root':root};rows=[]
    for name,method,budget in settings:
        a=saved[name];info=decisions[name];actual=sim.execute_dense(a[:5],a[5:]) if info['active'] else reference
        archive[name+'_actions']=a;archive[name+'_physics']=actual['physics_trace'];archive[name+'_coverage']=np.array([actual['coverage_t'],actual['coverage_suffix']])
        rows.append({k:meta[k] for k in ['state_id','initial_pose_family','split','episode','timestep']}|{'variant':name,'method':method,'budget':budget,'active':info['active'],'actual_norm':info['actual_norm'],'predicted_gain':info['predicted_gain'],'nominal_contact':bool(np.any(reference['physics_contacts'][:50])),'solver_success':info['solver_success'],'mean_seconds':np.mean([v['seconds'] for v in times[name]]),'median_seconds':np.median([v['seconds'] for v in times[name]]),**dense_metrics(actual,reference,cfg['tolerances'])})
    sim.close();pd.DataFrame(rows).to_csv(done.with_suffix('.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**archive)
    clean=json.loads(json.dumps(decisions,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
    write_json(done,{'state_id':meta['state_id'],'signature':sig,'decisions':clean,'timings':times,'orders':orders,'seconds':time.perf_counter()-started});print(done.stem,len(rows),round(time.perf_counter()-started,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['calibration','development'],required=True);p.add_argument('--limit',type=int);args=p.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text());tcfg=json.loads(resolve('configs/followup_strong_runtime.json').read_text());base=resolve(cfg['output']);out=base/'strong_runtime';out.mkdir(exist_ok=True);sig=signature()
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())['signature']==sig
    else:write_json(out/'manifest.json',{'signature':sig,'config':tcfg,'hardware':{'device':cfg['device'],'GPU':torch.cuda.get_device_name(cfg['device']),'threads':cfg['threads'],'torch':torch.__version__,'platform':platform.platform()},'resources':resources()})
    torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);rules=json.loads((base/'selection/selector_rules.json').read_text())['rules'];strong=json.loads((base/'strong_baselines/calibrated_settings.json').read_text())['rules'];dest=out/args.split;dest.mkdir(exist_ok=True);paths=sorted((base/'f1').glob(args.split+'*h05.json'));assert len(paths)==105
    warmup=True
    for path in paths[:args.limit]:
        exists=(dest/path.name).exists();process(path,cfg,tcfg,data,model,rules,strong,dest,sig,warmup)
        if not exists:warmup=False
    write_json(dest/'resources_at_end.json',{'resources':resources(),'unix':time.time()})
if __name__=='__main__':main()
