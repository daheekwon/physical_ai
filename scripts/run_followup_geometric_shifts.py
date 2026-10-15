"""Frozen strong non-WM baselines on the existing eight physical/sensing shifts."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.adapters import ExpertDataset
from zj_wam.shifted_execution import ShiftedSimulator
from zj_wam.locked_geometric import refine
from zj_wam.followup import dense_metrics
from zj_wam.io import resolve,write_json,sha256

METHODS=['gaussian_action','lipo_full_command','groove_full_command','lipo_polynomial_executed','groove_polynomial_executed']

def signature():
    return {p:sha256(resolve(p)) for p in ['scripts/run_followup_geometric_shifts.py','zj_wam/locked_geometric.py','zj_wam/strong_optimizers.py','zj_wam/shifted_execution.py','zj_wam/followup.py','configs/followup_shift_evaluation.json','artifacts/followup_v1/strong_baselines/calibrated_settings.json']}

def process(path,cfg,scfg,data,rules,dest,sig):
    done=dest/path.name
    if done.exists():
        assert json.loads(done.read_text())['signature']==sig
        return
    meta=json.loads(path.read_text());t=meta['timestep'];actions,states=data.episode(meta['episode']);nominal=actions[t:t+5];suffix=actions[t+5:t+8]
    archive={};rows=[];decisions={};start=time.perf_counter()
    # Prior raw outcomes are for post-selection audit only, never selector inputs.
    for condition,params in scfg['conditions'].items():
        sim=ShiftedSimulator(actions,states,t,cfg['seed']+meta['episode'],condition=params,lead_actions=scfg['shift_before_root_actions'])
        _,past=sim.restore(capture=True);root=sim.root_full_state.copy();prefix=sim.prefix_physics.copy()
        chosen={m:refine(m,nominal,suffix,root,prefix,past,data.scaler.scale_,rules[m]) for m in METHODS}
        reference=sim.execute_dense(nominal,suffix)
        archive[condition+'_nominal_physics']=reference['physics_trace'];archive[condition+'_nominal_coverage']=np.array([reference['coverage_t'],reference['coverage_suffix']]);archive[condition+'_nominal_actions']=nominal;archive[condition+'_suffix']=suffix;archive[condition+'_root']=root;archive[condition+'_past']=past
        decisions[condition]={}
        for method,(a,info) in chosen.items():
            # Match prior shared-bank dtype at delivery (Gaussian was stacked with float64 candidates).
            a=np.asarray(a,dtype=np.float64)
            actual=sim.execute_dense(a,suffix) if info['active'] else reference
            rows.append({k:meta[k] for k in ['state_id','initial_pose_family','split','episode','timestep']} | {'condition':condition,'method':method,'nominal_contact':bool(np.any(reference['physics_contacts'][:50]>0)),**{k:v for k,v in info.items() if k!='optimizer'},**dense_metrics(actual,reference,cfg['tolerances'])})
            archive[condition+'_'+method+'_actions']=a;archive[condition+'_'+method+'_physics']=actual['physics_trace'];archive[condition+'_'+method+'_coverage']=np.array([actual['coverage_t'],actual['coverage_suffix']]);decisions[condition][method]=info
        sim.close()
    pd.DataFrame(rows).to_csv(done.with_suffix('.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**archive)
    serial=json.loads(json.dumps(decisions,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
    write_json(done,{**{k:meta[k] for k in ['state_id','initial_pose_family','split']},'signature':sig,'decisions':serial,'seconds':time.perf_counter()-start})
    print(done.stem,len(rows),round(time.perf_counter()-start,2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','calibration'],required=True);p.add_argument('--limit',type=int);args=p.parse_args()
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());scfg=json.loads(resolve('configs/followup_shift_evaluation.json').read_text());base=resolve(cfg['output']);dest=base/'geometric_shifts';dest.mkdir(exist_ok=True);sig=signature()
    manifest={'signature':sig,'methods':METHODS,'conditions':scfg['conditions'],'scope':'Original nominal-calibrated settings unchanged. Same shifted root/history as shift_evaluation. Robot proprioception/history only; no RGB/object state/shift parameter used in selection. Unknown gain retains nominal PD model. Full-command and polynomial families reported separately. Delivery casts active corrections to float64, matching prior shift bank. Nominal reference remains original float32. Post-choice comparison to previous Gaussian/reference verifies exact protocol reproduction. Development/calibration only; no test tuning.'}
    if (dest/'manifest.json').exists():assert json.loads((dest/'manifest.json').read_text())==manifest
    else:write_json(dest/'manifest.json',manifest)
    rules=json.loads((base/'strong_baselines/calibrated_settings.json').read_text())['rules'];data=ExpertDataset(cfg['dataset']);paths=sorted((base/'f1').glob(args.split+'*h05.json'));assert len(paths)==105
    for path in paths[:args.limit]:process(path,cfg,scfg,data,rules,dest,sig)
if __name__=='__main__':main()
