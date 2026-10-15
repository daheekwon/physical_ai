"""Reference-available20action refinement, calibration and fixed development rules."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.adapters import ExpertDataset
from zj_wam.followup import DenseSimulator,dense_metrics
from zj_wam.cumulative_allowance import choose
from scripts.run_followup_repeated_refinement import execute_with_coverage,core_metrics
from zj_wam.io import resolve,write_json,sha256


def signature():return {p:sha256(resolve(p)) for p in ['configs/followup_cumulative_allowance.json','zj_wam/cumulative_allowance.py','scripts/run_followup_cumulative_allowance.py','zj_wam/repeated_refinement.py','zj_wam/response_models.py','zj_wam/followup.py']}

def specifications(ccfg,tuning,lock):
    out=[]
    for tier in ccfg['tiers']:
        out.append((f'{tier}_local',tier,'local',1.))
        for method in ccfg['methods'][1:]:
            values=ccfg['multiplier_grid'] if tuning else [lock['rules'][tier][method]['multiplier']]
            out.extend((f'{tier}_{method}_m{m:g}' if tuning else f'{tier}_{method}',tier,method,m) for m in values)
    return out


def process(path,cfg,rcfg,ccfg,data,rule,out,tuning,lock,sig):
    meta=json.loads(path.read_text());key=meta['state_id'];done=out/f'{key}.json'
    if done.exists():assert json.loads(done.read_text())['signature']==sig;return
    started=time.perf_counter();original,recorded=data.episode(meta['episode']);t=meta['timestep'];h=ccfg['total_actions'];suffix=original[t+h:t+h+ccfg['long_suffix_max_actions']];assert len(suffix)>=3
    sim=DenseSimulator(original,recorded,t,cfg['seed']+meta['episode']);reference=execute_with_coverage(sim,original[t:t+h],suffix);sim.close();ref_context={}
    for offset in range(0,h,5):
        sim=DenseSimulator(original,recorded,t+offset,cfg['seed']+meta['episode']);images,past=sim.restore(capture=True);ref_context[offset]=(images,sim.context_control.copy());sim.close()
    archive={'nominal_physics':reference['physics_trace'],'nominal_control':reference['control_trace'],'nominal_coverage':np.array([reference['coverage_t'],reference['coverage_core_suffix'],reference['coverage_suffix']]),'nominal_actions':original[t:t+h],'suffix_actions':suffix};records={};rows=[];contact=(reference['physics_contacts'][:h*10].reshape(h,10)>0).any(1)
    for label,tier,method,multiplier in specifications(ccfg,tuning,lock):
        delivered=original.copy();logs=[]
        for offset in range(0,h,5):
            current=t+offset;sim=DenseSimulator(delivered,recorded,current,cfg['seed']+meta['episode']);images,past=sim.restore(capture=True);refimages,refproprio=ref_context[offset]
            selected,d,a=choose(images,sim.context_control, past,sim.prefix_physics,refimages,refproprio,original[current:current+5],original[current+5:current+8],data.scaler.scale_,np.random.SeedSequence([cfg['seed'],meta['episode'],current,5]),cfg,rcfg,rule,tier,method,multiplier)
            delivered[current:current+5]=selected;d['offset']=offset;logs.append(d);tag=f'{label}_block{offset//5}'
            for k,v in a.items():archive[tag+'_'+k]=v
            archive[tag+'_past_actions']=past;archive[tag+'_prefix']=sim.prefix_physics.copy();archive[tag+'_proprio']=sim.context_control.copy();archive[tag+'_reference_proprio']=refproprio;archive[tag+'_images']=images;archive[tag+'_reference_images']=refimages
            # Current state is the result of this method's delivered past; future physical truth is not passed to choose.
            sim.close()
        sim=DenseSimulator(original,recorded,t,cfg['seed']+meta['episode']);result=execute_with_coverage(sim,delivered[t:t+h],suffix);sim.close();core=core_metrics(result,reference,cfg['tolerances']);long=dense_metrics(result,reference,cfg['tolerances']);norm=float(np.linalg.norm((delivered[t:t+h]-original[t:t+h])/data.scaler.scale_));nested=max(core['object_error'],long['object_error'])
        velocity={}
        for endpoint in [20,23,20+len(suffix)]:
            z=result['physics'][endpoint*10-1];r=reference['physics'][endpoint*10-1];velocity[f'velocity_delta_t{endpoint}']=float(np.linalg.norm(z[7:9]-r[7:9]));velocity[f'angular_velocity_delta_t{endpoint}']=float(abs(z[9]-r[9]))
        row={k:meta[k] for k in ['state_id','split','episode','timestep','initial_pose_family']};row.update({'variant':label,'tier':tier,'method':method,'multiplier':multiplier,'actual_norm':norm,'active':norm>1e-6,'active_chunks':sum(d['active'] for d in logs),'refinement_seconds':sum(d['seconds'] for d in logs),'suffix_actions':len(suffix),'nominal_contact':bool(contact.any()),'nominal_contact_transition':bool(np.any(contact[1:]!=contact[:-1])),'nominal_success':reference['nominal_success'],'modified_success':result['nominal_success'],**core,**{'long_'+k:v for k,v in long.items()},'nested_long_error':nested,**velocity});rows.append(row);records[label]=logs
        archive[label+'_actions']=delivered[t:t+h];archive[label+'_physics']=result['physics_trace'];archive[label+'_coverage']=np.array([result['coverage_t'],result['coverage_core_suffix'],result['coverage_suffix']])
    pd.DataFrame(rows).to_csv(done.with_suffix('.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**archive);write_json(done,{'state_id':key,'initial_pose_family':meta['initial_pose_family'],'split':meta['split'],'signature':sig,'decisions':records,'seconds':time.perf_counter()-started,'tuning':tuning,'locked_settings':lock,'scope':ccfg['scope']});print(key,len(rows),'seconds',round(time.perf_counter()-started,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tune','evaluate'],required=True);p.add_argument('--split',choices=['development','calibration'],default='calibration');p.add_argument('--limit',type=int);args=p.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text());rcfg=json.loads(resolve('configs/followup_response_models.json').read_text());ccfg=json.loads(resolve('configs/followup_cumulative_allowance.json').read_text());base=resolve(cfg['output']);directory=base/'cumulative_allowance';directory.mkdir(exist_ok=True);sig=signature();tuning=args.phase=='tune';assert not tuning or args.split=='calibration';lock=None if tuning else json.loads((directory/'calibrated_settings.json').read_text());rule=json.loads((base/'selection/selector_rules.json').read_text())['rules']['vision_adaptive'];manifest={'signature':sig,'config':ccfg,'base_rule':rule};mp=directory/'manifest.json'
    if lock:assert lock['signature']==sig
    if mp.exists():assert json.loads(mp.read_text())['signature']==sig
    else:write_json(mp,manifest)
    out=directory/('calibration_tuning' if tuning else args.split);out.mkdir(exist_ok=True);paths=sorted((base/'f1').glob(args.split+'*h05.json'));assert len(paths)==105
    if args.limit:paths=paths[:args.limit]
    data=ExpertDataset(cfg['dataset'])
    for path in paths:process(path,cfg,rcfg,ccfg,data,rule,out,tuning,lock,sig)
if __name__=='__main__':main()
