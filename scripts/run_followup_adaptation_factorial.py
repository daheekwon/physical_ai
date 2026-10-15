"""Calibrate nominal-only adaptation factors, then evaluate fixed rules under shifts."""
import argparse,json,sys,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.adaptation_factorial import decide
from zj_wam.adapters import ExpertDataset
from zj_wam.shifted_execution import ShiftedSimulator
from zj_wam.followup import dense_metrics
from zj_wam.io import resolve,write_json,sha256


def signature():
    paths=['configs/followup_adaptation_factorial.json','configs/followup_response_models.json','configs/followup_shift_evaluation.json','scripts/run_followup_adaptation_factorial.py','zj_wam/adaptation_factorial.py','zj_wam/response_models.py','zj_wam/repeated_refinement.py','zj_wam/followup.py']
    return {p:sha256(resolve(p)) for p in paths}


def specifications(acfg,tuning,lock):
    if tuning:
        out=[('fixed','fixed',0,None),('compute','compute',0,None),('cheap_only','cheap_only',0,None)]
        for alpha in acfg['alpha_grid']:
            for method in ['allowance','both']:out.append((f'{method}_a{alpha:g}',method,alpha,None))
        out += [(f'constant_r{r:g}','constant_shrink',0,r) for r in acfg['constant_radius_grid']]
        return out
    return [(m,m,lock['alpha'] if m in ['allowance','both'] else 0,lock['constant_radius'] if m=='constant_shrink' else None) for m in acfg['methods']]


def process(path,cfg,scfg,rcfg,acfg,data,rule,out,tuning,lock,source_signature):
    meta=json.loads(path.read_text());key=meta['state_id'];done=out/f'{key}.json'
    if done.exists():
        assert json.loads(done.read_text())['signature']==source_signature
        return
    start=time.perf_counter();actions,recorded=data.episode(meta['episode']);t=meta['timestep'];suffix=actions[t+5:t+8]
    source=resolve(cfg['output'])/'shift_evaluation'/f'{key}.json';source_meta=json.loads(source.read_text());rows=[];records={};archive={};equivalence=0.
    with np.load(source.with_suffix('.npz')) as raw:
        for condition_number,(name,condition) in enumerate(scfg['conditions'].items()):
            if tuning and name!='nominal':continue
            images=raw[name+'_history_images'];past=raw[name+'_past_actions'];proprio=np.asarray(source_meta['decisions'][name]['observed_history']);prefix=raw[name+'_nominal_physics'][:30];nominal=actions[t:t+5];np.testing.assert_array_equal(nominal,raw[name+'_actions'][0]);seed=np.random.SeedSequence([cfg['seed'],meta['episode'],t,5])
            pending=[];specs=specifications(acfg,tuning,lock)
            # Rotate execution order to reduce systematic warm-cache/host-load bias.
            rotate=(meta['episode']+t+condition_number)%len(specs);specs=specs[rotate:]+specs[:rotate]
            for label,method,alpha,radius in specs:
                selected,decision,bank=decide(images,proprio,past,prefix,nominal,suffix,data.scaler.scale_,seed,cfg['radii'],rcfg,acfg,rule,method,alpha,radius)
                # Verify same potential actions/scores as existing matched shift bank, before execution.
                if decision['vision_valid']:
                    old=raw[name+'_actions'][:-1];oldscore=raw[name+'_vision_adaptive_scores']
                    for i,a in enumerate(bank['actions']):
                        distances=np.max(np.abs(old-a),axis=(1,2));j=int(np.argmin(distances));assert distances[j]<1e-10,(key,name,label,i,distances[j])
                        discrepancy=max(abs(bank['costs'][i]-raw[name+'_candidate_costs'][j])/max(1,abs(bank['costs'][i])),abs(bank['norms'][i]-raw[name+'_candidate_norms'][j]),abs(bank['scores'][i]-oldscore[j]))
                        equivalence=max(equivalence,float(discrepancy));assert discrepancy<1e-7,(key,name,label,discrepancy)
                pending.append((label,selected,decision,bank))
            # Future simulation starts only after every selection in this condition is fixed.
            sim=ShiftedSimulator(actions,recorded,t,cfg['seed']+meta['episode'],condition=condition,lead_actions=scfg['shift_before_root_actions'])
            reference=sim.execute_dense(nominal,suffix)
            np.testing.assert_allclose(reference['physics_trace'],raw[name+'_nominal_physics'],rtol=0,atol=1e-9)
            archive[name+'_nominal_physics']=reference['physics_trace'];archive[name+'_nominal_coverage']=np.array([reference['coverage_t'],reference['coverage_suffix']]);archive[name+'_history_images']=images;archive[name+'_proprio']=proprio;archive[name+'_past_actions']=past
            cache={nominal.tobytes():reference}
            for label,selected,decision,bank in pending:
                action_key=selected.tobytes()
                if action_key not in cache:cache[action_key]=sim.execute_dense(selected,suffix)
                executed=cache[action_key];metrics=dense_metrics(executed,reference,cfg['tolerances']);tag=name+'__'+label
                row={'state_id':key,'initial_pose_family':meta['initial_pose_family'],'split':meta['split'],'episode':meta['episode'],'timestep':t,'condition':name,'variant':label,'nominal_contact':bool(np.any(reference['physics_contacts'][:50]>0)),**{k:v for k,v in decision.items() if k not in ['timing','labels','past_error_components']},**metrics}
                row.update({f'time_{k}':v for k,v in decision['timing'].items()});rows.append(row);records[tag]=decision
                for k,v in bank.items():archive[tag+'_'+k]=v
                archive[tag+'_physics']=executed['physics_trace'];archive[tag+'_coverage']=np.array([executed['coverage_t'],executed['coverage_suffix']])
            sim.close()
    pd.DataFrame(rows).to_csv(done.with_suffix('.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**archive)
    write_json(done,{'state_id':key,'initial_pose_family':meta['initial_pose_family'],'split':meta['split'],'signature':source_signature,'source_shift_sha256':sha256(source),'decisions':records,'bank_equivalence_max_error':equivalence,'seconds':time.perf_counter()-start,'tuning':tuning,'locked_settings':lock})
    print(key,'rows',len(rows),'seconds',round(time.perf_counter()-start,2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tune','evaluate'],required=True);p.add_argument('--split',choices=['development','calibration','test'],default='calibration');p.add_argument('--limit',type=int);args=p.parse_args()
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());scfg=json.loads(resolve('configs/followup_shift_evaluation.json').read_text());rcfg=json.loads(resolve('configs/followup_response_models.json').read_text());acfg=json.loads(resolve('configs/followup_adaptation_factorial.json').read_text());base=resolve(cfg['output']);directory=base/'adaptation_factorial';directory.mkdir(exist_ok=True)
    if args.split=='test':raise ValueError('Test evaluation requires a separately audited final lock and shift source; not enabled by this runner yet')
    tuning=args.phase=='tune'
    if tuning and args.split!='calibration':raise ValueError('Tuning only nominal calibration')
    lock=None if tuning else json.loads((directory/'calibrated_settings.json').read_text());sig=signature()
    if lock:assert lock['signature']==sig
    out=directory/('calibration_tuning' if tuning else args.split);out.mkdir(exist_ok=True)
    data=ExpertDataset(cfg['dataset']);rule=json.loads((base/'selection/selector_rules.json').read_text())['rules'][acfg['verifier']]
    manifest={'signature':sig,'config':acfg,'base_verifier_rule':rule,'platform':platform.platform(),'scope':'Actual selector compute after inputs available; includes pose estimation, excludes simulator image acquisition and physical diagnostic replay.'}
    target=directory/'manifest.json'
    if target.exists():assert json.loads(target.read_text())['signature']==sig
    else:write_json(target,manifest)
    paths=sorted((base/'f1').glob(args.split+'*h05.json'));assert len(paths)==105
    if args.limit:paths=paths[:args.limit]
    for path in paths:process(path,cfg,scfg,rcfg,acfg,data,rule,out,tuning,lock,sig)
if __name__=='__main__':main()
