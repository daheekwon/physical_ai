"""Locked verifier response to effective physics and sensor shifts, selected actions only."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from scipy.ndimage import gaussian_filter1d
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.shifted_execution import ShiftedSimulator,noisy_history
from zj_wam.repeated_refinement import visual_history,generate_bank
from zj_wam.response_models import score_models
from zj_wam.model_audit import predict_candidates
from zj_wam.followup import dense_metrics
from zj_wam.diagnostics import jerk_energy
from zj_wam.selection import select_indices
from zj_wam.io import resolve,write_json,sha256

def process(path,cfg,scfg,rcfg,data,model,rules,gauss,out):
 meta=json.loads(path.read_text());key=meta['state_id'];done=out/f'{key}.json'
 if done.exists():return
 started=time.perf_counter();actions,recorded=data.episode(meta['episode']);t=meta['timestep'];nominal=actions[t:t+5];suffix=actions[t+5:t+8];rows=[];archive={};decision_records={}
 for number,(name,condition) in enumerate(scfg['conditions'].items()):
  sim=ShiftedSimulator(actions,recorded,t,cfg['seed']+meta['episode'],condition=condition,lead_actions=scfg['shift_before_root_actions']);clean,past=sim.restore(capture=True);images=noisy_history(clean,condition,np.random.SeedSequence([cfg['seed'],meta['episode'],t,number]));observed,valid=visual_history(images,sim.context_control,rcfg);root=observed[-1];prefix=sim.prefix_physics.copy()
  candidates,labels=generate_bank(nominal,suffix,root,prefix,past,data.scaler.scale_,np.random.SeedSequence([cfg['seed'],meta['episode'],t,5]),cfg['radii'])
  count=len(candidates);filt=gaussian_filter1d(np.concatenate([past[-3:],nominal,suffix]),gauss['setting'],axis=0,mode='nearest')[3:8];delta=(filt-nominal)/data.scaler.scale_;filt=nominal+min(1.,gauss['radius']*np.sqrt(10)/max(np.linalg.norm(delta),1e-12))*(filt-nominal);candidates=np.concatenate([candidates,filt[None]])
  scores,info=score_models(candidates,suffix,root,observed,rcfg);predq=info['pusher'];cost=np.array([jerk_energy(np.concatenate([prefix[:,:2],q]),.01) for q in predq]);norm=np.linalg.norm(((candidates-candidates[:1])/data.scaler.scale_).reshape(len(candidates),-1),axis=1)
  _,context=model.predictor(images,past);pred=predict_candidates(model,context['embedding'],context['past'],candidates[:count]);wm=np.linalg.norm(pred-pred[:1],axis=-1).max(1)
  frame=pd.DataFrame({'state_id':key,'horizon':5,'actual_norm':norm[:count],'predicted_jerk':cost[:count],'nominal_predicted_jerk':cost[0],'controller_pusher_score':scores['pusher'][:count],'vision_contact_score':scores['contact'][:count] if valid else np.inf,'vision_adaptive_score':scores['adaptive'][:count] if valid else np.inf,'lewm_global_score':wm})
  selected={method:max(0,select_indices(frame,method,rules[method])[key]) for method in scfg['methods'] if method!='gaussian_action'};selected['gaussian_action']=count if gauss['enabled'] and cost[-1]<cost[0]*(1-1e-8) and norm[-1]>1e-6 else 0
  archive[name+'_candidate_costs']=cost;archive[name+'_candidate_norms']=norm
  for method in scfg['methods']:
   if method!='gaussian_action':archive[name+'_'+method+'_scores']=frame[method+'_score'].to_numpy()
  # Only now execute reference and selected physical outcomes.
  reference=sim.execute_dense(nominal,suffix);unique={0:reference}
  for index in sorted(set(selected.values())-{0}):unique[index]=sim.execute_dense(candidates[index],suffix)
  for method,index in selected.items():
   metrics=dense_metrics(unique[index],reference,cfg['tolerances']);gain=(cost[0]-cost[index])/max(cost[0],1e-8);active=bool(norm[index]>1e-6)
   rows.append({'state_id':key,'initial_pose_family':meta['initial_pose_family'],'split':meta['split'],'episode':meta['episode'],'timestep':t,'condition':name,'method':method,'index':index,'active':active,'actual_norm':norm[index],'predicted_gain':gain,'predicted_nominal_jerk':cost[0],'predicted_selected_jerk':cost[index],'vision_valid':valid,'nominal_contact':bool(np.any(reference['physics_contacts'][:50]>0)),'translation_gain':info['past']['gains'][0],'rotation_gain':info['past']['gains'][1],**metrics})
   archive[name+'_'+method+'_physics']=unique[index]['physics_trace'];archive[name+'_'+method+'_coverage']=np.array([unique[index]['coverage_t'],unique[index]['coverage_suffix']])
  archive[name+'_nominal_physics']=reference['physics_trace'];archive[name+'_nominal_coverage']=np.array([reference['coverage_t'],reference['coverage_suffix']]);archive[name+'_actions']=candidates;archive[name+'_history_images']=images;archive[name+'_past_actions']=past
  decision_records[name]={'selected_indices':selected,'candidate_count':count-1,'actual_kp':sim.env.k_p,'actual_pair_friction':next(iter(sim.env.block.shapes)).friction*next(iter(sim.env.agent.shapes)).friction,'root':sim.root_full_state.tolist(),'observed_history':observed.tolist(),'input_noise_std':condition.get('rgb_noise_std',0)}
  sim.close()
 pd.DataFrame(rows).to_csv(done.with_suffix('.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**archive)
 write_json(done,{'state_id':key,'split':meta['split'],'initial_pose_family':meta['initial_pose_family'],'decisions':decision_records,'seconds':time.perf_counter()-started,'source_sha256':sha256(Path(__file__)),'config_sha256':sha256(resolve('configs/followup_shift_evaluation.json')),'scope':scfg['adaptation_scope']});print(key,len(rows),round(time.perf_counter()-started,2),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','calibration','test'],default='development');p.add_argument('--limit',type=int);args=p.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text());scfg=json.loads(resolve('configs/followup_shift_evaluation.json').read_text());rcfg=json.loads(resolve('configs/followup_response_models.json').read_text());base=resolve(cfg['output']);out=base/'shift_evaluation';out.mkdir(exist_ok=True)
 if args.split=='test' and not (base/'protocol/final_selection_lock.json').exists():raise ValueError('Test locked')
 torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);rules=json.loads((base/'selection/selector_rules.json').read_text())['rules'];gauss=json.loads((base/'strong_baselines/calibrated_settings.json').read_text())['rules']['gaussian_action']
 paths=sorted((base/'f1').glob(args.split+'*h05.json'));assert len(paths)==(45 if args.split=='test' else 105)
 if args.limit:paths=paths[:args.limit]
 for path in paths:process(path,cfg,scfg,rcfg,data,model,rules,gauss,out)
if __name__=='__main__':main()
