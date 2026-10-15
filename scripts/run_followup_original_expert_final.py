"""Execute a frozen nativeCPU verifier test on21reservedoriginalexpert families."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from scripts.model_runtime_imports import load_dino
from scripts.original_expert_evaluation_core import original,nominal,window_steps,context,branch,scores,choose
from scripts.output_replacement_core import physical
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text());SOURCES=['configs/followup_original_expert_final.json','scripts/original_expert_evaluation_core.py','scripts/run_followup_original_expert_final.py','scripts/audit_followup_original_expert_adapter.py','scripts/dinowm_runtime_core.py','scripts/model_runtime_imports.py','scripts/retrained_dinowm_core.py','scripts/output_replacement_core.py','zj_wam/adapters.py','zj_wam/policy_refinement.py','zj_wam/diagnostics.py','zj_wam/followup.py','zj_wam/response_models.py','zj_wam/repeated_refinement.py','zj_wam/model_audit.py','configs/followup_v1.json','configs/followup_response_models.json']

def main(smoke):
    cfg=read(resolve('configs/followup_original_expert_final.json'));base=read(resolve('configs/followup_v1.json'));rcfg=read(resolve('configs/followup_response_models.json'));inventory=read(BASE/'protocol/original_expert_new_families.json');archive=resolve('data/datasets/diffusion-policy-official/pusht.zip');assert sha256(archive)==inventory['archive_sha256'];out=BASE/('original_expert_final_smoke' if smoke else 'original_expert_final');out.mkdir(exist_ok=True);episodes=cfg['smoke_original_episodes'] if smoke else [r['episode'] for r in inventory['reserved_expert_episodes']];rules={r['method']:r['rule'] for r in read(BASE/'dinowm_selection/rules.json')['rules'] if r['environment']=='pusht'} if smoke else read(BASE/'pusht_cpu_calibration/rules.json')['rules']
    if smoke:assert all(inventory['old_family_match_for_official_episode'][i]>=0 for i in episodes)
    else:
        lock=read(out/'final_protocol.json');assert lock['test_original_episodes']==episodes and len(episodes)==21
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        assert lock['rules']==rules
    files=[resolve(p) for p in SOURCES]+[archive,BASE/'protocol/original_expert_new_families.json',BASE/'original_expert_adapter/audit.json',BASE/('dinowm_selection/rules.json' if smoke else 'pusht_cpu_calibration/rules.json'),resolve(base['checkpoint'])/'weights.pt'];files+=sorted((BASE/'protocol/dinowm_retrained_inventory').glob('*.json'));manifest={'smoke':smoke,'config':cfg,'episodes':episodes,'rules':rules,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    torch.set_num_threads(cfg['torch_threads']);data=ExpertDataset(base['dataset']);lewm=FrozenLeWM(base['checkpoint'],data.scaler,'cpu');dino=load_dino('pusht');count=total=ineligible=0
    for episode in episodes:
        recorded,targets=original(archive,episode);a=nominal(recorded[0],targets);steps=window_steps(len(targets),cfg);family=cfg['family_id_offset']+episode;nompath=out/f'e{episode:03d}_nominal.npz';np.savez_compressed(nompath,recorded_states=recorded,absolute_targets=targets,**a)
        for t in steps:
            dest=out/f'e{episode:03d}_t{t:03d}.json'
            if dest.exists():m=read(dest);assert m['manifest_sha256']==sha256(out/'manifest.json');count+=1;total+=m['candidates'];ineligible+=not m['eligible'];continue
            start=time.perf_counter();c=context(a,family,t,data.scaler.mean_,data.scaler.scale_);eligible=bool(np.max(abs(np.r_[c['nominal'],c['suffix']]))<=1+cfg['nominal_bound_tolerance'])
            if eligible:raw,solver,valid=scores(c,base,rcfg,lewm,dino);chosen=choose(raw,rules,cfg)
            else:
                raw={'actions':c['nominal'][None],'labels':np.array([['nominal',0,0]],dtype=str),'norms':np.zeros(1),'predicted_costs':np.zeros(1),**{'score_'+m:np.zeros(1) for m in cfg['methods']}};solver=[];valid=False;chosen={m:0 for m in cfg['methods']};ineligible+=1
            # Decisions are fixed now. No modified physics has been executed.
            write_json(dest.with_suffix('.decision.json'),{'eligible':eligible,'selected_before_physics':chosen,'rules_sha256':sha256(BASE/('dinowm_selection/rules.json' if smoke else 'pusht_cpu_calibration/rules.json')),'manifest_sha256':sha256(out/'manifest.json'),'unix':time.time()})
            raw.update({'input_'+k:np.asarray(v) for k,v in c.items()});traces=[];contacts=[];covers=[];endpoints=[]
            for i,action in enumerate(raw['actions']):
                b=branch(recorded[0],a,t,action);traces.append(np.r_[c['prefix'],b['physics']]);contacts.append(b['contacts']);covers.append(b['coverage']);endpoints.append(b['endpoint_images'])
                if i==0:np.testing.assert_array_equal(b['physics'],a['physics'][t*10:(t+8)*10]);np.testing.assert_array_equal(b['coverage'],a['coverage'][[t+4,t+7]])
            raw.update(physics=np.array(traces),contacts=np.array(contacts),coverage=np.array(covers),endpoint_images=np.array(endpoints));truth=physical('pusht',raw);phase='contact' if raw['contacts'][0].any() else 'free';gain=(raw['predicted_costs'][0]-raw['predicted_costs'])/max(raw['predicted_costs'][0],1e-12);rows=[]
            for i,label in enumerate(raw['labels']):rows.append({'original_episode':episode,'initial_pose_family':family,'state_id':dest.stem,'step':t,'phase':phase,'eligible':eligible,'candidate':i,'generator':label[0],'radius':float(label[1]),'sign':int(label[2]),'norm':raw['norms'][i],'predicted_cost':raw['predicted_costs'][i],'predicted_gain':gain[i],**{k:v[i] for k,v in truth.items()},**{m:raw['score_'+m][i] for m in cfg['methods']}})
            np.savez_compressed(dest.with_suffix('.npz'),**raw);pd.DataFrame(rows).to_csv(dest.with_suffix('.csv'),index=False);write_json(dest,{'original_episode':episode,'initial_pose_family':family,'step':t,'state_id':dest.stem,'phase':phase,'eligible':eligible,'eligibility_reason':'eligible' if eligible else 'nominal_action_bound','visual_contact_valid':valid,'candidates':len(rows),'solver':solver,'selected_before_physics':chosen,'nominal_sha256':sha256(nompath),'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'seconds':time.perf_counter()-start});count+=1;total+=len(rows);print('ORIGINAL EXPERT',episode,t,eligible,len(rows),round(time.perf_counter()-start,2),flush=True)
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    assert count==3*len(episodes);write_json(out/'execution_complete.json',{'smoke':smoke,'families':len(episodes),'states':count,'candidates':total,'ineligible_nominal_windows':ineligible,'manifest_sha256':sha256(out/'manifest.json'),'scope':'Raw collection only; independent action/native prediction/physics/statistical audit required.'})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=read(resolve('configs/followup_original_expert_final.json'))
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):main(args.smoke)
