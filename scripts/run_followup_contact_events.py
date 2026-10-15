"""Run fixed contact-event windows with native LeWM and non-WM verifier controls."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.model_audit import predict_candidates
from zj_wam.repeated_refinement import visual_history
from zj_wam.response_models import score_models
from scripts.dinowm_runtime_core import generate
from scripts.run_followup_dinowm_policy_bank import context,branch
from scripts.original_expert_evaluation_core import choose
from scripts.collect_followup_lewm_common_effect_features import encode
from scripts.output_replacement_core import physical
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_contact_events.json','configs/followup_v1.json','configs/followup_response_models.json','scripts/run_followup_contact_events.py','scripts/contact_event_core.py','scripts/run_followup_dinowm_policy_bank.py','scripts/dinowm_runtime_core.py','scripts/original_expert_evaluation_core.py','scripts/collect_followup_lewm_common_effect_features.py','scripts/output_replacement_core.py','zj_wam/adapters.py','zj_wam/policy_refinement.py','zj_wam/model_audit.py','zj_wam/response_models.py','zj_wam/repeated_refinement.py','zj_wam/followup.py','zj_wam/diagnostics.py']

def main(smoke):
    cfg=read(resolve(SOURCES[0]));base=read(resolve(SOURCES[1]));rcfg=read(resolve(SOURCES[2]));cat=BASE/'contact_events_catalogue';m=read(cat/'manifest.json');audit=read(cat/'raw_audit.json');assert audit['manifest_sha256']==sha256(cat/'manifest.json') and audit['all_nominal_contacts_and_physics_freshly_replayed'];states=[r for r in m['unique_states'] if r['seed']==cfg['development_seeds'][0]] if smoke else m['unique_states'];out=BASE/('contact_events_smoke' if smoke else 'contact_events');out.mkdir(exist_ok=True);rules={r['method']:r['rule'] for r in read(BASE/'dinowm_policy_selection/rules.json')['rules'] if r['method'] in cfg['methods']};files=[resolve(p) for p in SOURCES]+[cat/'manifest.json',cat/'resolution.json',cat/'raw_audit.json',BASE/'dinowm_policy_selection/rules.json',BASE/'dinowm_policy_selection/raw_audit.json',BASE/'dinowm_readouts/raw_audit.json',resolve(base['checkpoint'])/'weights.pt']
    for rep in cfg['representations']:files.append(BASE/'dinowm_readouts/pusht'/f'{rep}__raw__ridge.npz')
    for seed in sorted({r['seed'] for r in states}):
        for ext in ['.json','.npz']:files.append(BASE/'dinowm_policy_nominal/development'/f'{seed}_nominal{ext}')
    protocol={'smoke':smoke,'config':cfg,'states':states,'cases':[r for r in m['cases'] if r['state_id'] in {x['state_id'] for x in states}],'rules':rules,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope']}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write_json(out/'protocol.json',protocol)
    torch.set_num_threads(cfg['threads']);data=ExpertDataset(base['dataset']);model=FrozenLeWM(base['checkpoint'],data.scaler,'cpu');count=total=0
    for spec in states:
        seed=spec['seed'];t=spec['step'];p=out/(spec['state_id']+'.json');src=BASE/'dinowm_policy_nominal/development'/f'{seed}_nominal.npz';a=dict(np.load(src));start=time.perf_counter();c=context(a,seed,t,data.scaler.scale_,data.scaler.mean_);actions,labels,robot,cost,norm,solver=generate('pusht',c,base);observed,valid=visual_history(c['images'],c['proprio'],rcfg);sc,_=score_models(actions,c['suffix'],observed[-1],observed,rcfg);_,lc=model.predictor(c['images'],c['past'])
        with torch.no_grad():pred=predict_candidates(model,lc['embedding'],lc['past'],actions,batch_size=64)[:,0]
        scores={'controller':np.linalg.norm(robot-robot[:1],axis=2).max(1),'contact':sc['contact'] if valid else np.full(len(actions),np.inf),'lewm_global':np.linalg.norm(pred-pred[:1],axis=1).astype(float)};raw={'actions':actions,'labels':np.array(labels,dtype=str),'norms':norm,'predicted_robot':robot,'predicted_costs':cost,'history':lc['embedding'][0].cpu().numpy(),'predicted':pred,**{'score_'+k:v for k,v in scores.items()},**{'input_'+k:np.asarray(v) for k,v in c.items()}};selected=choose(raw,rules,cfg);write_json(p.with_suffix('.decision.json'),{'selected_before_physics':selected,'protocol_sha256':sha256(out/'protocol.json'),'unix':time.time()});physics=[];contacts=[];coverage=[];images=[]
        for i,action in enumerate(actions):
            b=branch(seed,t,a['actions'],action);np.testing.assert_array_equal(b['root'],a['states'][t]);np.testing.assert_array_equal(b['image'],a['images'][t]);physics.append(np.r_[c['prefix'],b['physics']]);contacts.append(b['contacts']);coverage.append(b['coverage'][[4,7]]);images.append(b['endpoint'])
        raw.update(physics=np.array(physics),contacts=np.array(contacts),coverage=np.array(coverage),endpoint_images=np.array(images));np.testing.assert_array_equal(raw['physics'][0,30:],a['physics'][10*t:10*(t+8)]);raw['actual']=encode(model,raw['endpoint_images'],64);assert raw['actual'].shape==pred.shape==(len(actions),192);truth=physical('pusht',raw);phase='contact' if raw['contacts'][0].any() else 'free';gain=(cost[0]-cost)/max(cost[0],1e-12);rows=[]
        for i,label in enumerate(labels):rows.append({'state_id':spec['state_id'],'initial_pose_family':seed,'phase':phase,'candidate':i,'generator':label[0],'radius':float(label[1]),'sign':int(label[2]),'norm':norm[i],'predicted_cost':cost[i],'predicted_gain':gain[i],**{k:v[i] for k,v in truth.items()},**{k:v[i] for k,v in scores.items()}})
        np.savez_compressed(p.with_suffix('.npz'),**raw);pd.DataFrame(rows).to_csv(p.with_suffix('.csv'),index=False);write_json(p,{'seed':seed,'step':t,'state_id':spec['state_id'],'phase':phase,'candidates':len(actions),'visual_contact_valid':bool(valid),'selected_before_physics':selected,'solver':solver,'source':str(src.relative_to(resolve('.'))),'protocol_sha256':sha256(out/'protocol.json'),'data_sha256':sha256(p.with_suffix('.npz')),'seconds':time.perf_counter()-start});count+=1;total+=len(actions);print('CONTACT EVENT',spec['state_id'],len(actions),round(time.perf_counter()-start,2),flush=True)
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':smoke,'states':count,'candidates':total,'cases':len(protocol['cases']),'protocol_sha256':sha256(out/'protocol.json'),'scope':'Raw collection; independent action/physics/model/readout/statistical audits required.'})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
