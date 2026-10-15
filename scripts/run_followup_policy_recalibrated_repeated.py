"""Inject locked common-bank selectors into the already validated policy execution loop."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dill,hydra,numpy as np,torch
from threadpoolctl import threadpool_limits
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from scripts.dinowm_runtime_core import run as choose
import scripts.run_followup_policy_refinement as loop
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_policy_recalibrated_repeated.json','configs/followup_v1.json','configs/followup_response_models.json','configs/followup_dinowm_policy_nominal.json','scripts/run_followup_policy_recalibrated_repeated.py','scripts/run_followup_policy_refinement.py','scripts/dinowm_runtime_core.py','zj_wam/policy_refinement.py','zj_wam/adapters.py','zj_wam/followup.py','zj_wam/repeated_refinement.py','zj_wam/response_models.py','zj_wam/model_audit.py','zj_wam/diagnostics.py']

def main(smoke):
    cfg=read(resolve(SOURCES[0]));base=read(resolve('configs/followup_v1.json'));base['device']=cfg['policy_device'];rcfg=read(resolve('configs/followup_response_models.json'));pcfg=read(resolve('configs/followup_dinowm_policy_nominal.json'));src=resolve(cfg['nominal_source']);nominal=read(src/'raw_audit.json');assert nominal['episodes']==36 and nominal['all_physics_and_RGB_replayed_exact'];selection=BASE/'dinowm_policy_selection';assert read(selection/'pipeline_complete.json')['raw_audit_sha256']==sha256(selection/'raw_audit.json');rules={x['method']:x['rule'] for x in read(resolve(cfg['rules_source']))['rules']};out=BASE/('policy_recalibrated_repeated_smoke' if smoke else 'policy_recalibrated_repeated');out.mkdir(exist_ok=True);maximum=cfg['smoke_actions'] if smoke else cfg['max_actions'];paths=[src/'development'/f"{pcfg['development_seeds'][0]}_nominal.json"] if smoke else sorted(src.glob('*/*_nominal.json'));assert len(paths)==(1 if smoke else 36)
    files=[resolve(p) for p in SOURCES]+[src/'manifest.json',src/'raw_audit.json',selection/'rules.json',selection/'raw_audit.json',selection/'pipeline_complete.json',resolve(base['checkpoint'])/'weights.pt',resolve(pcfg['policy_checkpoint'])]
    for p in paths:files.extend(p.with_suffix(ext) for ext in ['.json','.npz','.csv'])
    manifest={'smoke':smoke,'config':cfg,'max_actions':maximum,'sources':[str(p.relative_to(resolve('.'))) for p in paths],'rules':{m:rules[m] for m in cfg['methods']},'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':cfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    torch.set_num_threads(cfg['torch_threads']);data=ExpertDataset(base['dataset']);model=FrozenLeWM(base['checkpoint'],data.scaler,'cpu');payload=torch.load(resolve(pcfg['policy_checkpoint']),map_location='cpu',pickle_module=dill,weights_only=False);policy=hydra.utils.instantiate(payload['cfg'].policy);policy.load_state_dict(payload['state_dicts']['ema_model'],strict=True);del payload;policy.to(cfg['policy_device']).eval().requires_grad_(False);assert policy.num_inference_steps==100 and policy.n_action_steps==8 and policy.n_obs_steps==2
    active_folder=[None]
    def refine(method,model_arg,images,past,proprio,prefix,nominal_actions,scale,seed,cfg_arg,response_cfg,unused_rules,unused_strong):
        started=time.perf_counter();_,family,t=seed.entropy;c={'episode':family,'timestep':t,'images':images,'past':past,'proprio':proprio,'prefix':prefix,'root':proprio[-1].copy(),'nominal':nominal_actions[:5],'suffix':nominal_actions[5:8],'scale':scale,'mean':data.scaler.mean_};assert np.all(c['root'][[2,3,4,7,8,9]]==0)
        result=choose('pusht',c,cfg_arg,method,rules[method],lewm=model_arg if method=='lewm_global' else None,response_cfg=response_cfg);i=result['selected'];bank=active_folder[0]/'banks'/f'{family}_{method}_t{t:03d}.npz';bank.parent.mkdir(exist_ok=True)
        arrays={k:np.asarray(v) for k,v in c.items()};arrays={'input_'+k:v for k,v in arrays.items()};arrays.update({k:np.asarray(result[k]) for k in ['actions','costs','norms','scores']});arrays['labels']=np.array(result['labels'],dtype=str);np.savez_compressed(bank,**arrays)
        decision={k:v for k,v in result.items() if k not in ['actions','costs','norms','scores']};decision.update({'selected_before_execution':True,'data_sha256':sha256(bank),'manifest_sha256':sha256(out/'manifest.json'),'state_id':f'{family}_{method}_t{t:03d}','seed':family,'step':t,'method':method});write_json(bank.with_suffix('.json'),decision)
        chosen=nominal_actions.copy();chosen[:5]=result['actions'][i];info={'active':bool(result['norms'][i]>1e-6),'norm':float(result['norms'][i]),'score':float(result['scores'][i]),'selected_index':i,'selected_generator':result['labels'][i],'candidate_count':len(result['actions'])-1,'predicted_gain':float((result['costs'][0]-result['costs'][i])/max(result['costs'][0],1e-12)),'seconds':time.perf_counter()-started,'bank':str(bank.relative_to(resolve('.'))),'bank_sha256':sha256(bank),'bank_decision_sha256':sha256(bank.with_suffix('.json'))};return chosen,info
    # The imported loop's dependency is explicitly replaced only in this process.
    # Original source and its prior experiments remain unchanged and hash-locked.
    loop.refine=refine;episodes=0;blocks=0
    for p in paths:
        seed=read(p)['seed'];folder=out/p.parent.name;folder.mkdir(exist_ok=True);active_folder[0]=folder
        for method in cfg['methods']:
            loop.run_episode(seed,method,pcfg,base,rcfg,policy,model,data.scaler.scale_,{},None,folder,maximum);dest=folder/f'{seed}_{method}.json';meta=read(dest);assert meta['actions']==maximum
            if 'recalibrated_manifest_sha256' in meta:assert meta['recalibrated_manifest_sha256']==sha256(out/'manifest.json')
            else:meta.update({'recalibrated_manifest_sha256':sha256(out/'manifest.json'),'nominal_source':str(p.relative_to(resolve('.'))),'recalibrated_runner_sha256':sha256(Path(__file__))});write_json(dest,meta)
            episodes+=1;blocks+=meta['eligible_chunks'];print('RECALIBRATED REPEATED',p.parent.name,seed,method,episodes,flush=True)
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':smoke,'episodes':episodes,'eligible_blocks':blocks,'manifest_sha256':sha256(out/'manifest.json'),'scope':'Collection only; independent policy/physics/selector audit required.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();cfg=read(resolve(SOURCES[0]))
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):main(a.smoke)
