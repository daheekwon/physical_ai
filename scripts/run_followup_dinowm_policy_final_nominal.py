"""Collect reserved policy nominal episodes only after the downstream method lock."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dill,hydra,numpy as np,torch
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset
from scripts.run_followup_policy_refinement import run_episode
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_dinowm_policy_final.json','configs/followup_dinowm_policy_nominal.json','configs/followup_v1.json','configs/followup_response_models.json','scripts/run_followup_dinowm_policy_final_nominal.py','scripts/run_followup_policy_refinement.py','zj_wam/policy_refinement.py','zj_wam/diagnostics.py']

def main(smoke):
    cfg=read(resolve(SOURCES[0]));nomcfg=read(resolve(SOURCES[1]));base=read(resolve(SOURCES[2]));base['device']=nomcfg['policy_device'];rcfg=read(resolve(SOURCES[3]));parent=read(BASE/'dinowm_policy_nominal/manifest.json');out=BASE/('dinowm_policy_final_nominal_smoke' if smoke else 'dinowm_policy_final_nominal');out.mkdir(exist_ok=True)
    if not smoke:
        lock=read(BASE/'dinowm_policy_final/final_protocol.json')
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        assert lock['config']==cfg
    seeds=[cfg['smoke_seed']] if smoke else list(range(cfg['test_seeds']['start'],cfg['test_seeds']['start']+cfg['test_seeds']['count']))
    cfg={**nomcfg,**cfg,'test_seeds':seeds};initials={str(seed):{**parent['initials'][str(seed)],'split':'test'} for seed in seeds}
    assert all(parent['initials'][str(s)]['split']==('development' if smoke else 'reserved_test') for s in seeds)
    ckpt=resolve(nomcfg['policy_checkpoint']);assert sha256(ckpt)==parent['checkpoint_sha256'];manifest={'smoke':smoke,'config':cfg,'initials':initials,'checkpoint_sha256':sha256(ckpt),'input_hashes':{p:sha256(resolve(p)) for p in SOURCES},'parent_manifest_sha256':sha256(BASE/'dinowm_policy_nominal/manifest.json'),'scope':'Nominal interface collection only; correction choices/physics and final efficacy remain.'}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    torch.set_num_threads(base['threads']);data=ExpertDataset(base['dataset']);payload=torch.load(ckpt,map_location='cpu',pickle_module=dill,weights_only=False);policy=hydra.utils.instantiate(payload['cfg'].policy);policy.load_state_dict(payload['state_dicts']['ema_model'],strict=True);del payload;policy.to(base['device']).eval().requires_grad_(False);assert policy.num_inference_steps==100 and policy.n_action_steps==8 and policy.n_obs_steps==2
    folder=out/'test';folder.mkdir(exist_ok=True)
    for seed in seeds:
        run_episode(seed,'nominal',cfg,base,rcfg,policy,None,data.scaler.scale_,{},None,folder,cfg['max_actions']);m=read(folder/f'{seed}_nominal.json');assert m['actions']==300 and m['active_chunks']==0
        with np.load(folder/f'{seed}_nominal.npz') as a:
            np.testing.assert_array_equal(a['initial_state'],initials[str(seed)]['initial_state']);assert a['physics'].shape==(3000,10) and a['images'].shape==(301,224,224,3)
        print('FINAL POLICY NOMINAL',seed,flush=True)
    for p,h in manifest['input_hashes'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':smoke,'episodes':len(seeds),'reserved_test_outcomes_evaluated':not smoke,'manifest_sha256':sha256(out/'manifest.json'),'scope':'Nominal only; raw audit required. No correction efficacy claim.'})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
