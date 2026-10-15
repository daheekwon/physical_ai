"""Collect fresh nominal policy families, preserving the validated frozen adapter."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dill,hydra,numpy as np,torch
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset
from zj_wam.policy_refinement import make_scene,full_state
from scripts.run_followup_policy_refinement import run_episode
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def canon(x):
    out=np.asarray(x,float).copy()[...,:5];out[...,4]=np.arctan2(np.sin(out[...,4]),np.cos(out[...,4]));return out


def main(smoke):
    cfg=read(resolve('configs/followup_dinowm_policy_nominal.json'));basecfg=read(resolve('configs/followup_v1.json'));basecfg['device']=cfg['policy_device'];torch.set_num_threads(basecfg['threads']);out=BASE/('dinowm_policy_nominal_smoke' if smoke else 'dinowm_policy_nominal');out.mkdir(exist_ok=True);data=ExpertDataset(basecfg['dataset']);expert=np.unique(canon(data.states[np.asarray(data.dataset.offsets,int)]),axis=0);prior=[];priorinputs={}
    for folder in ['dp_native','policy_refinement/development','policy_refinement/calibration','policy_refinement/test','policy_refinement_smoke/development','policy_latency/development','policy_latency/test']:
        for p in sorted((BASE/folder).glob('*.npz')):
            with np.load(p) as a:
                if 'initial_state' in a:prior.append(canon(a['initial_state']));priorinputs[str(p.relative_to(resolve('.')))]=sha256(p)
    existing=np.r_[expert,np.array(prior).reshape(-1,5)];allseeds={s:split for split in ['development','calibration'] for s in cfg[split+'_seeds']};allseeds.update({s:'reserved_test' for s in range(cfg['reserved_test_seed_start'],cfg['reserved_test_seed_start']+cfg['reserved_test_family_count'])});assert len(allseeds)==100;initials={};seen=[]
    for seed,split in allseeds.items():
        env=make_scene(seed);initial=full_state(env);env.close();x=canon(initial);delta=np.max(abs(existing-x),axis=1);assert delta.min()>1e-6,(seed,'prior initial family overlap')
        if seen:assert np.max(abs(np.array(seen)-x),axis=1).min()>1e-6,(seed,'new family collision')
        seen.append(x);initials[str(seed)]={'split':split,'initial_state':initial.tolist(),'nearest_prior_max_coordinate_difference':float(delta.min())}
    ckpt=resolve(cfg['policy_checkpoint']);known=read(BASE/'dp_native/manifest.json')['checkpoint_sha256'];assert sha256(ckpt)==known,'Frozen official DP asset changed';paths=['configs/followup_dinowm_policy_nominal.json','scripts/run_followup_dinowm_policy_nominal.py','scripts/run_followup_policy_refinement.py','zj_wam/policy_refinement.py','zj_wam/diagnostics.py','configs/followup_v1.json','configs/followup_response_models.json'];manifest={'smoke':smoke,'config':cfg,'initials':initials,'input_hashes':{p:sha256(resolve(p)) for p in paths},'prior_initial_inputs':priorinputs,'checkpoint_sha256':known,'expert_unique_canonical_families':len(expert),'scope':cfg['scope']}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    payload=torch.load(ckpt,map_location='cpu',pickle_module=dill,weights_only=False);policy=hydra.utils.instantiate(payload['cfg'].policy);policy.load_state_dict(payload['state_dicts']['ema_model'],strict=True);del payload;policy.to(cfg['policy_device']).eval().requires_grad_(False);assert policy.num_inference_steps==100 and policy.n_action_steps==8 and policy.n_obs_steps==2;rcfg=read(resolve('configs/followup_response_models.json'));count=0
    for split in ['development','calibration']:
        seeds=cfg[split+'_seeds'][:1] if smoke else cfg[split+'_seeds']
        if smoke and split=='calibration':continue
        folder=out/split;folder.mkdir(exist_ok=True)
        for seed in seeds:
            run_episode(seed,'nominal',cfg,basecfg,rcfg,policy,None,data.scaler.scale_,{},None,folder,cfg['max_actions']);meta=read(folder/f'{seed}_nominal.json');assert meta['actions']==300 and meta['active_chunks']==0
            with np.load(folder/f'{seed}_nominal.npz') as a:
                np.testing.assert_array_equal(a['initial_state'],initials[str(seed)]['initial_state']);assert a['physics'].shape==(3000,10);assert a['images'].shape==(301,224,224,3)
            count+=1;print('FRESH NOMINAL',split,seed,count,flush=True)
    for p,h in manifest['input_hashes'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':smoke,'episodes':count,'reserved_test_outcomes_evaluated':False,'manifest_sha256':sha256(out/'manifest.json'),'scope':'Nominal collection only; independent physical/action/policy-interface audit required before candidate comparison.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();main(args.smoke)
