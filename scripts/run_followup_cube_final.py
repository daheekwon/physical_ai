"""Separate Cube final-family planning, immutable lock and native execution."""
import argparse,json,sys,time,os,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,h5py,hdf5plugin
from zj_wam.io import resolve,write_json,sha256

BASE=resolve('artifacts/followup_v1/cube');LOCK=BASE/'final_protocol.json'

def load(p):return json.loads(p.read_text())

def frozen_inputs():
    paths=['configs/followup_cube.json','configs/followup_cube_final.json','configs/followup_cube_probes.json','zj_wam/cube_execution.py','zj_wam/cube_experiment.py','zj_wam/cube_readouts.py','scripts/run_followup_cube.py','scripts/run_followup_cube_final.py','scripts/analyze_followup_cube_final.py','scripts/analyze_followup_cube.py','scripts/analyze_followup_cube_probes.py','scripts/analyze_followup_cube_proposals.py','scripts/analyze_followup_response_selection.py','zj_wam/io.py']
    inputs=[resolve(p) for p in paths]+[BASE/p for p in ['manifest.json','final_preflight_validation.json','evaluation.csv','raw_audit.json','calibrated_settings.json','proposal_ablation/analysis.json','matched_proposals/analysis.json','matched_proposals/raw_audit.json','probes_v2/selection.json','probes_v2/analysis.json']]+list((BASE/'probes_v2').glob('*.joblib'))
    return {str(p.relative_to(resolve('.'))):sha256(p) for p in inputs}


def verify():
    m=load(LOCK)
    for p,h in m['frozen_inputs'].items():assert sha256(resolve(p))==h,('Cube frozen input changed',p)
    cfg=load(resolve('configs/followup_cube.json'));stat=resolve(cfg['dataset']).stat();assert stat.st_size==m['dataset_file']['bytes'] and stat.st_mtime_ns==m['dataset_file']['mtime_ns'];assert sha256(resolve(cfg['checkpoint'])/'weights.pt')==m['checkpoint_sha256']
    return m


def freeze():
    if LOCK.exists():verify();return
    assert not list((BASE/'test').glob('e*.npz')),'Test already exposed without final protocol'
    cfg=load(resolve('configs/followup_cube.json'));fcfg=load(resolve('configs/followup_cube_final.json'));parent=load(BASE/'manifest.json');dev=pd.read_csv(BASE/'evaluation.csv');dev=dev[dev.split=='development'];planning=[]
    for method in ['controller_displacement','proprio_contact','lewm_global']:
        g=dev[dev.method==method].copy();g['violation']=g.object_error>1
        for metric in ['joint','violation']:
            values=g.groupby('initial_pose_family')[metric].mean().to_numpy();sd=float(np.std(values,ddof=1));needed=int(np.ceil((fcfg['planning_z']*sd/fcfg['planning_half_width'])**2));planning.append({'method':method,'metric':metric,'family_sd':sd,'estimated_families':needed})
    n=max(fcfg['minimum_families'],int(np.ceil(max(x['estimated_families'] for x in planning)/fcfg['family_rounding'])*fcfg['family_rounding']))
    from scripts.run_followup_cube import phases,signature
    assert parent['signature']==signature()
    excluded=set(cfg['excluded_pilot_episodes'])|set(sum([v for k,v in parent['splits'].items() if k!='test'],[]));reserved=parent['splits']['test'];rng=np.random.default_rng(fcfg['seed']);chosen=[];rejected=[];phase_map=dict(parent['phases'])
    with h5py.File(resolve(cfg['dataset']),'r') as f:
        offsets=f['ep_offset'][:];initial=np.c_[f['qpos'][offsets],f['qvel'][offsets]];family_keys=[tuple(x) for x in np.round(initial,8)];seen={family_keys[i] for i in excluded};pool=np.setdiff1d(np.arange(len(offsets)),list(excluded|set(reserved)));rng.shuffle(pool)
        for ep in [*reserved,*pool.tolist()]:
            if family_keys[ep] in seen:rejected.append({'episode':ep,'reason':'duplicate initial family'});continue
            try:times=phases(f,int(offsets[ep]),int(f['ep_len'][ep]),cfg)
            except ValueError as e:rejected.append({'episode':ep,'reason':str(e)});continue
            chosen.append(ep);seen.add(family_keys[ep]);phase_map[str(ep)]=times
            if len(chosen)==n:break
    assert len(chosen)==n and not set(chosen)&excluded;stat=resolve(cfg['dataset']).stat();final_manifest={**parent,'splits':{**parent['splits'],'test':sorted(chosen)},'phases':phase_map}
    record={'config':fcfg,'planning':planning,'families':sorted(chosen),'source_reserved_families':reserved,'all_reserved_retained':set(reserved).issubset(chosen),'planned_states':n*4,'source_phase_exclusions':rejected,'manifest':final_manifest,'frozen_inputs':frozen_inputs(),'dataset_file':{'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns},'dataset_archive_sha256':parent['dataset_archive_sha256'],'checkpoint_sha256':parent['checkpoint_sha256'],'created_unix':time.time(),'scope':'Final Cube expert only; policy/multimodel/hardware extensions remain separate.'}
    with LOCK.open('x') as handle:json.dump(record,handle,indent=2);handle.write('\n')
    print('LOCKED CUBE',n,'families',4*n,'states','reserved retained',record['all_reserved_retained'],flush=True)


def run():
    m=verify();cfg=load(resolve('configs/followup_cube.json'));out=BASE/'test';out.mkdir(exist_ok=True)
    import torch
    from scripts.run_followup_cube import Model,run_episode
    torch.set_num_threads(4);manifest=m['manifest'];model=Model(cfg,np.asarray(manifest['normalization']['mean']),np.asarray(manifest['normalization']['scale']))
    with h5py.File(resolve(cfg['dataset']),'r') as f:
        for ep in m['families']:run_episode(f,ep,'test',cfg,manifest,model,out)
    verify();assert len(list(out.glob('e*.json')))==m['planned_states'];write_json(BASE/'final_execution_complete.json',{'lock_sha256':sha256(LOCK),'states':m['planned_states'],'families':len(m['families']),'unix':time.time()})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run','continue-when-ready']);p.add_argument('--dependency-pid',type=int);args=p.parse_args()
    if args.mode=='freeze':freeze();return
    if args.mode=='continue-when-ready':
        freeze();assert args.dependency_pid;start=time.monotonic();gate=resolve('artifacts/followup_v1/final_expert/runtime_strong.done.json');print('WAITING for PushT timed test stage',flush=True)
        while not gate.exists():
            try:os.kill(args.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError('PushT parent ended before timed-test gate; inspect its failure, no automatic restart')
            if time.monotonic()-start>14400:raise RuntimeError('Four-hour observation window elapsed; revalidate dependency')
            time.sleep(10)
        from scripts.run_followup_final_expert import verify_lock
        verify_lock();assert load(gate)['lock_sha256']==sha256(resolve('artifacts/followup_v1/protocol/final_selection_lock.json'))
    run();subprocess.run([sys.executable,str(resolve('scripts/analyze_followup_cube_final.py'))],check=True)
if __name__=='__main__':main()
