"""Read official PushT initial-family metadata without evaluating held-out actions."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,zarr
from zj_wam.adapters import ExpertDataset
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main():
    archive=resolve('data/datasets/diffusion-policy-official/pusht.zip');digest=sha256(archive);assert digest=='63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9'
    store=zarr.ZipStore(str(archive),mode='r');root=zarr.open_group(store,path='pusht/pusht_cchi_v7_replay.zarr',mode='r');ends=root['meta/episode_ends'][:];starts=np.r_[0,ends[:-1]];x=root['data/state'].oindex[starts,np.arange(5)].astype(float);assert len(x)==206 and len(np.unique(x,axis=0))==206
    cfg=read(resolve('configs/followup_v1.json'));data=ExpertDataset(cfg['dataset']);old=data.states[np.asarray(data.dataset.offsets,int),:5].astype(float);old=np.unique(old,axis=0);assert len(old)==185
    def distances(a,b):
        d=a[:,None,:]-b[None,:,:];d[:,:,4]=np.arctan2(np.sin(d[:,:,4]),np.cos(d[:,:,4]));return np.max(abs(d),axis=2)
    dist=distances(x,old);overlap=dist.min(1)<=1e-6;assert overlap.sum()==185 and len(np.unique(dist.argmin(1)[overlap]))==185
    prior=[];inputs={str(resolve(cfg['dataset']).relative_to(resolve('.'))):sha256(resolve(cfg['dataset']))}
    for folder in ['dp_native','policy_refinement/development','policy_refinement/calibration','policy_refinement/test','policy_latency/development','policy_latency/test','dinowm_policy_nominal/development','dinowm_policy_nominal/calibration']:
        for p in sorted((BASE/folder).glob('*.npz')):
            # Ongoing nominal collection is not part of the exposure freeze; skip
            # any artifact that has not acquired its corresponding completion JSON.
            if not p.with_suffix('.json').exists():continue
            with np.load(p) as a:
                if 'initial_state' in a:prior.append(a['initial_state'][:5]);inputs[str(p.relative_to(resolve('.')))]=sha256(p)
    fresh=np.flatnonzero(~overlap);assert len(fresh)==21
    if prior:assert np.min(distances(x[fresh],np.array(prior)))>1e-6
    manifest=read(BASE/'dinowm_policy_nominal/manifest.json');newpolicy=np.array([v['initial_state'][:5] for v in manifest['initials'].values()]);assert np.min(distances(x[fresh],newpolicy))>1e-6
    rows=[{'episode':int(i),'start':int(starts[i]),'length':int(ends[i]-starts[i]),'initial_pose':x[i].tolist(),'nearest_previous_max_coordinate_difference':float(dist[i].min())} for i in fresh]
    record={'source_url':'https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip','archive_sha256':digest,'archive_bytes':archive.stat().st_size,'official_episodes':206,'official_unique_initial_families':206,'previous_unique_families':185,'matched_previous_families':185,'new_initial_families':21,'reserved_expert_episodes':rows,'all_episode_initial_poses':x.tolist(),'old_family_match_for_official_episode':np.where(overlap,dist.argmin(1),-1).tolist(),'prior_artifact_inputs':inputs,'new_policy_manifest_sha256':sha256(BASE/'dinowm_policy_nominal/manifest.json'),'source_sha256':sha256(Path(__file__)),'unix':time.time(),'scope':'Initial-pose and episode-boundary audit only.21families absent from existing185familyHDF5 and recorded policy initial conditions. No heldout corrected rollout, model scoring, success analysis or threshold selection. Original archive contains absolute target actions and float32poses; validate adapter on185alreadyexposed families before locking21family expert evaluation. This is new downstream initial-family coverage, not proof of absence from all model/policy pretraining.'}
    store.close();write_json(BASE/'protocol/original_expert_new_families.json',record);print('VERIFIED official206 = previous185 + new21 initialfamilies',flush=True)
if __name__=='__main__':main()
