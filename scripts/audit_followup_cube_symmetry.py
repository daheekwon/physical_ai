"""Visual Cube symmetry control; does not alter the full-rotation preservation metric."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,h5py,hdf5plugin
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,write_json,sha256
from zj_wam.cube_execution import make_env,initialize_recorded
from scripts.run_followup_cube import Model


def main():
 cfg=json.loads(resolve('configs/followup_cube.json').read_text());out=resolve(cfg['output']);manifest=json.loads((out/'manifest.json').read_text());dest=out/'symmetry';dest.mkdir(exist_ok=True);torch.set_num_threads(4);model=Model(cfg,np.asarray(manifest['normalization']['mean']),np.asarray(manifest['normalization']['scale']));rows=[]
 with h5py.File(resolve(cfg['dataset']),'r') as f:
  for path in sorted((out/'pilot').glob('e*.npz')):
   meta=json.loads(path.with_suffix('.json').read_text());ep=meta['episode'];offset=int(f['ep_offset'][ep]);env=make_env(cfg['seed']+ep)
   with np.load(path) as a:qpos=a['prefix_qpos'][-1].copy();qvel=a['prefix_qvel'][-1].copy()
   initialize_recorded(env,qpos,qvel,f['control'][offset+meta['timestep']],meta['timestep']*.05,f['privileged_target_block_pos'][offset]);joint=env._model.joint('object_joint_0').id;address=int(env._model.jnt_qposadr[joint]);rotation=Rotation.from_quat(qpos[address+3:address+7][[1,2,3,0]]);images=[];poses=[]
   for angle in [0.,90.,180.,270.,22.5]:
    changed=qpos.copy();r=rotation*Rotation.from_rotvec(np.array([0.,0.,np.deg2rad(angle)]));changed[address+3:address+7]=r.as_quat()[[3,0,1,2]];env.set_state(changed,qvel);images.append(env.render().copy());poses.append(changed[address:address+7])
   latent=model.encode(np.asarray(images)).cpu().numpy()[0]
   for i,angle in enumerate([0.,90.,180.,270.,22.5]):
    diff=np.abs(images[i].astype(float)-images[0].astype(float));rows.append({'state_id':meta['state_id'],'phase':meta['phase'],'local_rotation_deg':angle,'image_mae_255':float(diff.mean()),'image_max_abs_255':float(diff.max()),'changed_pixel_fraction':float(np.any(diff>0,axis=-1).mean()),'encoded_latent_l2':float(np.linalg.norm(latent[i]-latent[0]))})
   np.savez_compressed(dest/(meta['state_id']+'.npz'),images=np.asarray(images),latent=latent,object_poses=np.asarray(poses));env.close()
 write_json(dest/'audit.json',{'rows':rows,'source_sha256':sha256(Path(__file__)),'scope':'Same robot/objectposition/background/renderer; rotate uniform40mmcube about localz by90/180/270degrees, with22.5degree nonsymmetry control. Render only, no physics intervention. Physical trajectory E_obj retains full quaternion rotation error, not quotient symmetry. Ambiguous absolute orientation readout cannot alone diagnose WM capture failure.'});print(rows,flush=True)
if __name__=='__main__':main()
