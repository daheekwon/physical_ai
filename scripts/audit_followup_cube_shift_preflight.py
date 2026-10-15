"""Independent model mutation, fresh branch and physical-effect audit for Cube shifts."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_execution import DenseCube,snapshot,restore
from scripts.run_followup_cube_native_policy import scene
from scripts.audit_followup_dinowm_policy_final import interval
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def effect(pose,reference):
    position=np.linalg.norm(pose[:,:3]-reference[:,:3],axis=1).max();q=Rotation.from_quat(pose[:,[4,5,6,3]]);q0=Rotation.from_quat(reference[:,[4,5,6,3]]);angle=(q*q0.inv()).magnitude().max();return float(position),float(angle)
def main(smoke):
    out=BASE/('cube_shift_preflight_smoke' if smoke else 'cube_shift_preflight');manifest=read(out/'manifest.json');cfg=manifest['config'];done=read(out/'execution_complete.json');assert done['smoke']==smoke and done['manifest_sha256']==sha256(out/'manifest.json')
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    parent=BASE/'cube_policy_refinement';native=read(parent/'manifest.json')['native'];paths=[resolve(p) for p in manifest['sources']];families=sorted({read(p)['seed'] for p in paths});errors={};rows=[];inputs={};branches=0
    def check(name,a,b,tol=0):
        a=np.asarray(a);b=np.asarray(b);np.testing.assert_allclose(a,b,atol=tol,rtol=0,err_msg=name);errors[name]=max(errors.get(name,0.),float(np.max(abs(a.astype(float)-b.astype(float)))) if a.size else 0.)
    for seed in families:
        original=dict(np.load(parent/f'{seed}_nominal.npz'));env,obs,info,_,_=scene(seed,native);agent=CubePlanOracle(env,segment_dt=native['plan_segment_dt'],noise=native['plan_noise']);np.random.seed(seed+1000000);agent.reset(obs,info);meter=DenseCube(env);m=env._model;mass=m.body_mass.copy();inertia=m.body_inertia.copy();friction=m.geom_friction.copy();geoms=np.array(env._cube_geom_ids_list[0],int);bodies=np.unique(m.geom_bodyid[geoms]);current=0
        for p in sorted([p for p in paths if read(p)['seed']==seed],key=lambda p:read(p)['step']):
            meta=read(p);t=meta['step'];dest=out/p.name;saved=read(dest);raw=dict(np.load(dest.with_suffix('.npz')));source=dict(np.load(p.with_suffix('.npz')));assert saved['manifest_sha256']==sha256(out/'manifest.json') and saved['data_sha256']==sha256(dest.with_suffix('.npz'));check('original_mass',raw['baseline_mass'],mass);check('original_inertia',raw['baseline_inertia'],inertia);check('original_friction',raw['baseline_friction'],friction);check('object_body',raw['body_ids'],bodies);check('object_geom',raw['geom_ids'],geoms)
            if t>current:
                trace=meter.rollout(original['actions'][current:t])['physics']
                for key,value in trace.items():check('prefix_'+key,value,original['physics_'+key][current*25:t*25])
                current=t
            root=snapshot(env);action=np.concatenate((source['actions'][0],source['input_suffix_actions']));check('same_actions',action,raw['actions']);sham_position,sham_rotation=effect(raw['sham_forward_r0_object_pose'],raw['nominal_r0_object_pose'])
            for condition in cfg['conditions']:
                name=condition['name']
                for repeat in range(cfg['repeats']):
                    tag=f'{name}_r{repeat}_';m.body_mass[:]=mass;m.body_inertia[:]=inertia;m.geom_friction[:]=friction
                    for body in bodies:m.body_mass[body]=mass[body]*condition['mass_factor'];m.body_inertia[body]=inertia[body]*condition['mass_factor']
                    for geom in geoms:m.geom_friction[geom,0]=friction[geom,0]*condition['friction_factor']
                    mujoco.mj_setConst(m,mujoco.MjData(m));restore(env,root)
                    if condition['forward']:mujoco.mj_forward(m,env._data)
                    for key,value in [('mass',m.body_mass[bodies]),('inertia',m.body_inertia[bodies]),('friction',m.geom_friction[geoms]),('root_qpos',env._data.qpos),('root_qvel',env._data.qvel),('root_time',env._data.time)]:check('parameter_'+key,value,raw[tag+'parameter_'+key])
                    check('same_root_qpos',env._data.qpos,root['data'].qpos);check('same_root_qvel',env._data.qvel,root['data'].qvel);check('same_clock',env._data.time,root['data'].time);trace=meter.rollout(action)['physics'];branches+=1
                    for key,value in trace.items():
                        check('fresh_branch_'+key,value,raw[tag+key]);check('repeat_'+key,raw[tag+key],raw[name+'_r0_'+key])
                        if name=='nominal':check('parent_nominal_'+key,value,source['candidate_'+key][0])
                pos,rot=effect(raw[name+'_r0_object_pose'],raw['nominal_r0_object_pose']);eef=np.linalg.norm(raw[name+'_r0_eef_position']-raw['nominal_r0_eef_position'],axis=1).max();pos_floor=max(cfg['minimum_position_effect_m'],cfg['sham_multiplier']*sham_position);rot_floor=max(cfg['minimum_rotation_effect_rad'],cfg['sham_multiplier']*sham_rotation);rows.append({'state_id':p.stem,'initial_pose_family':seed,'phase':meta['phase'],'condition':name,'object_position_effect_m':pos,'object_rotation_effect_rad':rot,'eef_position_effect_m':float(eef),'sham_position_m':sham_position,'sham_rotation_rad':sham_rotation,'position_effect_threshold_m':pos_floor,'rotation_effect_threshold_rad':rot_floor,'effective_object_change':bool(pos>pos_floor or rot>rot_floor),'contact_sample_mismatch_fraction':float(np.mean(raw[name+'_r0_robot_object_contacts']!=raw['nominal_r0_robot_object_contacts']))})
            m.body_mass[:]=mass;m.body_inertia[:]=inertia;m.geom_friction[:]=friction;mujoco.mj_setConst(m,mujoco.MjData(m));restore(env,root)
            for q in [dest,dest.with_suffix('.npz')]:inputs[str(q.relative_to(resolve('.')))]=sha256(q)
            print('AUDIT CUBE SHIFT',seed,meta['phase'],flush=True)
        env.close()
    f=pd.DataFrame(rows);summary=[]
    for condition,x in f.groupby('condition'):
        for phase in ['all']+sorted(x.phase.unique()):
            g=x if phase=='all' else x[x.phase==phase];summary.append({'condition':condition,'phase':phase,'states':len(g),**{k:interval(g,k) for k in ['object_position_effect_m','object_rotation_effect_rad','eef_position_effect_m','effective_object_change','contact_sample_mismatch_fraction']}})
    assert branches==done['branches'] and len(f)==done['states']*len(cfg['conditions']);f.to_csv(out/'effects.csv',index=False);write_json(out/'analysis.json',{'smoke':smoke,'states':done['states'],'families':len(families),'branches':branches,'rows':len(f),'summary':summary,'maximum_sham_position_m':float(f.sham_position_m.max()),'maximum_sham_rotation_rad':float(f.sham_rotation_rad.max()),'scope':cfg['scope']});write_json(out/'raw_audit.json',{'smoke':smoke,'states':done['states'],'families':len(families),'fresh_physical_branches':branches,'rows':len(f),'max_errors':errors,'inputs':inputs,'manifest_sha256':sha256(out/'manifest.json'),'analysis_sha256':sha256(out/'analysis.json'),'effects_sha256':sha256(out/'effects.csv'),'source_sha256':sha256(Path(__file__)),'audit_helpers':{p:sha256(resolve(p)) for p in ['scripts/audit_followup_dinowm_policy_final.py']},'scope':'Independent parameter assignments/constants/root restoration, all fresh duplicate physical branches, originalnominal exact reproduction, SO3/position effects and sham-relative effectivecondition classification. Condition validity only.'});print('AUDITED CUBE SHIFT',done['states'],branches,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
