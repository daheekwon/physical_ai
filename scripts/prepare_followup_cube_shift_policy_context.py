"""Validate causal command reconstruction for all audited Cube shifted histories."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from zj_wam.io import resolve,sha256,write_json
from scripts.run_followup_cube_native_policy import scene
from scripts.cube_shift_policy_context_core import compile_current
from scripts.analyze_followup_cube_plan_chunks import independent_compile
BASE=resolve('artifacts/followup_v1');SRC=BASE/'cube_shift_history';PARENT=BASE/'cube_policy_refinement';OUT=BASE/'cube_shift_policy_context';read=lambda p:json.loads(p.read_text())
def main():
    source_audit=read(SRC/'raw_audit.json');assert source_audit['condition_contexts']==336 and source_audit['fresh_history_RGB']==1008
    assert read(SRC/'pipeline_complete.json')['raw_audit_sha256']==sha256(SRC/'raw_audit.json')
    for p,h in {**read(SRC/'manifest.json')['inputs'],**source_audit['inputs']}.items():assert sha256(resolve(p))==h,p
    compiler=read(resolve('configs/followup_cube_plan_chunks.json'));native=read(PARENT/'manifest.json')['native'];paths=sorted(SRC.glob('*/*_t*.json'));assert len(paths)==336;OUT.mkdir(exist_ok=True)
    files=[SRC/'manifest.json',SRC/'raw_audit.json',SRC/'pipeline_complete.json',PARENT/'manifest.json',resolve('configs/followup_cube_plan_chunks.json'),resolve('scripts/cube_shift_policy_context_core.py'),Path(__file__),resolve('scripts/analyze_followup_cube_plan_chunks.py'),resolve('scripts/cube_plan_chunk_core.py'),resolve('zj_wam/cube_experiment.py')];files += [p.with_suffix(ext) for p in paths for ext in ['.json','.npz']]
    for seed in sorted({read(p)['seed'] for p in paths}):files += [PARENT/f'{seed}_nominal{ext}' for ext in ['.json','.npz']]
    manifest={'sources':[str(p.relative_to(resolve('.'))) for p in paths],'inputs':{str(p.resolve().relative_to(resolve('.'))):sha256(p) for p in files},'scope':'Prepare deployable current-observation H5+3 commands for336 already audited shifted development histories. Known initial privileged plan; current robot pose and past robot/gripper observations only. Material factors, object state, archived future actions and future physics are excluded from compile_current. No correction/allowance/risk tuning or adaptation efficacy claim.'}
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    else:write_json(OUT/'manifest.json',manifest)
    errors={};nominal_count=0;count=0;inputs={};cached={};environments={}
    def check(name,x,y,tol=0):
        x=np.asarray(x);y=np.asarray(y);np.testing.assert_allclose(x,y,atol=tol,rtol=0,err_msg=name);errors[name]=max(errors.get(name,0.),float(abs(x.astype(float)-y.astype(float)).max()) if x.size else 0.)
    for path in paths:
        meta=read(path);seed=meta['seed'];step=meta['step'];change=meta['change_step'];raw=dict(np.load(path.with_suffix('.npz')));assert meta['data_sha256']==sha256(path.with_suffix('.npz'))
        if seed not in cached:
            cached[seed]=dict(np.load(PARENT/f'{seed}_nominal.npz'));env,*_=scene(seed,native);environments[seed]=env
        source=cached[seed];env=environments[seed];m=env._model;data=mujoco.MjData(m);data.qpos[:]=raw['warmup_qpos'][-2];mujoco.mj_kinematics(m,data);mat=data.site_xmat[env._pinch_site_id].reshape(3,3);root=np.r_[data.site_xpos[env._pinch_site_id],np.arctan2(mat[1,0],mat[0,0])];gid=env._gripper_opening_joint_id
        bi=next(i for i in range(read(PARENT/f'{seed}_nominal.json')['blocks']) if int(source[f'block_{i:03d}_step'])==step);get=lambda k:source[f'block_{bi:03d}_{k}']
        before=np.clip(np.r_[raw['change_root_qpos'][gid],raw['warmup_qpos'][24::25,gid]]/.8,0,1)
        d=.5*(raw['past_actions']+1)*(source['action_high']-source['action_low'])+source['action_low'];newtargets=np.clip(before[:-1]+d[:,4],0,1)
        context={'past_points':raw['past_points'],'past_targets':raw['past_targets'],'gripper_history':np.r_[get('gripper_history')[:change],before],'gripper_targets':np.r_[get('gripper_targets')[:change],newtargets],'root_proprio':root,'root_time':float(raw['root_time'])}
        assert len(context['gripper_history'])==step+1 and len(context['gripper_targets'])==step
        result=compile_current(context,source['plan'],float(source['plan_t_init']),source['bounds'],source['action_low'],source['action_high'],compiler)
        independent,robot,grip=independent_compile(result['targets'],root,result['velocity'],float(before[-1]),result['gains'],float(result['gripper_gain']),source['bounds'],source['action_low'],source['action_high'],'past_pd_h5')
        check('independent_actions',result['actions'],independent);check('independent_robot',result['predicted_robot_control'],robot,1e-11);check('independent_gripper',result['predicted_gripper_control'],grip,1e-12)
        if meta['condition']=='nominal':
            check('original_root',root,get('root'),1e-12);check('original_gripper_history',context['gripper_history'],get('gripper_history'));check('original_gripper_targets',context['gripper_targets'],get('gripper_targets'),1e-12);check('original_commands',result['actions'],np.r_[get('nominal_actions'),get('suffix_actions')]);check('original_robot_gains',result['gains'],get('gains'),1e-8);check('original_gripper_gain',result['gripper_gain'],get('gripper_gain'),1e-12);nominal_count+=1
        dest=OUT/meta['condition']/path.name;dest.parent.mkdir(exist_ok=True);arrays={**{'input_'+k:np.asarray(v) for k,v in context.items()},**result};np.savez_compressed(dest.with_suffix('.npz'),**arrays);write_json(dest,{'seed':seed,'phase':meta['phase'],'step':step,'condition':meta['condition'],'history_source':str(path.relative_to(resolve('.'))),'manifest_sha256':sha256(OUT/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz'))})
        for p in [dest,dest.with_suffix('.npz')]:inputs[str(p.relative_to(resolve('.')))]=sha256(p)
        count+=1
    for env in environments.values():env.close()
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    assert count==336 and nominal_count==48;write_json(OUT/'raw_audit.json',{'contexts':count,'nominal_command_exact_states':nominal_count,'independent_command_rows':8*count,'max_errors':errors,'inputs':inputs,'manifest_sha256':sha256(OUT/'manifest.json'),'source_sha256':sha256(Path(__file__)),'scope':'Independent compiler equations and exact48originalnominal commands; every changed context reconstructed from previously audited past physical history. No fresh physics/RGB rerun in this command-preparation stage. Cached current robot site reconstructed from pre-integration qpos, gripper opening from past qpos; no future object state or material factors supplied.'});print('PREPARED CAUSAL SHIFT POLICY',count,nominal_count,flush=True)
if __name__=='__main__':main()
