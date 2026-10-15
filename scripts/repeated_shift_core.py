"""Fixed-onset shifted repeated expert execution; original selectors remain frozen."""
import time
import numpy as np,pandas as pd
from zj_wam.shifted_execution import ShiftedSimulator,noisy_history
from zj_wam.repeated_refinement import choose_correction,generate_bank,visual_history
from zj_wam.followup import controller_rollout,dense_metrics
from zj_wam.diagnostics import jerk_energy
from zj_wam.selection import select_indices
from zj_wam.locked_geometric import refine
from scripts.run_followup_repeated_refinement import execute_with_coverage,core_metrics
STRONG={'gaussian_action','lipo_full_command','groove_full_command'}

def simulator(actions,recorded,current,initial,seed,condition):
    sim=ShiftedSimulator(actions,recorded,current,seed,condition=condition,lead_actions=10)
    sim.shift_step=max(0,initial-10)
    return sim

def select(method,model,images,past,proprio,prefix,nominal,suffix,scale,seed,cfg,rcfg,rules,strong):
    started=time.perf_counter()
    if method in STRONG:
        selected,info=refine(method,nominal,suffix,proprio[-1],prefix,past,scale,strong[method]);return selected,{**info,'seconds':time.perf_counter()-started}
    if method!='controller_pusher':
        chosen,info,_=choose_correction(model,images,past,proprio,prefix,nominal,suffix,scale,seed,cfg,rcfg,rules,method)
        return chosen,info
    # Robot-only baseline uses the identical non-WM candidate generator.
    root=np.zeros_like(proprio[-1]);root[[0,1,5,6]]=proprio[-1,[0,1,5,6]]
    actions,labels=generate_bank(nominal,suffix,root,prefix,past,scale,seed,cfg['radii'])
    motion=np.array([controller_rollout(np.r_[a,suffix],root[:2],root[5:7]) for a in actions]);cost=np.array([jerk_energy(np.r_[prefix[:,:2],q],.01) for q in motion]);score=np.linalg.norm(motion-motion[:1],axis=2).max(1);norm=np.linalg.norm(((actions-actions[:1])/scale).reshape(len(actions),-1),axis=1)
    f=pd.DataFrame({'state_id':'live','horizon':5,'actual_norm':norm,'predicted_jerk':cost,'nominal_predicted_jerk':cost[0],method+'_score':score});idx=max(0,select_indices(f,method,rules[method])['live'])
    return actions[idx],{'selected_index':idx,'selected_generator':labels[idx],'active':bool(norm[idx]>1e-6),'norm':float(norm[idx]),'score':float(score[idx]),'predicted_gain':float((cost[0]-cost[idx])/max(cost[0],1e-8)),'candidate_count':len(actions)-1,'seconds':time.perf_counter()-started}

def clean_robot_proprio(proprio):
    p=np.zeros_like(proprio);p[:,[0,1,5,6]]=proprio[:,[0,1,5,6]];return p

def branch(original,recorded,root,seed,condition,raw,suffix):
    sim=simulator(original,recorded,root,root,seed,condition)
    try:return execute_with_coverage(sim,raw,suffix)
    finally:sim.close()

def run_method(original,recorded,meta,condition,condition_index,method,model,data,cfg,rcfg,rules,strong,save_decision):
    root=meta['timestep'];h=20;seed=cfg['seed']+meta['episode'];delivered=original.copy();logs=[];local=[];decision_arrays={}
    for offset in range(0,h,5):
        current=root+offset;sim=simulator(delivered,recorded,current,root,seed,condition)
        try:
            clean,past=sim.restore(capture=True);images=noisy_history(clean,condition,np.random.SeedSequence([cfg['seed'],meta['episode'],current,condition_index]));proprio=clean_robot_proprio(sim.context_control);prefix=sim.prefix_physics.copy();nominal=original[current:current+5];suffix=original[current+5:current+8]
            a,info=select(method,model,images,past,proprio,prefix,nominal,suffix,data.scaler.scale_,np.random.SeedSequence([cfg['seed'],meta['episode'],current,5]),cfg,rcfg,rules,strong)
            delivered[current:current+5]=a;applied=delivered[current:current+5].copy();norm=float(np.linalg.norm((applied-nominal)/data.scaler.scale_));info={**info,'offset':offset,'applied_norm':norm,'applied_active':bool(norm>1e-6),'shift_step':sim.shift_step}
            arrays={'images':images,'clean_images':clean,'past':past,'proprio':proprio,'prefix':prefix,'nominal':nominal,'suffix':suffix,'applied':applied,'root':sim.root_full_state.copy()}
            save_decision(offset,info,arrays)  # Persist selection before future physical branches.
            ref=sim.execute_dense(nominal,suffix);actual=sim.execute_dense(applied,suffix)
            local.append({'offset':offset,'active':bool(norm>1e-6),**dense_metrics(actual,ref,cfg['tolerances'])});logs.append(info)
            for k,v in arrays.items():decision_arrays[f'block{offset:02d}_{k}']=v
            for k in ['physics','control','physics_trace','control_trace','physics_contacts']:
                decision_arrays[f'block{offset:02d}_nominal_{k}']=ref[k];decision_arrays[f'block{offset:02d}_actual_{k}']=actual[k]
            decision_arrays[f'block{offset:02d}_coverage']=np.array([[r['coverage_t'],r['coverage_suffix']] for r in [ref,actual]])
        finally:sim.close()
    suffix=original[root+h:root+h+10];assert len(suffix)>=3
    result=branch(original,recorded,root,seed,condition,delivered[root:root+h],suffix)
    return delivered[root:root+h],result,logs,local,decision_arrays
