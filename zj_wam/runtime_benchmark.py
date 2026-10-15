"""Time the actual causal refinement path, with separate diagnostic stage profiles."""
import contextlib,time
import numpy as np
import torch
from zj_wam import repeated_refinement as repeated
from zj_wam import policy_refinement as policy
from zj_wam import followup


@contextlib.contextmanager
def stages(model):
    measured={};originals=[];counts={'optimizer_calls':0,'optimizer_iterations':0,'solver_failures':0,'wm_predict_batches':0,'wm_encode_calls':0}
    def patch(obj,name,key):
        original=getattr(obj,name);originals.append((obj,name,original))
        def wrapped(*args,**kwargs):
            torch.cuda.synchronize(model.device);start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                torch.cuda.synchronize(model.device);measured[key]=measured.get(key,0.)+time.perf_counter()-start
        setattr(obj,name,wrapped)
    patch(repeated,'visual_history','pose_estimation');patch(repeated,'generate_bank','candidate_generation');patch(policy,'generate_bank','candidate_generation');patch(model,'predictor','image_preprocess_encoder');patch(repeated,'predict_candidates','world_model_prediction');patch(repeated,'score_models','contact_model_prediction')
    def count_patch(obj,name,key):
        original=getattr(obj,name);originals.append((obj,name,original))
        def wrapped(*args,**kwargs):
            counts[key]+=1;result=original(*args,**kwargs)
            if key=='optimizer_calls':counts['optimizer_iterations']+=int(result.nit);counts['solver_failures']+=int(not result.success)
            return result
        setattr(obj,name,wrapped)
    count_patch(followup,'minimize','optimizer_calls');count_patch(model.model,'predict','wm_predict_batches');count_patch(model.model,'encode','wm_encode_calls')
    try:yield measured,counts
    finally:
        for obj,name,original in reversed(originals):setattr(obj,name,original)


def causal_inputs(proprio,prefix):
    robot=np.zeros_like(proprio);robot[:,[0,1,5,6]]=proprio[:,[0,1,5,6]]
    past=np.zeros_like(prefix);past[:,[0,1,5,6]]=prefix[:,[0,1,5,6]]
    return robot,past


def measure(env,model,method,images,past,proprio,prefix,nominal,scale,seed,cfg,rcfg,rules,strong):
    torch.cuda.synchronize(model.device);resident=torch.cuda.memory_allocated(model.device);torch.cuda.reset_peak_memory_stats(model.device);start=time.perf_counter();current=env.render().copy();render_seconds=time.perf_counter()-start
    frames=np.stack([images[0],images[1],current]);robot,trace=causal_inputs(proprio,prefix)
    result,info=policy.refine(method,model,frames,past,robot,trace,nominal,scale,seed,cfg,rcfg,rules,strong)
    torch.cuda.synchronize(model.device);elapsed=time.perf_counter()-start
    return result,info,{'total_seconds':elapsed,'render_seconds':render_seconds,'resident_gpu_bytes':resident,'additional_peak_gpu_bytes':max(0,torch.cuda.max_memory_allocated(model.device)-resident)},current
