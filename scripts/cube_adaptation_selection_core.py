"""Select from a precomputed Cube bank; used only for offline calibration/audit."""
import numpy as np

def select(costs,norms,scores,labels,error,rule,cfg,mode,alpha=0.,constant_radius=None):
    radius=cfg['base_radius']/(1+alpha*min(error,cfg['residual_cap'])) if mode in ['allowance','both'] else cfg['base_radius']
    if mode=='constant_shrink':radius=float(constant_radius)
    stage1=np.array([i for i,x in enumerate(labels) if x[0] in ['nominal','past_pd_optimizer']],int);gain=(costs[0]-costs)/max(costs[0],1e-12)
    def best(ids):
        if not rule['enabled']:return 0
        eligible=[i for i in ids if norms[i]>cfg['minimum_norm'] and norms[i]<=radius+1e-6 and scores[i]<=rule['tau'] and gain[i]>cfg['minimum_predicted_gain']]
        return int(min(eligible,key=lambda i:(costs[i],norms[i],i))) if eligible else 0
    first=best(stage1);early=bool(mode in ['compute','both'] and first>0 and gain[first]>=cfg['early_stop_gain'] and gain[first]>=cfg['early_stop_fraction']*gain[stage1].max());ids=stage1 if mode=='cheap_only' or early else np.arange(len(costs));return {'candidate':best(ids),'radius':float(radius),'early_stop':early,'stages':1 if mode=='cheap_only' or early else 2,'scored_candidates':len(ids)}
