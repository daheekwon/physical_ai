"""Independent long-suffix physical audit and finite-family diagnostic comparison."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
from zj_wam.adapters import ExpertDataset
from scripts.analyze_followup_final_expert import Coverage,physical
from scripts.analyze_followup_response_selection import mean_ci

BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();out=BASE/('terminal_residual_smoke' if args.smoke else 'terminal_residual');m=read(out/'manifest.json');assert m['smoke']==args.smoke;cfg=m['main'];tcfg=m['config'];data=ExpertDataset(cfg['dataset']);errors={};rows=[];states=0
    complete=read(out/'execution_complete.json');assert complete['manifest_sha256']==sha256(out/'manifest.json')
    for path,h in m['inputs'].items():assert sha256(resolve(path))==h,path
    def check(name,x,y,tol=1e-7):
        error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),error);assert error<=tol,(name,error)
    for split in tcfg['splits']:
        paths=sorted((out/split).glob('*_h05.json'));assert len(paths)==(1 if args.smoke else 105)
        for path in paths:
            meta=read(path);assert meta['manifest_sha256']==sha256(out/'manifest.json');f=pd.read_csv(path.with_suffix('.csv'));assert len(f)==24 and set(f.method)==set(tcfg['families']);cov=Coverage(data,meta,cfg);states+=1
            for parent,h in meta['parent_sha256'].items():assert sha256(resolve(parent))==h,parent
            old=pd.read_csv(BASE/'family_ablation'/path.with_suffix('.csv').name).set_index('index')
            with np.load(path.with_suffix('.npz')) as a,np.load(BASE/'family_ablation'/path.with_suffix('.npz').name) as parent:
                ref=a['physics'][0];assert ref.shape==(260,10);check('old_nominal_prefix_and_chunk',ref[:110],parent['physics'][0],0.)
                for row in f.itertuples():
                    trace=a['physics'][row.index];assert trace.shape==ref.shape;check('fixed_prefix',trace[:30],ref[:30],0.);check('old_short_trace',trace[:110],parent['physics'][row.parent_index],1e-8);check('old_actions',a['actions'][row.index],parent['actions'][row.parent_index],1e-9)
                    check('action_bound',np.maximum(0,np.abs(a['actions'][row.index])-1),0.,1e-6);norm=np.linalg.norm((a['actions'][row.index]-a['actions'][0])/data.scaler.scale_);check('norm',norm,row.norm);assert row.active==(norm>1e-6)
                    local=physical(trace[:110],ref[:110],5,cov,cfg);long=physical(trace,ref,5,cov,cfg)
                    check('shortE',local['object_error'],row.short_object_error);check('short_gain',local['jerk_gain'],row.short_jerk_gain);check('old_shortE',local['object_error'],old.loc[row.parent_index,'object_error']);check('long_nested_E',max(local['object_error'],long['object_error']),row.long_object_error);check('long_gain',long['jerk_gain'],row.long_jerk_gain)
                    check('endpoint_coverage',[cov(trace[79]),cov(trace[-1])],a['coverage'][row.index],1e-10)
                    d=trace[30:]-ref[30:];pq=np.linalg.norm(d[49,:2]);pv=np.linalg.norm(d[49,5:7]);check('pusher_terminal_q',pq,row.terminal_pusher_position);check('pusher_terminal_v',pv,row.terminal_pusher_velocity);check('pusher_postcorrection',np.linalg.norm(d[50:,:2],axis=1).max(),row.post_correction_pusher_max)
                    if row.method.endswith('_terminal'):assert max(pq,pv)<=1e-3
                    for step in [5,8,23]:
                        x=d[step*10-1]
                        for key,value in [('position',np.linalg.norm(x[2:4])),('velocity',np.linalg.norm(x[7:9])),('angular_velocity',abs(x[9]))]:check('object_'+key+'_'+str(step),value,getattr(row,'object_'+key+'_'+str(step)))
            rows.append(f)
    f=pd.concat(rows,ignore_index=True);f['short_safe']=f.short_object_error<=1;f['long_safe']=f.long_object_error<=1;f['short_useful']=f.active&f.short_safe&(f.short_jerk_gain>=.1);f['carryover_useful']=f.short_useful&f.long_safe;f['full_useful']=f.active&f.long_safe&(f.long_jerk_gain>=.1);state_rows=[]
    for (split,state,method),g in f.groupby(['split','state_id','method']):
        active=g[g.active];safe=active[active.short_safe];useful=g[g.short_useful];longgood=active[active.long_safe];shortgood=active[active.short_safe];dimension=int(g.dimension.iloc[0]);assert g.dimension.eq(dimension).all()
        state_rows.append({'split':split,'state_id':state,'method':method,'initial_pose_family':int(g.initial_pose_family.iloc[0]),'dimension':dimension,'active_fraction':float(g.active.mean()),'short_oracle_U':bool(g.short_useful.any()),'carryover_oracle_U':bool(g.carryover_useful.any()),'full_oracle_U':bool(g.full_useful.any()),'short_oracle_gain':max(0.,float(shortgood.short_jerk_gain.max())) if len(shortgood) else 0.,'full_oracle_gain':max(0.,float(longgood.long_jerk_gain.max())) if len(longgood) else 0.,'long_safe_given_short':float(safe.long_safe.mean()) if len(safe) else np.nan,'long_safe_given_useful':float(useful.long_safe.mean()) if len(useful) else np.nan,'post_pusher_max':float(active.post_correction_pusher_max.mean()) if len(active) else np.nan,'end_object_velocity':float(active.object_velocity_5.mean()) if len(active) else np.nan,'candidate_long_risk':float((~active.long_safe).mean()) if len(active) else np.nan})
    state=pd.DataFrame(state_rows);summary=[];paired=[];metrics=['active_fraction','short_oracle_U','carryover_oracle_U','full_oracle_U','short_oracle_gain','full_oracle_gain','long_safe_given_short','long_safe_given_useful','post_pusher_max','end_object_velocity','candidate_long_risk']
    for (split,method),g in state.groupby(['split','method']):summary.append({'split':split,'method':method,'dimension':int(g.dimension.iloc[0]),'states':len(g),**{key:mean_ci(g.dropna(subset=[key]),key) for key in metrics}})
    for split in tcfg['splits']:
        for family in ['quintic_full','interior_values','cubic_spline']:
            x=state[(state.split==split)&(state.method==family+'_terminal')].merge(state[(state.split==split)&(state.method==family)],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
            for key in ['short_oracle_U','carryover_oracle_U','full_oracle_U','full_oracle_gain']:x['delta_'+key]=x[key].astype(float)-x[key+'_base'].astype(float)
            paired.append({'split':split,'family':family,'terminal_minus_unconstrained':{key:mean_ci(x,'delta_'+key) for key in ['short_oracle_U','carryover_oracle_U','full_oracle_U','full_oracle_gain']}})
    f.to_csv(out/'evaluation.csv',index=False);state.to_csv(out/'state_metrics.csv',index=False)
    write_json(out/'analysis.json',{'smoke':args.smoke,'summary':summary,'paired':paired,'scope':tcfg['scope']+' '+tcfg['interpretation']+' Conditional survival is state-equal among states with eligible candidates; denominators and family counts explicit. Short jerk and entire23action jerk are distinct endpoints. Finite oracle only, no deployable selector.'})
    write_json(out/'raw_audit.json',{'smoke':args.smoke,'states':states,'candidate_rows':len(f),'max_error':errors,'manifest_sha256':sha256(out/'manifest.json'),'source_sha256':sha256(Path(__file__)),'scope':'Independent dense same-time object error and polygoncoverage at5/8/23, original first8physics, fixedprefix/bounds/norms, fullwindow jerk, actual terminalrobotq/v, postcorrectionpusher andobjectvelocities. Allparent hashes unchanged.'})
    print('AUDITED terminal residual',states,'states',len(f),'long candidate branches',flush=True)
if __name__=='__main__':main()
