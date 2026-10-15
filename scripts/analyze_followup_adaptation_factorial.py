"""Audit saved decisions/physical traces, lock nominal-only factors and summarize shifts."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json
from zj_wam.adaptation_factorial import pick,past_holdout_error
from zj_wam.diagnostics import jerk_energy
from scripts.analyze_followup_response_selection import mean_ci


def load_audit(directory,cfg,acfg,rule,rcfg,expected,partial):
    paths=sorted(directory.glob('*_f*_e*_t*.json'))
    if not partial:assert len(paths)==expected,(directory,len(paths),expected)
    frames=[];errors={};counts={'states':len(paths),'rows':0}
    def check(name,x,y,tol=1e-7):
        e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),e)
        assert e<=tol,(name,e)
    for path in paths:
        meta=json.loads(path.read_text());frame=pd.read_csv(path.with_suffix('.csv'));frames.append(frame)
        with np.load(path.with_suffix('.npz')) as a:
            for row in frame.itertuples(index=False):
                tag=row.condition+'__'+row.variant;d=meta['decisions'][tag];actions=a[tag+'_actions'];cost=a[tag+'_costs'];norm=a[tag+'_norms'];score=a[tag+'_scores'];obs=a[tag+'_observed'];ref=a[row.condition+'_nominal_physics'];actual=a[tag+'_physics'];eps=cfg['tolerances']
                if d['stages']:
                    radius=rule['relative_radius']
                    if row.method in ['allowance','both']:
                        residual,parts=past_holdout_error(obs,a[row.condition+'_past_actions'],rcfg);check('past_holdout_error',residual,d['past_error']);check('past_holdout_components',parts,d['past_error_components']);radius/=1+row.alpha*min(residual,acfg['residual_cap'])
                    elif row.method=='constant_shrink':radius=row.radius
                    check('effective_radius',radius,row.radius)
                    idx=pick(cost,norm,score,True,rule['tau'],radius);assert idx==d['index']==row.index
                    nfirst=1+len(cfg['radii']);i=pick(cost[:nfirst],norm[:nfirst],score[:nfirst],True,rule['tau'],radius);g=(cost[0]-cost[i])/max(cost[0],1e-8);upper=(cost[0]-cost[:nfirst].min())/max(cost[0],1e-8)
                    early=row.method in ['compute','both'] and i>0 and g>=acfg['early_stop_gain'] and g>=acfg['early_stop_fraction']*upper
                    assert bool(early)==bool(d['early_stop']);assert d['stages']==(1 if early or row.method=='cheap_only' else 2)
                    assert d['scored_candidates']==len(actions)+(1 if d['stages']==2 else 0)
                    check('predicted_gain',(cost[0]-cost[idx])/max(cost[0],1e-8),row.predicted_gain)
                else:assert not row.vision_valid and row.index==0 and not row.active
                check('unchanged_prefix',actual[:30],ref[:30],1e-9)
                position=np.linalg.norm(actual[30:,2:4]-ref[30:,2:4],axis=1).max();delta=actual[30:,4]-ref[30:,4];rotation=np.abs(np.arctan2(np.sin(delta),np.cos(delta))).max();loss=np.maximum(0,a[row.condition+'_nominal_coverage']-a[tag+'_coverage']);e=max(position/eps['position'],rotation/eps['rotation'],loss.max()/eps['coverage_loss'])
                j0=jerk_energy(ref[:,:2],.01);j=jerk_energy(actual[:,:2],.01);gain=(j0-j)/j0 if j0>1e-8 else 0.
                check('executed_E',e,row.object_error);check('executed_jerk_gain',gain,row.jerk_gain);check('selected_norm',norm[row.index],row.actual_norm)
                assert bool(row.preserved)==bool(e<=1)
                assert np.max(np.abs(actions))<=1+1e-6
                check('polynomial_endpoint',actions[:,[0,-1]]-actions[0:1,[0,-1]],0.,1e-8)
    f=pd.concat(frames,ignore_index=True);f['joint']=f.active & f.preserved & (f.jerk_gain>=cfg['meaningful_jerk_reduction']);f['violation']=~f.preserved;counts['rows']=len(f)
    return f,{'counts':counts,'max_absolute_recalculation_error':errors,'scope':'Reconstructed selected index, radius and causal past residual, early-stop decision and actually evaluated candidates; raw100Hz E and jerk; saved coverage endpoints. Candidate equality and response-score equality to original shift bank checked during collection.'}


def tune(f,cfg,acfg,signature):
    def measure(g):
        risks=[]
        for x in [g,g[g.nominal_contact]]:
            active=x[x.active];risks.append(float(active.violation.mean()) if len(active) else 1.)
        return {'joint':float(g.joint.mean()),'gain':float(g.jerk_gain.mean()),'active':float(g.active.mean()),'risk':risks[0],'contact_risk':risks[1]}
    choices=[]
    for alpha in acfg['alpha_grid']:
        metrics={m:measure(f[f.variant==f'{m}_a{alpha:g}']) for m in ['allowance','both']}
        feasible=all(v['active']>=cfg['minimum_active_coverage'] and max(v['risk'],v['contact_risk'])<=cfg['target_violation_rate'] for v in metrics.values())
        choices.append({'alpha':alpha,'feasible':feasible,'metrics':metrics,'joint':np.mean([v['joint'] for v in metrics.values()]),'worst_risk':max(max(v['risk'],v['contact_risk']) for v in metrics.values())})
    feasible=[x for x in choices if x['feasible']]
    best=max(feasible,key=lambda x:(x['joint'],-x['alpha'])) if feasible else min(choices,key=lambda x:(x['worst_risk'],-x['joint'],x['alpha']))
    constants=[]
    for r in acfg['constant_radius_grid']:
        m=measure(f[f.variant==f'constant_r{r:g}']);valid=m['active']>=cfg['minimum_active_coverage'] and max(m['risk'],m['contact_risk'])<=cfg['target_violation_rate'];constants.append({'radius':r,'feasible':valid,**m})
    valid=[x for x in constants if x['feasible']];bestc=max(valid,key=lambda x:(x['joint'],x['radius'])) if valid else min(constants,key=lambda x:(max(x['risk'],x['contact_risk']),-x['joint']))
    return {'signature':signature,'alpha':best['alpha'],'constant_radius':bestc['radius'],'alpha_feasible':best['feasible'],'constant_feasible':bestc['feasible'],'alpha_search':choices,'constant_search':constants,'calibration_states':f.state_id.nunique(),'calibration_families':f.initial_pose_family.nunique(),'scope':acfg['calibration'],'warning':'An alpha of zero is an allowed null result: nominal calibration found no benefit from shrinking with this past-residual proxy. Empirical risk is not a guarantee under shifts.'}


def summarize(f,acfg):
    summary=[];paired=[];factorial=[]
    for (split,condition,method),g in f.groupby(['split','condition','method']):
        active=g[g.active];valid=g[g.vision_valid]
        summary.append({'split':split,'condition':condition,'method':method,'states':len(g),**{m:mean_ci(g,m) for m in ['joint','active','jerk_gain','seconds','scored_candidates','early_stop']},'risk':mean_ci(active,'violation'),'valid_images':len(valid),'seconds_p50':float(g.seconds.median()),'seconds_p95':float(g.seconds.quantile(.95)),'valid_seconds_p50':float(valid.seconds.median()) if len(valid) else None,'deadlines':{str(d):float((g.seconds>d).mean()) for d in acfg['deadline_seconds']}})
    for (split,condition),g in f.groupby(['split','condition']):
        base=g[g.method=='fixed']
        for method in acfg['methods'][1:]:
            x=g[g.method==method].merge(base,on='state_id',suffixes=('','_base'))
            for metric in ['joint','jerk_gain','seconds','scored_candidates']:x[metric+'_diff']=x[metric].astype(float)-x[metric+'_base'].astype(float)
            paired.append({'split':split,'condition':condition,'method':method,'baseline':'fixed',**{m:mean_ci(x,m+'_diff') for m in ['joint','jerk_gain','seconds','scored_candidates']}})
        indexed={m:g[g.method==m].set_index('state_id').sort_index() for m in ['fixed','allowance','compute','both']}
        x=indexed['fixed'][['initial_pose_family']].copy()
        for metric in ['joint','jerk_gain','seconds','scored_candidates']:
            vals={m:v[metric].astype(float) for m,v in indexed.items()}
            x[metric+'_allowance']=((vals['allowance']-vals['fixed'])+(vals['both']-vals['compute']))/2
            x[metric+'_compute']=((vals['compute']-vals['fixed'])+(vals['both']-vals['allowance']))/2
            x[metric+'_interaction']=vals['both']-vals['allowance']-vals['compute']+vals['fixed']
        factorial.append({'split':split,'condition':condition,**{m:mean_ci(x,m) for m in x.columns if m!='initial_pose_family'}})
    return {'summary':summary,'paired':paired,'factorial_effects':factorial}


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tune','evaluate'],required=True);p.add_argument('--partial',action='store_true');args=p.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text());acfg=json.loads(resolve('configs/followup_adaptation_factorial.json').read_text());rcfg=json.loads(resolve('configs/followup_response_models.json').read_text());base=resolve(cfg['output']);out=base/'adaptation_factorial';manifest=json.loads((out/'manifest.json').read_text());rule=manifest['base_verifier_rule']
    if args.phase=='tune':
        assert not args.partial,'Never lock incomplete calibration'
        f,audit=load_audit(out/'calibration_tuning',cfg,acfg,rule,rcfg,105,False);assert set(f.condition)=={'nominal'}
        result=tune(f,cfg,acfg,manifest['signature']);write_json(out/'calibration_tuning/raw_audit.json',audit);write_json(out/'calibrated_settings.json',result);f.to_csv(out/'calibration_tuning/evaluation.csv',index=False);print('Locked alpha',result['alpha'],'constant radius',result['constant_radius'],'feasible',result['alpha_feasible'],result['constant_feasible'],flush=True)
    else:
        frames=[];audits={}
        for split in ['development','calibration']:
            directory=out/split
            if args.partial and not list(directory.glob('*_f*_e*_t*.json')):continue
            f,a=load_audit(directory,cfg,acfg,rule,rcfg,105,args.partial);frames.append(f);audits[split]=a
        f=pd.concat(frames,ignore_index=True);result=summarize(f,acfg);result.update({'scope':acfg['scope'],'partial':args.partial,'locked':json.loads((out/'calibrated_settings.json').read_text()),'timing_scope':acfg['cost']});write_json(out/'analysis.json',result);write_json(out/'raw_audit.json',audits);f.to_csv(out/'evaluation.csv',index=False)
        for r in result['summary']:
            if r['split']=='development':print(r['condition'],r['method'],'U',round(r['joint']['mean'],3),'R',r['risk']['mean'],'scored',round(r['scored_candidates']['mean'],1),flush=True)
if __name__=='__main__':main()
