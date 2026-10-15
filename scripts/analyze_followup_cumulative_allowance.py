"""Reconstruct cumulative choices and raw trajectory preservation before calibration."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from zj_wam.adapters import ExpertDataset
from zj_wam.io import resolve,write_json,sha256
from zj_wam.diagnostics import jerk_energy
from zj_wam.adaptation_factorial import pick
from zj_wam.cumulative_allowance import remaining_margin,deviation_score
from zj_wam.repeated_refinement import visual_history
from scripts.analyze_followup_response_selection import mean_ci


def load(directory,cfg,rcfg,base_rule,scale,partial=False):
    paths=sorted(directory.glob('*_f*_e*_t*.json'))
    if not partial:assert len(paths)==105,(directory,len(paths))
    frames=[];errors={};block_rows=[]
    def check(name,x,y,tol=1e-7):
        e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),e);assert e<=tol,(name,e)
    for path in paths:
        meta=json.loads(path.read_text());frame=pd.read_csv(path.with_suffix('.csv'));frames.append(frame)
        with np.load(path.with_suffix('.npz')) as a:
            ref=a['nominal_physics'];nom=a['nominal_actions'];eps=cfg['tolerances'];covref=a['nominal_coverage']
            for row in frame.itertuples(index=False):
                actual=a[row.variant+'_physics'];delivered=a[row.variant+'_actions'];cov=a[row.variant+'_coverage'];check('global_prefix',actual[:30],ref[:30],1e-9)
                for block,d in enumerate(meta['decisions'][row.variant]):
                    tag=f'{row.variant}_block{block}';offset=d['offset'];assert offset==block*5;index=30+offset*10-1
                    check('current_root_from_delivered_history',a[tag+'_proprio'][-1],actual[index],1e-8);check('reference_root_from_unmodified_history',a[tag+'_reference_proprio'][-1],ref[index],1e-8);check('current_prefix',a[tag+'_prefix'],actual[index-29:index+1],1e-8)
                    if row.tier=='vision':
                        obs,valid=visual_history(a[tag+'_images'],a[tag+'_proprio'],rcfg);reference,refvalid=visual_history(a[tag+'_reference_images'],a[tag+'_reference_proprio'],rcfg)
                    else:obs=a[tag+'_proprio'];reference=a[tag+'_reference_proprio'];valid=refvalid=True
                    check('observed_history',obs,a[tag+'_history']);check('reference_history',reference,a[tag+'_reference_history']);margin,e=remaining_margin(obs[-1],reference[-1],eps);check('observed_root_error',e,d['observed_reference_error']);check('remaining_margin',margin,d['remaining_margin'])
                    factor=row.multiplier*(margin if row.method=='remaining_margin' else 1.);check('radius',base_rule['relative_radius']*factor,d['effective_radius']);check('tau',base_rule['tau']*factor,d['effective_tau']);global_score=deviation_score(a[tag+'_predicted_poses'],a[tag+'_predicted_reference'],eps);local_score=deviation_score(a[tag+'_predicted_poses'],a[tag+'_predicted_poses'][:1],eps);check('global_prediction_scores',global_score,a[tag+'_reference_scores']);check('local_prediction_scores',local_score,a[tag+'_local_scores'])
                    scores=global_score if row.method=='reference_forecast' else local_score;use=valid and (refvalid if row.method in ['remaining_margin','reference_forecast'] else True);idx=pick(a[tag+'_costs'],a[tag+'_norms'],scores,use,d['effective_tau'],d['effective_radius']);assert idx==d['index'];candidate=a[tag+'_actions'][idx];check('selected_action_float32_delivery',candidate,delivered[offset:offset+5],1e-7);check('candidate_nominal_expert',a[tag+'_actions'][0],nom[offset:offset+5],1e-9);check('polynomial_endpoints',candidate[[0,-1]]-nom[offset:offset+5][[0,-1]],0.,1e-8)
                    current_error=deviation_score(actual[index,2:5][None,None],ref[index,2:5][None,None],eps)[0]
                    block_rows.append({'state_id':row.state_id,'initial_pose_family':row.initial_pose_family,'split':row.split,'variant':row.variant,'tier':row.tier,'method':row.method,'offset':offset,'active':d['active'],'true_current_error':current_error,'observed_current_error':e,'predicted_nominal_score':d['nominal_reference_score'],'fallback_after_true_violation':bool(not d['active'] and current_error>1)})
                norm=np.linalg.norm((delivered-nom)/scale);check('delivered_norm',norm,row.actual_norm)
                pose_delta=actual[30:,2:5]-ref[30:,2:5];pos=np.linalg.norm(pose_delta[:,:2],axis=-1);rot=np.abs(np.arctan2(np.sin(pose_delta[:,2]),np.cos(pose_delta[:,2])));loss=np.maximum(0,covref-cov);core=max(pos[:230].max()/eps['position'],rot[:230].max()/eps['rotation'],loss[:2].max()/eps['coverage_loss']);long=max(core,pos.max()/eps['position'],rot.max()/eps['rotation'],loss.max()/eps['coverage_loss']);check('core_E',core,row.object_error);check('nested_long_E',long,row.nested_long_error)
                for prefix,n,key in [('core',260,'jerk_gain'),('long',len(actual),'long_jerk_gain')]:
                    j=jerk_energy(actual[:n,:2],.01);j0=jerk_energy(ref[:n,:2],.01);gain=(j0-j)/j0 if j0>1e-8 else 0.;check(prefix+'_jerk',gain,getattr(row,key))
                for endpoint in [20,23,20+row.suffix_actions]:
                    index=30+endpoint*10-1;check('terminal_velocity',np.linalg.norm(actual[index,7:9]-ref[index,7:9]),getattr(row,f'velocity_delta_t{endpoint}'));check('terminal_angular_velocity',abs(actual[index,9]-ref[index,9]),getattr(row,f'angular_velocity_delta_t{endpoint}'))
    f=pd.concat(frames,ignore_index=True);f['joint']=f.active&(f.object_error<=1)&(f.jerk_gain>=.1);f['long_joint']=f.active&(f.nested_long_error<=1)&(f.long_jerk_gain>=.1);f['violation']=f.object_error>1;f['long_violation']=f.nested_long_error>1;f['core_pass_long_fail']=(f.object_error<=1)&f.long_violation
    return f,pd.DataFrame(block_rows),{'states':len(paths),'rows':len(f),'blocks':len(block_rows),'max_absolute_recalculation_error':errors,'scope':'Reconstruct observation tier, current/reference roots and prefix from delivered/unmodified traces, margin/radius/tau, predicted scores and selected actions; core+nestedlong E, global jerk and terminal velocity from raw100Hzstates. Coverage endpoints reused.'}


def lock_rules(f,ccfg,cfg,signature):
    rules={};curves=[]
    for tier in ccfg['tiers']:
        rules[tier]={}
        for method in ccfg['methods'][1:]:
            choices=[]
            for multiplier in ccfg['multiplier_grid']:
                x=f[(f.tier==tier)&(f.method==method)&(f.multiplier==multiplier)];assert len(x)==105;active=x[x.active];contact=active[active.nominal_contact];risk=float(active.violation.mean()) if len(active) else None;cr=float(contact.violation.mean()) if len(contact) else None;valid=risk is not None and cr is not None and max(risk,cr)<=cfg['target_violation_rate'] and x.active.mean()>=cfg['minimum_active_coverage']
                choices.append({'tier':tier,'method':method,'multiplier':multiplier,'feasible':bool(valid),'joint':float(x.joint.mean()),'long_joint':float(x.long_joint.mean()),'active':float(x.active.mean()),'risk':risk,'contact_risk':cr})
            possible=[x for x in choices if x['feasible']];best=max(possible,key=lambda x:(x['joint'],x['long_joint'],-x['multiplier'])) if possible else min(choices,key=lambda x:(max(x['risk'] if x['risk'] is not None else 1.,x['contact_risk'] if x['contact_risk'] is not None else 1.),-x['joint'],x['multiplier']))
            rules[tier][method]=best;curves.extend(choices)
    return {'signature':signature,'rules':rules,'calibration_curves':curves,'scope':ccfg['calibration'],'note':'Infeasible chosen settings remain diagnostic, not risk-qualified deployment rules; calibration uncertainty and finaltest remain.'}


def summarize(f,b):
    result=[];paired=[];drift=[]
    for (split,tier,method),g in f.groupby(['split','tier','method']):
        active=g[g.active];result.append({'split':split,'tier':tier,'method':method,'states':len(g),**{m:mean_ci(g,m) for m in ['joint','long_joint','active','jerk_gain','long_jerk_gain','core_pass_long_fail','refinement_seconds','velocity_delta_t20','velocity_delta_t23','modified_success']},'risk':mean_ci(active,'violation'),'long_risk':mean_ci(active,'long_violation')})
        x=g[g.active].copy()
        if len(x)>2 and x.velocity_delta_t20.nunique()>1 and x.nested_long_error.nunique()>1:
            r=spearmanr(x.velocity_delta_t20,x.nested_long_error).statistic;drift.append({'split':split,'tier':tier,'method':method,'states':len(x),'terminal_speed_vs_long_E_spearman':float(r),'interpretation':'Descriptive association, not causal velocity effect; position/angle andcontactconfounding remain.'})
    for (split,tier),g in f.groupby(['split','tier']):
        baseline=g[g.method=='local']
        for method in ['constant_shrink','remaining_margin','reference_forecast']:
            x=g[g.method==method].merge(baseline,on='state_id',suffixes=('','_base'))
            for m in ['joint','long_joint','jerk_gain','nested_long_error']:x[m+'_difference']=x[m].astype(float)-x[m+'_base'].astype(float)
            paired.append({'split':split,'tier':tier,'method':method,'baseline':'local',**{m:mean_ci(x,m+'_difference') for m in ['joint','long_joint','jerk_gain','nested_long_error']}})
    fallback=[]
    for keys,g in b.groupby(['split','tier','method']):fallback.append({**dict(zip(['split','tier','method'],keys)),'blocks':len(g),'fallback_after_true_violation':int(g.fallback_after_true_violation.sum()),'nominal_predicted_violation_blocks':int((g.predicted_nominal_score>1).sum())})
    return {'summary':result,'paired':paired,'drift_association':drift,'fallback_diagnostics':fallback}


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tune','evaluate'],required=True);p.add_argument('--partial',action='store_true');args=p.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text());rcfg=json.loads(resolve('configs/followup_response_models.json').read_text());ccfg=json.loads(resolve('configs/followup_cumulative_allowance.json').read_text());out=resolve(cfg['output'])/'cumulative_allowance';manifest=json.loads((out/'manifest.json').read_text());data=ExpertDataset(cfg['dataset']);frames=[];blocks=[];audits={}
    if args.phase=='tune':
        f,b,a=load(out/'calibration_tuning',cfg,rcfg,manifest['base_rule'],data.scaler.scale_,args.partial);write_json(out/'calibration_tuning/raw_audit.json',a);f.to_csv(out/'calibration_tuning/evaluation.csv',index=False);b.to_csv(out/'calibration_tuning/block_evaluation.csv',index=False)
        if not args.partial:
            lock=lock_rules(f,ccfg,cfg,manifest['signature']);write_json(out/'calibrated_settings.json',lock)
            for tier,v in lock['rules'].items():
                for method,r in v.items():print(tier,method,r,flush=True)
        else:print('Partial audit only, calibration NOT locked',a,flush=True)
    else:
        for split in ['development','calibration']:
            if args.partial and not list((out/('calibration_tuning' if split=='calibration' else split)).glob('*_f*_e*_t*.json')):continue
            source=out/split if split=='development' else out/'calibration_tuning'
            f,b,a=load(source,cfg,rcfg,manifest['base_rule'],data.scaler.scale_,args.partial)
            if split=='calibration':
                locked=json.loads((out/'calibrated_settings.json').read_text())['rules']
                mask=np.array([r.method=='local' or r.multiplier==locked[r.tier][r.method]['multiplier'] for r in f.itertuples()]);f=f[mask].copy();selected=set(f.variant);b=b[b.variant.isin(selected)].copy();a['selected_rows']=len(f);a['calibration_reuse']='Same deterministic configurations already physically executed during tuning; only locked multipliers retained. Calibration remains fitting data, not held-out evidence.'
            frames.append(f);blocks.append(b);audits[split]=a
        f=pd.concat(frames,ignore_index=True);b=pd.concat(blocks,ignore_index=True);analysis=summarize(f,b);analysis.update({'scope':ccfg['scope'],'partial':args.partial,'locked':json.loads((out/'calibrated_settings.json').read_text())});write_json(out/'analysis.json',analysis);write_json(out/'raw_audit.json',audits);f.to_csv(out/'evaluation.csv',index=False);b.to_csv(out/'block_evaluation.csv',index=False)
        for r in analysis['summary']:
            if r['split']=='development':print(r['tier'],r['method'],'U',r['joint']['mean'],'R',r['risk']['mean'],'longU',r['long_joint']['mean'],flush=True)
if __name__=='__main__':main()
