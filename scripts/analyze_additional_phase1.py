"""Separate calibration and descriptive audits for proposal-driven diagnostics."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zj_wam.io import resolve,sha256,write_json
from zj_wam.metrics import preservation_mask,cluster_bootstrap


def prepare(frame,cfg):
    f=frame.copy();f['preserved_final']=preservation_mask(f,cfg['tolerances'])
    pos='position_with_suffix_max' if 'position_with_suffix_max' in f else 'position_path_max'
    rot='rotation_with_suffix_max' if 'rotation_with_suffix_max' in f else 'rotation_path_max'
    f['preserved_path']=f.preserved_final&(f[pos]<=cfg['tolerances']['position'])&(f[rot]<=cfg['tolerances']['rotation'])
    if 'executed_jerk' in f:
        valid=f.nominal_executed_jerk>cfg['minimum_nominal_jerk']
        f['jerk_reduction']=np.where(valid,1-f.executed_jerk/f.nominal_executed_jerk,0.)
        f['physics_jerk_reduction']=np.where(f.nominal_physics_jerk>cfg['minimum_nominal_jerk'],1-f.physics_jerk/f.nominal_physics_jerk,0.)
        f['meaningful_jerk']=valid&(f.jerk_reduction>=cfg['meaningful_jerk_reduction'])
        f['joint_final']=f.preserved_final&f.meaningful_jerk
        f['joint_path']=f.preserved_path&f.meaningful_jerk
        f['roughness_reduction']=np.where(f.nominal_action_roughness>1e-12,1-f.action_roughness/f.nominal_action_roughness,0.)
    return f


def nominal_row(g):
    r=g.iloc[0].copy()
    for key in ['position_error','rotation_error','position_path_mean','position_path_max','rotation_path_mean','rotation_path_max','object_error','task_score_difference','position_with_suffix_max','rotation_with_suffix_max','applied_radius','actual_norm','predicted_change','jerk_reduction','physics_jerk_reduction','roughness_reduction','contact_changed_steps','relative_radius']:
        r[key]=0.
    for key in ['executed_jerk','physics_jerk','target_jerk','action_roughness']:r[key]=r['nominal_'+key]
    r['preserved_final']=r['preserved_path']=True
    r['meaningful_jerk']=r['joint_final']=r['joint_path']=False
    r['direction']='nominal';r['trajectory_index']=0;r['fallback']=True
    return r


def select_model(g,rule):
    eligible=g[(g.relative_radius<=rule['relative_radius'])&(g.predicted_change<=rule['tau'])&(g.roughness_reduction>1e-12)]
    if eligible.empty:return nominal_row(g)
    selected=eligible.sort_values(['action_roughness','predicted_change','applied_radius'],kind='stable').iloc[0].copy()
    selected['fallback']=False
    return selected


def fit(out,cfg,manifest):
    files=list((out/'oracle').glob('calibration_*.csv'))
    expected=sum(s['split']=='calibration' for s in manifest['oracle_states'])*len(cfg['oracle_horizons'])
    assert len(files)==expected,(len(files),expected)
    assert not list((out/'oracle').glob('test_*.json')),'Calibration must be frozen before new oracle test outcomes'
    frame=prepare(pd.concat([pd.read_csv(p) for p in files],ignore_index=True),cfg)
    rules={}
    for h,g in frame.groupby('horizon'):
        groups=[v for _,v in g.groupby('state_id')]
        best={'relative_radius':0.,'tau':0.,'objective':0.,'active_states':0,'observed_path_violation':None,'contact_violation':None}
        for radius in cfg['oracle_radii']:
            for tau in np.unique(np.quantile(g.predicted_change,np.linspace(0,1,41))):
                rule={'relative_radius':radius,'tau':float(tau)}
                selected=pd.DataFrame([select_model(v,rule) for v in groups])
                active=selected[~selected.fallback]
                if len(active)<max(3,len(groups)//2):continue
                failure=float((~active.preserved_path.astype(bool)).mean())
                contact=active[active.interaction_phase=='contact']
                contact_failure=float((~contact.preserved_path.astype(bool)).mean()) if len(contact) else 0.
                score=float(selected.roughness_reduction.mean())
                if failure<=cfg['calibration_violation_limit'] and contact_failure<=cfg['calibration_violation_limit'] and score>best['objective']:
                    best={**rule,'objective':score,'active_states':len(active),'observed_path_violation':failure,'contact_violation':contact_failure}
        rules[str(h)]=best
    artifact={'source_split':'calibration','manifest_sha256':sha256(out/'manifest.json'),'rules':rules,
              'selection':'minimum raw action third-difference energy among finite-prediction/radius accepted candidates; nominal fallback; actual jerk is never used at inference',
              'calibration_preservation':'final object tolerances plus time-aligned path position/rotation including three unchanged suffix actions',
              'gamma':cfg['meaningful_jerk_reduction'],'finite_sample_note':'12 independent calibration families; empirical rule, not a safety guarantee'}
    write_json(out/'oracle_calibration.json',artifact)
    (out/'oracle_calibration_frozen.sha256').write_text(sha256(out/'oracle_calibration.json')+'\n')
    print(json.dumps(artifact,indent=2))


def ci(values,ids,rng,n):return cluster_bootstrap(np.asarray(values,float),ids,rng,n)


def analyze(out,cfg,manifest):
    rng=np.random.default_rng(cfg['seed']+222);n=cfg['bootstrap_samples'];plots=out/'plots';plots.mkdir(exist_ok=True)
    controls=prepare(pd.concat([pd.read_csv(p) for p in sorted((out/'controls').glob('*.csv'))],ignore_index=True),cfg)
    assert controls.state_id.nunique()==len(manifest['controls_states'])
    controls['method']=controls.direction.str.replace(r'_\d+$','',regex=True)
    controls.to_csv(out/'controls_candidates.csv',index=False)
    controls_summary=[];paired=[];associations=[]
    for phase in ['all','contact','near_contact','free_motion']:
        f=controls if phase=='all' else controls[controls.interaction_phase==phase]
        for radius,g in f.groupby('relative_radius'):
            for method,v in g.groupby('method'):
                per=v.groupby(['episode','state_id']).agg(object_error=('object_error','mean'),position_error=('position_error','mean'),rotation_error=('rotation_error','mean'),pusher=('pusher_path_mean','mean'),final=('preserved_final','mean'),path=('preserved_path','mean'),centroid=('frequency_centroid','mean')).reset_index()
                controls_summary.append({'phase':phase,'relative_radius':float(radius),'method':method,'states':len(per),
                    **{key:float(per[key].mean()) for key in ['object_error','position_error','rotation_error','pusher','final','path','centroid']}})
            piv=g.groupby(['episode','state_id','sign','method']).object_error.mean().unstack('method')
            for other in ['frequency','random','coordinate','high']:
                diff=(piv.low-piv[other]).groupby(level=['episode','state_id']).mean()
                paired.append({'phase':phase,'relative_radius':float(radius),'comparison':'low-'+other,'mean_difference':float(diff.mean()),'ci':ci(diff.values,diff.index.get_level_values('episode'),rng,n)})
            for response in ['object_error','pusher_path_mean']:
                rho=None if g[response].nunique()<2 else float(spearmanr(g.predicted_change,g[response]).statistic)
                associations.append({'phase':phase,'relative_radius':float(radius),'response':response,'spearman':rho})
    write_json(out/'controls_statistics.json',{'descriptive':controls_summary,'paired':paired,'representation_associations':associations})
    c=pd.DataFrame(controls_summary);x=c[(c.phase=='contact')&(c.relative_radius==.1)]
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    for ax,key,title in zip(axs,['position_error','pusher','centroid'],['Object displacement (px)','Pusher mean displacement (px)','DCT frequency centroid']):
        ax.bar(x.method,x[key]);ax.set_title(title);ax.tick_params(axis='x',rotation=35)
    fig.suptitle('Matched action norms, contact states, relative radius 0.10');fig.tight_layout();fig.savefig(plots/'01_frequency_controller_controls.png',dpi=150);plt.close(fig)
    oracle=prepare(pd.concat([pd.read_csv(p) for p in sorted((out/'oracle').glob('*.csv'))],ignore_index=True),cfg)
    expected=len(manifest['oracle_states'])*len(cfg['oracle_horizons'])*manifest['polynomial_candidate_budget_per_state_horizon']
    assert len(oracle)==expected,(len(oracle),expected)
    oracle.to_csv(out/'oracle_candidates.csv',index=False)
    calibration=json.loads((out/'oracle_calibration.json').read_text());chosen=[]
    for (sid,h),g in oracle[oracle.split=='test'].groupby(['state_id','horizon']):
        for method in ['nominal','unconstrained_command','oracle_final','oracle_path','model_selected']:
            if method=='nominal':row=nominal_row(g)
            elif method=='model_selected':row=select_model(g,calibration['rules'][str(h)])
            else:
                pool=g if method=='unconstrained_command' else g[g.preserved_final if method=='oracle_final' else g.preserved_path]
                objective='action_roughness' if method=='unconstrained_command' else 'executed_jerk'
                if pool.empty or pool[objective].min()>=g['nominal_'+objective].iloc[0]:row=nominal_row(g)
                else:row=pool.sort_values(objective,kind='stable').iloc[0].copy();row['fallback']=False
            row['selection_method']=method;chosen.append(row)
    selected=pd.DataFrame(chosen);selected.to_csv(out/'oracle_selected.csv',index=False)
    summary=[]
    for phase in ['all','contact','near_contact','free_motion']:
        f=selected if phase=='all' else selected[selected.interaction_phase==phase]
        for (h,method),g in f.groupby(['horizon','selection_method']):
            rec={'phase':phase,'horizon':int(h),'method':method,'states':len(g),'families':g.episode.nunique()}
            for key in ['preserved_final','preserved_path','joint_final','joint_path','jerk_reduction','physics_jerk_reduction','fallback','applied_radius']:
                rec[key]=float(g[key].mean());rec[key+'_ci']=ci(g[key],g.episode,rng,n)
            rec['nominal_jerk_median']=float(g.nominal_executed_jerk.median())
            summary.append(rec)
    write_json(out/'oracle_statistics.json',{'summary':summary,'calibration':calibration})
    fig,axs=plt.subplots(1,3,figsize=(12,3.5));v=pd.DataFrame(summary)
    for method in ['unconstrained_command','model_selected','oracle_final','oracle_path']:
        g=v[(v.phase=='all')&(v.method==method)].sort_values('horizon')
        for ax,key in zip(axs,['preserved_path','joint_path','jerk_reduction']):ax.plot(g.horizon,g[key],marker='o',label=method)
    for ax,title in zip(axs,['Path preservation','Path preserved and jerk reduced >=10%','Mean executed jerk reduction']):ax.set(xlabel='Action horizon',title=title)
    axs[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(plots/'02_oracle_gap.png',dpi=150);plt.close(fig)
    # All original test interventions: stricter path criteria and nominal quality.
    old=pd.read_csv(resolve('artifacts/phase1/heldout_candidates.csv'))
    old=prepare(old,cfg);path_rows=[]
    for (phase,radius,group,success),g in old.groupby(['interaction_phase','relative_radius','group','nominal_task_success']):
        path_rows.append({'phase':phase,'relative_radius':float(radius),'method':group,'nominal_task_success':bool(success),'states':g.state_id.nunique(),'final_rate':float(g.preserved_final.mean()),'path_rate':float(g.preserved_path.mean()),'new_contact_rate':float(g.candidate_contact.mean()),'position_path_max_mean':float(g.position_path_max.mean())})
    pd.DataFrame(path_rows).to_csv(out/'original_path_and_nominal_quality.csv',index=False)
    # Primary oracle cost/gradient/prediction diagnostics, not a policy benchmark.
    checks=[json.loads(p.read_text()) for p in sorted((out/'oracle').glob('*.json'))]
    cost=pd.DataFrame(checks);cost.to_csv(out/'horizon_checks.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(8,3.5))
    for ax,key,label in zip(axs,['nominal_future_latent_prediction_error','jacobian_seconds'],['Nominal latent prediction error','Action Jacobian time (s)']):
        grouped=cost.groupby('horizon')[key];ax.plot(grouped.mean().index,grouped.mean().values,'o-');ax.set(xlabel='Action horizon',ylabel=label)
    fig.tight_layout();fig.savefig(plots/'03_horizon_diagnostics.png',dpi=150);plt.close(fig)
    print(json.dumps({'controls_states':controls.state_id.nunique(),'controls_executions':len(controls),'oracle_states':oracle.state_id.nunique(),'oracle_executions':len(oracle),'primary_controls':[p for p in paired if p['phase']=='contact' and p['relative_radius']==.1],'oracle_test':[s for s in summary if s['phase']=='all' and s['method'] in ['oracle_path','model_selected']]},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit',action='store_true');args=parser.parse_args()
    cfg=json.loads(resolve('configs/phase1_additional.json').read_text());out=resolve(cfg['output']);manifest=json.loads((out/'manifest.json').read_text())
    if args.fit:fit(out,cfg,manifest)
    else:analyze(out,cfg,manifest)

if __name__=='__main__':main()
