"""Audit dense metrics, exact repeats and fresh branches; summarize numerical sensitivity."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from zj_wam.io import resolve,sha256,write_json
from zj_wam.policy_refinement import make_scene,full_state
from scripts.analyze_followup_policy_final import Coverage
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main(smoke):
    out=BASE/('pusht_numerical_smoke' if smoke else 'pusht_numerical');lock=read(out/'protocol.json');done=read(out/'execution_complete.json');assert lock['smoke']==done['smoke']==smoke and done['protocol_sha256']==sha256(out/'protocol.json');cfg=lock['config']
    for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
    rows=[];state_rows=[];inputs={};fresh=0;max_error=0.
    for path in lock['sources']:
        source=resolve(path);meta=read(source);a=dict(np.load(source.with_suffix('.npz')));parent=dict(np.load(resolve(meta['source']).with_suffix('.npz')));p=out/(source.stem+'.npz');m=read(p.with_suffix('.json'));assert m['data_sha256']==sha256(p) and m['protocol_sha256']==sha256(out/'protocol.json');z=dict(np.load(p));n=len(a['actions']);assert m['candidates']==n;covfun=Coverage(meta['seed']);per_rate={}
        for variable in ['physics','contacts','coverage']:
            np.testing.assert_array_equal(z[variable+'_100_r0'],z[variable+'_100_r1'])
            original=a[variable][:,30:] if variable=='physics' else a[variable]
            np.testing.assert_array_equal(z[variable+'_100_r0'],original)
        for rate in cfg['rates']:
            f=rate//100;x=z[f'physics_{rate}_r0'];contact=z[f'contacts_{rate}_r0'];cov=z[f'coverage_{rate}_r0'];assert x.shape==(n,80*f,10) and contact.shape==(n,80*f) and cov.shape==(n,2);assert np.isfinite(x).all() and np.all(contact>=0)
            cv=np.array([[covfun(t) for t in q[[50*f-1,80*f-1]]] for q in x]);np.testing.assert_allclose(cv,cov,atol=1e-10,rtol=0)
            delta=x[:,:,2:5]-x[:1,:,2:5];position=np.sqrt(np.sum(delta[:,:,:2]**2,axis=2));rotation=np.abs(np.arctan2(np.sin(delta[:,:,2]),np.cos(delta[:,:,2])))*180/np.pi
            error=np.maximum(position/cfg['position_tolerance_px'],rotation/cfg['rotation_tolerance_deg']);cover_loss=np.maximum(0,cv[:1]-cv).max(axis=1)/cfg['coverage_loss_tolerance'];E=np.maximum(error.max(axis=1),cover_loss);Ec=np.maximum(error[:,f-1::f].max(axis=1),cover_loss)
            common=x[:,f-1::f];points=np.concatenate([np.broadcast_to(a['input_prefix'][:,:2],(n,30,2)),common[:,:,:2]],axis=1);third=points[:,3:]-3*points[:,2:-1]+3*points[:,1:-2]-points[:,:-3];cost=(third**2).sum(axis=(1,2))/.01**5;gain=(cost[0]-cost)/cost[0] if cost[0]>1e-8 else np.zeros(n);preserved=E<=1;useful=preserved&(gain>=cfg['minimum_jerk_gain'])&(a['norms']>1e-6);contact_time=np.array([(np.flatnonzero(c>0)[0]+1)/rate if (c>0).any() else np.nan for c in contact]);per_rate[rate]={'E':E,'Ec':Ec,'gain':gain,'cost':cost,'safe':preserved,'useful':useful,'contact_time':contact_time,'points':common}
            for i in range(n):rows.append({'state_id':meta['state_id'],'initial_pose_family':meta['seed'],'phase':meta['phase'],'candidate':i,'rate':rate,'E_dense':E[i],'E_common100':Ec[i],'jerk_common100':cost[i],'gain_common100':gain[i],'preserved':bool(preserved[i]),'useful':bool(useful[i]),'contact_first_seconds':contact_time[i]})
            for i in sorted({0,n-1}):
                # Fresh replay via direct space hook, not the collector rollout helper.
                env=make_scene(meta['seed'])
                for command in parent['actions'][:meta['step']]:env.step(command)
                np.testing.assert_array_equal(full_state(env),parent['states'][meta['step']]);env.dt=1./rate;assert env.control_hz==10 and env.k_p==100 and env.k_v==20;trace=[];step=env.space.step
                def record(dt):step(dt);trace.append(full_state(env))
                env.space.step=record
                for command in np.concatenate([a['actions'][i],a['input_suffix']]):env.step(command)
                env.space.step=step;env.close();np.testing.assert_array_equal(np.asarray(trace),x[i]);fresh+=1
        baseline=per_rate[100]
        for rate,r in per_rate.items():
            corr=spearmanr(baseline['gain'][1:],r['gain'][1:]).statistic if np.ptp(baseline['gain'][1:])>0 and np.ptp(r['gain'][1:])>0 else np.nan
            both=np.isfinite(baseline['contact_time'])&np.isfinite(r['contact_time']);pdiff=r['points'][:,:,2:4]-baseline['points'][:,:,2:4];theta=r['points'][:,:,4]-baseline['points'][:,:,4];active=np.arange(n)>0
            state_rows.append({'state_id':meta['state_id'],'initial_pose_family':meta['seed'],'phase':meta['phase'],'rate':rate,'active_candidates':n-1,'preservation_flip_fraction':float(np.mean((r['safe']!=baseline['safe'])[active])),'useful_flip_fraction':float(np.mean((r['useful']!=baseline['useful'])[active])),'dense_vs_common_preservation_flip_fraction':float(np.mean(((r['E']<=1)!=(r['Ec']<=1))[active])),'has_useful_candidate':bool(r['useful'].any()),'preserved_candidate_fraction':float(r['safe'][active].mean()),'useful_candidate_fraction':float(r['useful'][active].mean()),'jerk_gain_rank_spearman':float(corr),'max_abs_gain_change':float(abs(r['gain']-baseline['gain']).max()),'max_paired_E_change':float(abs(r['E']-baseline['E']).max()),'nominal_position_change_px':float(np.linalg.norm(pdiff[0],axis=1).max()),'all_branch_position_change_px':float(np.linalg.norm(pdiff,axis=2).max()),'nominal_rotation_change_deg':float(abs(np.arctan2(np.sin(theta[0]),np.cos(theta[0]))).max()*180/np.pi),'contact_existence_flip_fraction':float(np.mean(np.isfinite(r['contact_time'])!=np.isfinite(baseline['contact_time']))),'max_contact_time_change_seconds':float(abs(r['contact_time'][both]-baseline['contact_time'][both]).max()) if both.any() else None})
        for q in [p,p.with_suffix('.json')]:inputs[str(q.relative_to(resolve('.')))]=sha256(q)
        print('AUDIT NUMERICAL',meta['state_id'],n,flush=True)
    f=pd.DataFrame(rows);s=pd.DataFrame(state_rows);assert len(lock['sources'])==done['states'] and len(f)==done['candidates']*3;f.to_csv(out/'candidate_metrics.csv',index=False);s.to_csv(out/'state_metrics.csv',index=False);summary=[]
    metrics=['preservation_flip_fraction','useful_flip_fraction','dense_vs_common_preservation_flip_fraction','preserved_candidate_fraction','useful_candidate_fraction','jerk_gain_rank_spearman','max_abs_gain_change','max_paired_E_change','nominal_position_change_px','all_branch_position_change_px','nominal_rotation_change_deg','contact_existence_flip_fraction','max_contact_time_change_seconds']
    for (rate,phase),g in s.groupby(['rate','phase']):
        summary.append({'rate':int(rate),'phase':phase,'states':len(g),'families':int(g.initial_pose_family.nunique()),'states_with_useful_candidate':int(g.has_useful_candidate.sum()),**{k:{'mean':None if g[k].isna().all() else float(g[k].mean()),'max':None if g[k].isna().all() else float(g[k].max())} for k in metrics}})
    result={'smoke':smoke,'states':done['states'],'families':int(s.initial_pose_family.nunique()),'candidates':done['candidates'],'fresh_collection_branches':done['fresh_physical_branches'],'fresh_audit_branches':fresh,'summary':summary,'scope':cfg['scope'],'limitations':'Small prespecified phase-stratified development diagnostic. All55candidates/state share one root and are not independent samples. Original100Hzprefix retained for identical root; rates alter PD integration and contact discretization. Relative-action absolute targets can diverge after first action when robot state differs. Higher rate is not automatically ground truth. No change to old thresholds/test results; no training/new verifier.'};write_json(out/'analysis.json',result)
    write_json(out/'raw_audit.json',{'states':done['states'],'candidates':done['candidates'],'exact_100Hz_repeats_and_parent':True,'fresh_physical_branches':fresh,'all_dense_errors_common_jerk_and_independent_polygon_coverage':True,'inputs':inputs,'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'source_sha256':sha256(Path(__file__)),'helpers':{p:sha256(resolve(p)) for p in ['scripts/analyze_followup_policy_final.py','zj_wam/policy_refinement.py']}});print('AUDITED NUMERICAL',done['states'],done['candidates'],fresh,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
