"""Independent dense-state and coverage audit for policy terminal diagnosis."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_policy_final import Coverage
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())


def energy(q):
    j=np.diff(q,n=3,axis=0)/.01**3
    return float((j*j).sum()*.01)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=BASE/('policy_terminal_smoke' if args.smoke else 'policy_terminal');m=read(out/'manifest.json');cfg=m['config'];tol=m['main']['tolerances'];done=read(out/'execution_complete.json')
    assert done['manifest_sha256']==sha256(out/'manifest.json') and done['smoke']==args.smoke
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    errors={};rows=[];states=[];t=cfg['correction_step'];n=cfg['max_actions'];c=cfg['chunk_actions']
    def check(name,a,b,atol=1e-9):
        err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));errors[name]=max(errors.get(name,0.),err);assert err<=atol,(name,err)
    seeds=cfg['development_seeds'][:1] if args.smoke else cfg['development_seeds']
    for seed in seeds:
        cv=Coverage(seed)
        with np.load(BASE/f'policy_refinement/development/{seed}_nominal.npz') as original:parent={k:original[k].copy() for k in ['physics','actions','coverage','initial_state']}
        for h in cfg['horizons']:
            path=out/f'{seed}_h{h:02d}.json';meta=read(path);assert meta['manifest_sha256']==sha256(out/'manifest.json');records=pd.read_csv(path.with_suffix('.csv'))
            assert len(records)==len(cfg['families'])*len(cfg['relative_radii'])
            with np.load(path.with_suffix('.npz')) as a:
                traces=a['physics'];actions=a['actions'];cover=a['coverage'];initial=a['initial'];scale=a['scale']
                assert traces.shape==(17,3000,10) and actions.shape==(17,300,2)
                check('nominal_full_physics',traces[0],parent['physics'],0);check('nominal_actions',actions[0],parent['actions'],0);check('nominal_coverage',cover[0],parent['coverage'],0);check('initial',initial,parent['initial_state'],0)
                j0=energy(np.vstack([initial[:2],traces[0,:,:2]]));jl0=energy(traces[0,t*10-30:(t+c)*10,:2])
                for r in records.itertuples():
                    i=r.index;trace=traces[i];ac=actions[i];basis=a['basis_'+r.method];assert basis.shape==(2*h,r.dimension)
                    check('unchanged_past_actions',ac[:t],actions[0,:t],0);check('unchanged_future_actions',ac[t+h:],actions[0,t+h:],0);check('prefix_physics',trace[:t*10],traces[0,:t*10],0)
                    assert np.max(np.abs(ac))<=1+1e-6
                    delta=((ac[t:t+h]-actions[0,t:t+h])/scale).reshape(-1);norm=float(np.linalg.norm(delta));check('norm',norm,r.norm)
                    assert norm<=r.radius*np.sqrt(2*h)+1e-6
                    check('family_span',delta,basis@(basis.T@delta),1e-8);check('endpoint_correction',np.r_[delta[:2],delta[-2:]],np.zeros(4),1e-8)
                    recov=np.array([cv(x) for x in trace[9::10]]);check('independent_polygon_coverage',recov,cover[i],1e-10)
                    d=trace-traces[0];pose=np.maximum(np.linalg.norm(d[:,2:4],axis=1)/tol['position'],np.abs(np.arctan2(np.sin(d[:,4]),np.cos(d[:,4])))/tol['rotation']);closs=np.maximum(0,cover[0]-recov)/tol['coverage_loss']
                    localE=float(max(pose[t*10:(t+c)*10].max(),closs[t:t+c].max()));fullE=float(max(pose.max(),closs.max()));assert fullE+1e-10>=localE
                    gain=(j0-energy(np.vstack([initial[:2],trace[:,:2]])))/j0 if j0>1e-8 else 0.
                    lgain=(jl0-energy(trace[t*10-30:(t+c)*10,:2]))/jl0 if jl0>1e-8 else 0.
                    j=(t+h)*10-1;q=float(np.linalg.norm(d[j,:2]));v=float(np.linalg.norm(d[j,5:7]));check('terminal_q_reproduction',q,r.terminal_q);check('terminal_v_reproduction',v,r.terminal_v)
                    if r.method.endswith('_terminal'):assert max(q,v)<=cfg['terminal_tolerance']
                    active=norm>cfg['active_threshold'];useful=active and localE<=1 and lgain>=cfg['jerk_gain_threshold']
                    crossings=np.r_[(np.flatnonzero(pose>1)+1)*.01,(np.flatnonzero(closs>1)+1)*.1]
                    rows.append({'seed':seed,'initial_pose_family':seed,'horizon':h,'method':r.method,'radius':r.radius,'dimension':r.dimension,'active':active,'norm':norm,'local_E':localE,'full_E':fullE,'local_safe':localE<=1,'full_safe':fullE<=1,'local_gain':lgain,'full_gain':gain,'local_useful':useful,'carryover_useful':useful and fullE<=1,'full_useful':active and fullE<=1 and gain>=cfg['jerk_gain_threshold'],'terminal_q':q,'terminal_v':v,'post_terminal_pusher_max':float(np.linalg.norm(d[(t+h)*10:,:2],axis=1).max()),'first_violation_seconds':float(crossings.min()) if len(crossings) else None})
    f=pd.DataFrame(rows)
    for (seed,h,method),g in f.groupby(['seed','horizon','method']):
        states.append({'seed':seed,'initial_pose_family':seed,'horizon':int(h),'method':method,'dimension':int(g.dimension.iloc[0]),'active_fraction':float(g.active.mean()),'short_oracle_U':bool(g.local_useful.any()),'carryover_oracle_U':bool(g.carryover_useful.any()),'full_oracle_U':bool(g.full_useful.any()),'long_safe_given_local_safe':float(g[g.active & g.local_safe].full_safe.mean()) if (g.active & g.local_safe).any() else None})
    s=pd.DataFrame(states);summary=[];paired=[]
    for (h,method),g in s.groupby(['horizon','method']):
        summary.append({'horizon':int(h),'method':method,'dimension':int(g.dimension.iloc[0]),'families':len(g),**{k:mean_ci(g.dropna(subset=[k]),k) for k in ['active_fraction','short_oracle_U','carryover_oracle_U','full_oracle_U','long_safe_given_local_safe']}})
    for h in cfg['horizons']:
        for name in ['quintic','spline']:
            a=s[(s.horizon==h)&(s.method==name+'_terminal')];b=s[(s.horizon==h)&(s.method==name)];z=a.merge(b,on=['seed','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
            for k in ['short_oracle_U','carryover_oracle_U','full_oracle_U']:z[k+'_difference']=z[k].astype(float)-z[k+'_base'].astype(float)
            paired.append({'horizon':h,'method':name+'_terminal','baseline':name,**{k:mean_ci(z,k+'_difference') for k in ['short_oracle_U','carryover_oracle_U','full_oracle_U']}})
    f.to_csv(out/'evaluation.csv',index=False);s.to_csv(out/'state_metrics.csv',index=False)
    write_json(out/'analysis.json',{'smoke':args.smoke,'summary':summary,'paired':paired,'scope':cfg['scope'],'preservation':cfg['preservation']})
    write_json(out/'raw_audit.json',{'smoke':args.smoke,'families':len(seeds),'seed_horizons':len(seeds)*len(cfg['horizons']),'candidate_branches':len(f),'max_error':errors,'manifest_sha256':sha256(out/'manifest.json'),'source_sha256':sha256(Path(__file__))})
    print('AUDITED',len(f),'policy terminal branches',flush=True)
if __name__=='__main__':main()
