"""Independent sorted-table selection and raw physical checks for A-D diagnostic."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.run_followup_dinowm_selection import make_frame
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def main(smoke):
    out=BASE/('oracle_factorial_smoke' if smoke else 'oracle_factorial');protocol=read(out/'protocol.json');cfg=protocol['parent_config'];analysis=read(out/'analysis.json');done=read(out/'execution_complete.json');assert done['protocol_sha256']==sha256(out/'protocol.json') and done['analysis_sha256']==sha256(out/'analysis.json') and done['selected_sha256']==sha256(out/'selected.csv');rules={(r['environment'],r['method']):r['rule'] for r in read(BASE/'dinowm_selection/rules.json')['rules']};saved=pd.read_csv(out/'selected.csv');parent=pd.read_csv(BASE/'dinowm_selection/selected.csv');rows=[];errors={};candidates=0
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    def check(name,x,y,tol=1e-8):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);assert x.shape==y.shape;np.testing.assert_allclose(x,y,atol=tol,rtol=0,equal_nan=True,err_msg=name);e=abs(x-y);errors[name]=max(errors.get(name,0),float(np.nanmax(e)) if e.size and np.isfinite(e).any() else 0.)
    for env,split,name in protocol['sources']:
        src=resolve(name);g,_=make_frame(env,split,src,BASE/'dinowm_interventions',cfg);a=dict(np.load(src.with_suffix('.npz')));ids=g.candidate.to_numpy(int);candidates+=len(g)
        if env=='pusht':
            trace=a['physics'];d=trace[:,30:,2:5]-trace[:1,30:,2:5];pos=np.sqrt(np.sum(d[:,:,:2]**2,axis=2)).max(1)/2;ang=np.abs((d[:,:,2]+np.pi)%(2*np.pi)-np.pi).max(1)/np.deg2rad(2);E=np.maximum.reduce([pos,ang,np.maximum(0,a['coverage'][:1]-a['coverage']).max(1)/.02]);points=trace[:,:,:2];dt=.01
        else:
            trace=a['candidate_object_pose'];pos=np.linalg.norm(trace[:,:,:3]-trace[:1,:,:3],axis=2).max(1)/.002;q=trace[:,:,3:];q=q/np.linalg.norm(q,axis=2,keepdims=True);dots=np.clip(abs(np.einsum('nti,ti->nt',q,q[0])),0,1);ang=(2*np.arctan2(np.sqrt(np.maximum(0,1-dots*dots)),dots)).max(1)/np.deg2rad(2);E=np.maximum(pos,ang);points=np.concatenate([np.broadcast_to(a['prefix_eef_position'],(len(trace),*a['prefix_eef_position'].shape)),a['candidate_eef_position']],axis=1);dt=.002
        third=points[:,3:]-3*points[:,2:-1]+3*points[:,1:-2]-points[:,:-3];cost=np.sum(third**2,axis=(1,2))/dt**5;gain=(cost[0]-cost)/cost[0] if cost[0]>1e-8 else np.zeros(len(cost));check('independent_object_error',E[ids],g.dense_full,2e-6);check('independent_jerk_gain',gain[ids],g.jerk_gain,2e-8)
        for method in cfg['methods']:
            rule=rules[(env,method)];active=g[g.norm>cfg['minimum_norm']];good=active[(active.dense_full<=1)&(active.jerk_gain>=cfg['meaningful_gain'])]
            for cell in ['A','B','C','D']:
                approved=active[active.dense_full<=1] if cell in ['C','D'] else active[active[method]<=rule['tau']] if rule['enabled'] else active.iloc[:0]
                if cell in ['B','D']:eligible=approved[approved.jerk_gain>0].sort_values(['jerk_gain','norm','candidate'],ascending=[False,True,True],kind='stable')
                else:eligible=approved[approved.predicted_gain>cfg['minimum_predicted_gain']].sort_values(['predicted_cost','norm','candidate'],kind='stable')
                r=eligible.iloc[0] if len(eligible) else None;row={'environment':env,'split':split,'method':method,'cell':cell,'state_id':g.state_id.iloc[0],'initial_pose_family':int(g.initial_pose_family.iloc[0]),'phase':g.phase.iloc[0],'candidate':int(r.candidate) if r is not None else 0,'active':r is not None,'violation':bool(r is not None and r.dense_full>1),'joint':bool(r is not None and r.dense_full<=1 and r.jerk_gain>=cfg['meaningful_gain']),'jerk_gain':float(r.jerk_gain) if r is not None else 0.,'object_error':float(r.dense_full) if r is not None else 0.,'useful_candidate_exists':bool(len(good)),'approved_useful_exists':bool(len(approved[(approved.dense_full<=1)&(approved.jerk_gain>=cfg['meaningful_gain'])])),'approved_count':len(approved)};rows.append(row)
                if cell in ['C','D']:assert not row['violation']
                if cell=='D':assert row['joint']==row['useful_candidate_exists']
                if cell=='A':assert int(parent[(parent.environment==env)&(parent.state_id==row['state_id'])&(parent.method==method)].candidate.iloc[0])==row['candidate']
        print('AUDIT FACTORIAL',env,src.stem,flush=True)
    f=pd.DataFrame(rows);keys=['environment','split','method','cell','state_id'];f=f.sort_values(keys).reset_index(drop=True);saved=saved.sort_values(keys).reset_index(drop=True);assert len(f)==len(saved)==analysis['selection_rows'];assert candidates==analysis['candidate_rows']
    for col in f:
        if col in keys+['phase']:np.testing.assert_array_equal(f[col],saved[col])
        else:check('selected_'+col,f[col],saved[col])
    for r in analysis['summary']:
        g=f[(f.environment==r['environment'])&(f.split==r['split'])&(f.method==r['method'])&(f.cell==r['cell'])];assert len(g)==r['states']
        for metric in ['active','joint','jerk_gain','useful_candidate_exists','approved_useful_exists','selected_risk']:
            v=g[g.active] if metric=='selected_risk' else g;col='violation' if metric=='selected_risk' else metric
            if not len(v):assert r[metric]['mean'] is None;continue
            avg,ci=cluster(v[col].to_numpy(),v.initial_pose_family.to_numpy());check('summary_mean',avg,r[metric]['mean']);check('summary_ci',ci,r[metric]['ci95'])
    for r in analysis['paired']:
        g=f[(f.environment==r['environment'])&(f.split==r['split'])&(f.method==r['method'])];x=g[g.cell==r['later']].set_index('state_id');y=g[g.cell==r['earlier']].set_index('state_id').loc[x.index];v=x[r['metric']].astype(float)-y[r['metric']].astype(float);avg,ci=cluster(v.to_numpy(),x.initial_pose_family.to_numpy());check('paired_mean',avg,r['mean']);check('paired_ci',ci,r['ci95'])
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'raw_audit.json',{'smoke':smoke,'states':analysis['states'],'candidates':candidates,'selections':len(f),'max_errors':errors,'source_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'helpers':{p:sha256(resolve(p)) for p in ['scripts/run_followup_dinowm_selection.py','scripts/audit_retrained_dinowm_readouts.py']},'scope':'All raw dense object errors and executed jerk recomputed; scores/norms/candidate mapping reused from independently audited commonbank with hashes. All A-D choices independently sorted, exact parentA membership, all summary/paired family intervals reproduced. Finite future-oracle diagnostic only.'});print('AUDITED FACTORIAL',analysis['states'],len(f),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();main(args.smoke)
