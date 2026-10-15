"""Raw physical and mathematical audit of direction utility; family summaries."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def pd_trace(actions,root):
    p=np.array(root[:2],float);v=np.array(root[5:7],float);out=[]
    for action in actions:
        target=p+100*np.array(action,float)
        for _ in range(10):v=v+.01*(100*(target-p)-20*v);p=p+.01*v;out.append(p.copy())
    return np.array(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=BASE/('direction_utility_smoke' if args.smoke else 'direction_utility');m=read(out/'manifest.json');cfg=m['config'];done=read(out/'execution_complete.json');assert done['smoke']==args.smoke and done['manifest_sha256']==sha256(out/'manifest.json')
    for p,h in {**m['inputs'],**m['parents']}.items():assert sha256(resolve(p))==h,p
    rule=read(BASE/'selection/selector_rules.json')['rules']['controller_pusher'];errors={};state_rows=[];curves=[];total=0
    def check(name,a,b,atol=1e-8,rtol=1e-9):
        a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(name,a.shape,b.shape);diff=abs(a.astype(float)-b.astype(float));e=float(np.max(diff)) if a.size else 0.;errors[name]=max(errors.get(name,0.),e);assert np.all(diff<=atol+rtol*abs(b)),(name,e)
    for path in sorted(out.glob('development*_h05.json')):
        meta=read(path);assert meta['manifest_sha256']==sha256(out/'manifest.json') and meta['data_sha256']==sha256(path.with_suffix('.npz')) and meta['csv_sha256']==sha256(path.with_suffix('.csv'));f=pd.read_csv(path.with_suffix('.csv'))
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:a={k:z[k] for k in z.files}
        with np.load(BASE/'interventions'/path.with_suffix('.npz').name) as parent:
            for k in ['root','jacobian','basis']:check('parent_'+k,a[k],parent[k],0,0)
            check('parent_nominal_physics',a['physics'][0],parent['physics'][0],0,0);check('parent_nominal_action',a['actions'][0],parent['actions'][0],0,0)
        assert len(f)==meta['candidates']==60 and len(a['actions'])==61;total+=len(f)
        for name,v in a.items():assert np.isfinite(v).all(),name
        for name in ['physics','control']:assert a[name].shape[0]==61
        check('control_physics',a['control'],a['physics'][:,9::10],0,0)
        phys=a['physics'];position=np.linalg.norm(phys[:,30:,2:4]-phys[0:1,30:,2:4],axis=-1).max(1);d=phys[:,30:,4]-phys[0:1,30:,4];angle=abs(np.arctan2(np.sin(d),np.cos(d))).max(1);coverage=np.maximum(0,a['coverage'][0:1]-a['coverage']).max(1);obj=np.maximum.reduce([position/2,angle/np.deg2rad(2),coverage/.02]);jerk=np.sum(np.diff(phys[:,:,:2],n=3,axis=1)**2,axis=(1,2))/.01**5;gain=(jerk[0]-jerk)/jerk[0] if jerk[0]>1e-8 else np.zeros(61)
        norms=np.linalg.norm(((a['actions'][1:]-a['actions'][0])/a['scale']).reshape(60,-1),axis=1)
        for name,actual,stored in [('object_error',obj[1:],f.object_error),('physical_jerk',jerk[1:],f.physics_jerk),('jerk_gain',gain[1:],f.jerk_gain),('norm',norms,f.norm)]:check(name,actual,stored)
        assert np.array_equal(obj[1:]<=1,f.preserved);q=a['basis'];check('basis_orthonormal',q.T@q,np.eye(4),1e-12);check('coefficients_to_actions',(a['coefficients']@q.T).reshape(60,5,2)*a['scale'],a['actions'][1:]-a['actions'][0],1e-12)
        nominal=a['actions'][0];suffix=a['suffix'];base=pd_trace(np.r_[nominal,suffix],a['root']);check('independent_base',base,a['base'],1e-10);prefix=phys[0,:30,:2];cost=lambda pts:float(np.sum(np.diff(np.r_[prefix,pts],n=3,axis=0)**2)/.01**5);cost0=cost(base)
        response=np.stack([pd_trace(np.r_[nominal+q[:,i].reshape(5,2)*a['scale'],suffix],a['root'])-base for i in range(4)],axis=-1);check('independent_response',response,a['response'],1e-10)
        j0=np.diff(np.r_[prefix,base],n=3,axis=0).reshape(-1)/.01**3;design=np.diff(np.r_[np.zeros((30,2,4)),response],n=3,axis=0).reshape(-1,4)/.01**3;check('j0',j0,a['j0'],1e-7);check('design',design,a['design'],1e-7);gradient=2*.01*design.T@j0;check('gradient',gradient,a['gradient'],1e-4);relative=gradient/max(cost0,1.);check('relative_gradient',relative,a['relative_gradient'],1e-10)
        for eps in [1e-3,1e-4]:
            fd=np.array([(cost(base+eps*response[:,:,i])-cost(base-eps*response[:,:,i]))/(2*eps*max(cost0,1.)) for i in range(4)]);check('finite_difference_gradient',fd,relative,2e-6,2e-6)
        valid=np.linalg.norm(relative)>cfg['gradient_relative_floor'];assert valid==meta['gradient_valid'];robot=response[:50].reshape(-1,4);cmat=robot.T@robot;wm=a['jacobian'].astype(float)@q;wmat=wm.T@wm
        for name,mat in [('controller',cmat),('lewm',wmat)]:
            b=a['subspace_'+name+'_low'];eig=np.linalg.eigvalsh(mat);check(name+'_low_trace',np.trace(b.T@mat@b),eig[:2].sum(),1e-6,1e-7)
        allpred=np.stack([pd_trace(np.r_[raw,suffix],a['root']) for raw in a['actions'][1:]]);check('independent_candidate_prediction',allpred,a['predicted'],1e-10);check('actual_controller_position',allpred,phys[1:,30:,:2],1e-3,0);costs=np.array([cost(p) for p in allpred]);check('predicted_cost',costs,a['costs'],1e-2,1e-8);check('csv_predicted_cost',costs,f.predicted_jerk,1e-2,1e-8);scores=np.linalg.norm(allpred-base[None],axis=-1).max(1);check('score',scores,a['scores'],1e-9);check('csv_score',scores,f.controller_score,1e-9)
        allowed=(norms>1e-6)&(norms<=rule['relative_radius']*np.sqrt(10)+1e-7)&(scores<=rule['tau']);assert np.array_equal(allowed,f.approved)
        commonbound=np.inf
        for name in cfg['methods']:
            b=a['subspace_'+name];k=4 if name=='full' else 2;check('subspace_orthonormal',b.T@b,np.eye(k),1e-12);v=a['direction_'+name];check('unit_direction',np.linalg.norm(v),1.,1e-12);check('direction_in_subspace',b@(b.T@v),v,1e-12)
            eta=float(np.linalg.norm(b.T@relative)**2/(relative@relative)) if valid else None
            if valid:check('eta',eta,meta['eta'][name],1e-10);check('csv_eta',np.array([eta]*12),f[f.method==name].eta.to_numpy(),1e-10)
            if valid and name in ['full','lewm_plus_gradient']:check('gradient_capture',eta,1.,1e-10)
            raw=(q@v).reshape(5,2)*a['scale'];use=abs(raw)>1e-14;commonbound=min(commonbound,float(np.min((1-abs(nominal[use]))/abs(raw[use]))))
            ids=np.flatnonzero(f.method.to_numpy()==name);assert len(ids)==12;z=a['coefficients'][ids];check('coefficient_in_subspace',z@b@b.T,z,1e-10);assert np.all(np.linalg.norm(z,axis=1)<=f.iloc[ids].radius.to_numpy()+1e-6)
            candidate=ids[allowed[ids]];selected=int(candidate[np.lexsort((norms[candidate],costs[candidate]))[0]]) if len(candidate) else -1
            if selected>=0 and (cost0-costs[selected])/max(cost0,1e-8)<=1e-8:selected=-1
            assert selected+1==meta['controller_selected'][name];assert f.iloc[ids].selected.sum()==int(selected>=0)
            active=norms[ids]>1e-6;useful=active&(obj[ids+1]<=1)&(gain[ids+1]>=.1);safe=active&(obj[ids+1]<=1);chosen=selected+1
            state_rows.append({'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'method':name,'dimension':k,'eta':eta,'nominal_contact':bool(f.nominal_contact.iloc[0]),'oracle_useful':bool(useful.any()),'oracle_safe':bool(safe.any()),'oracle_best_gain':max(0,float(gain[ids[safe]+1].max())) if safe.any() else 0.,'controller_active':selected>=0,'controller_useful':bool(selected>=0 and obj[chosen]<=1 and gain[chosen]>=.1),'controller_violation':bool(selected>=0 and obj[chosen]>1),'controller_gain':float(gain[chosen]),'requested_candidates':12,'unique_actions':len(np.unique(np.round(a['actions'][ids+1].reshape(12,-1),12),axis=0)),'active_candidates':int(active.sum()),'optimizer_success':float(f.iloc[ids][f.iloc[ids].kind=='optimizer'].solver_success.mean())})
        check('common_bound',commonbound,meta['common_bound'],1e-10)
        for row in f.itertuples():
            radius=max(0,min(row.relative_radius*np.sqrt(10),.999*commonbound));check('radius',radius,row.radius,1e-12)
            if row.kind=='signed':check('signed_coefficients',a['coefficients'][row.index-1],row.sign*radius*a['direction_'+row.method],1e-12)
        for (method,rr,kind),group in f.groupby(['method','relative_radius','kind']):
            ids=group['index'].to_numpy(int);curves.append({'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'method':method,'relative_radius':float(rr),'kind':kind,'preservation':float((obj[ids]<=1).mean()),'useful':float(((obj[ids]<=1)&(gain[ids]>=.1)&(norms[ids-1]>1e-6)).mean()),'mean_gain':float(gain[ids].mean())})
        print('AUDITED',meta['state_id'],flush=True)
    frame=pd.DataFrame(state_rows);curve=pd.DataFrame(curves);assert frame.state_id.nunique()==done['states'];frame.to_csv(out/'state_metrics.csv',index=False);curve.to_csv(out/'response_curves.csv',index=False);summary=[]
    for method,g in frame.groupby('method'):
        summary.append({'method':method,'states':len(g),'dimension':int(g.dimension.iloc[0]),**{k:mean_ci(g.dropna(subset=[k]),k) for k in ['eta','oracle_useful','oracle_best_gain','controller_active','controller_useful','controller_gain','unique_actions','active_candidates','optimizer_success']},'controller_selected_risk':mean_ci(g[g.controller_active],'controller_violation')})
    contrasts=[]
    for method in ['controller_low','random','lewm_plus_gradient','full']:
        f=frame[frame.method==method].merge(frame[frame.method=='lewm_low'],on=['state_id','initial_pose_family'],suffixes=('','_lewm'))
        for metric in ['eta','oracle_useful','controller_useful','controller_gain']:
            f['difference']=f[metric].astype(float)-f[metric+'_lewm'].astype(float);contrasts.append({'method':method,'reference':'lewm_low','metric':metric,**mean_ci(f.dropna(subset=['difference']),'difference')})
    write_json(out/'analysis.json',{'smoke':args.smoke,'summary':summary,'paired_contrasts':contrasts,'scope':cfg['scope'],'interpretation':'Development exploratory basis comparison, full reference has more dimensions. Oracle is finite bank and future labels. Fixed controller rule is transferred, not calibrated to this bank. More gradient captured does not imply physical safety; same requested count, actual unique/active count reported.'});write_json(out/'raw_audit.json',{'smoke':args.smoke,'states':done['states'],'families':frame.initial_pose_family.nunique(),'candidates':total,'source_sha256':sha256(Path(__file__)),'manifest_sha256':sha256(out/'manifest.json'),'analysis_sha256':sha256(out/'analysis.json'),'max_error':errors,'scope':'Raw physical E/jerk, paired nominal exact, basis/eigenspace/gradient finite differences, causal PD prediction, equal radius/signed norm, polynomial/subspace membership, original controller rule and selected outcomes.'});print('COMPLETE',done['states'],total,flush=True)
if __name__=='__main__':main()
