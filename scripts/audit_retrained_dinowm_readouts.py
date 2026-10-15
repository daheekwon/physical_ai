"""Independently recompute raw-feature inference, geometry, fits and family intervals."""
import argparse,itertools,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import Ridge
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_readouts';read=lambda p:json.loads(p.read_text())


def features(env,split,cfg):
    capture=BASE/'dinowm_capture';manifest=read(capture/'manifest.json');xs={};ys=[];families=[];ids=[]
    for filename in manifest['states'][env][split]:
        src=resolve(filename);path=capture/env/src.with_suffix('.npz').name if env=='pusht' else capture/env/split/src.with_suffix('.npz').name
        with np.load(path,allow_pickle=False) as a:
            ys.append(a['targets']);families.extend(a['family']);ids.extend(src.stem+'_'+str(i) for i in range(len(a['targets'])))
            for rep in cfg['dino_representations']:xs.setdefault('dino_'+rep,[]).append(a[rep])
        with np.load(src.with_suffix('.npz'),allow_pickle=False) as a:
            if env=='pusht':
                np.testing.assert_array_equal(a['targets'],ys[-1])
                for old,new in [('projected_cls','frame'),('spatial_quadrants','spatial'),('history_projected','history')]:xs.setdefault('lewm_'+new,[]).append(a[old])
            else:
                h=a['history_embedding'];e=a['future_embedding'] if split=='probe_train' else a['actual_endpoint_latent'][0]
                xs.setdefault('lewm_frame',[]).append(np.stack([h[-1],e]));xs.setdefault('lewm_history',[]).append(np.stack([h.reshape(-1),np.concatenate([h[1:].reshape(-1),e])]))
    xs={k:np.concatenate(v).astype(float) for k,v in xs.items()};q=np.load(OUT/'projection.npy',allow_pickle=False)
    for rep in cfg['dino_representations']:
        x=xs['dino_'+rep];xs['dino_projected_'+rep]=np.concatenate([x[:,i:i+384]@q for i in range(0,x.shape[1],384)],axis=1)
    return xs,np.concatenate(ys).astype(float),np.asarray(families),np.asarray(ids)


def proper_rotation(y):
    u,_,vt=np.linalg.svd(y[:,3:12].reshape(-1,3,3));r=u@vt;bad=np.linalg.det(r)<0;u[bad,:,-1]*=-1;return u@vt


def symmetries():
    result=[]
    for order in itertools.permutations(range(3)):
        for signs in itertools.product([-1,1],repeat=3):
            m=np.eye(3)[list(order)]*np.array(signs)[:,None]
            if np.linalg.det(m)>.5:result.append(m)
    assert len(result)==24
    return np.array(result)


def metric(env,p,t):
    if env=='pusht':
        angle=np.abs((np.arctan2(p[:,2],p[:,3])-np.arctan2(t[:,2],t[:,3])+np.pi)%(2*np.pi)-np.pi);pos=np.linalg.norm(p[:,:2]-t[:,:2],axis=1)
        return {'position_px':pos,'rotation_rad':angle,'pusher_px':np.linalg.norm(p[:,4:6]-t[:,4:6],axis=1),'joint_pose':(pos<=2)&(angle<=np.deg2rad(2))}
    rp,rt=proper_rotation(p),proper_rotation(t);angle=np.arccos(np.clip((np.einsum('nij,nij->n',rp,rt)-1)/2,-1,1));candidates=rt[:,None]@symmetries()[None];sc=np.einsum('nij,nsij->ns',rp,candidates);sa=np.arccos(np.clip((sc.max(1)-1)/2,-1,1));pos=np.linalg.norm(p[:,:3]-t[:,:3],axis=1)
    return {'position_m':pos,'rotation_rad':angle,'rotation_mod_cube_symmetry_rad':sa,'linear_velocity_m_s':np.linalg.norm(p[:,12:15]-t[:,12:15],axis=1),'angular_velocity_rad_s':np.linalg.norm(p[:,15:18]-t[:,15:18],axis=1),'eef_position_m':np.linalg.norm(p[:,18:21]-t[:,18:21],axis=1),'joint_pose':(pos<=.002)&(angle<=np.deg2rad(2)),'joint_pose_symmetry':(pos<=.002)&(sa<=np.deg2rad(2))}


def infer(m,x):
    z=np.divide(x-m['x_mean'],m['x_scale'])
    if 'coef' in m:z=np.einsum('ij,kj->ik',z,m['coef'])+m['intercept']
    else:z=np.tanh(np.einsum('ij,jk->ik',z,m['w0'])+m['b0'])@m['w1']+m['b1']
    return m['y_mean']+z*m['y_scale']


def cluster(values,families):
    grouped=np.array([[np.sum(values[families==f]),np.count_nonzero(families==f)] for f in np.unique(families)])
    draws=np.random.default_rng(20261011).integers(len(grouped),size=(2000,len(grouped)));g=grouped[draws].sum(1)
    return float(np.sum(grouped[:,0])/np.sum(grouped[:,1])),np.quantile(g[:,0]/g[:,1],[.025,.975])


def audit():
    protocol=read(OUT/'protocol.json');lock=read(OUT/'data_lock.json');done=read(OUT/'fit_complete.json');selection=read(OUT/'selection.json');analysis=read(OUT/'analysis.json');cfg=protocol['config'];errors={};counts={'configurations':0,'learners':0,'evaluations':0,'physical_metric_rows':0,'alpha_trials':0}
    def check(name,x,y,tol=1e-8):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,(name,x.shape,y.shape)
        delta=float(np.max(np.abs(x.astype(float)-y.astype(float))));errors[name]=max(errors.get(name,0.),delta);assert delta<=tol,(name,delta,tol)
    for p,h in {**protocol['inputs'],**lock['inputs']}.items():assert sha256(resolve(p))==h,p
    assert done['selection_sha256']==sha256(OUT/'selection.json') and done['analysis_sha256']==sha256(OUT/'analysis.json')
    q=np.load(OUT/'projection.npy');check('projection_orthogonal',q.T@q,np.eye(192),1e-12)
    expected_q=np.linalg.qr(np.random.default_rng(cfg['projection']['seed']).normal(size=(384,192)))[0];check('projection_seed',q,expected_q,0.)
    for env in cfg['tasks']:
        data={s:features(env,s,cfg) for s in cfg['splits']};xtr,ytr,ftr,_=data['probe_train'];folder=OUT/env;shuffle=np.load(folder/'shuffle.npy');check('shuffle_seed',shuffle,np.random.default_rng(cfg['seed']).permutation(len(ytr)),0.)
        for i,s in enumerate(cfg['splits']):
            for other in cfg['splits'][i+1:]:assert not set(data[s][2])&set(data[other][2])
        for rep,x in xtr.items():
            for kind in (['raw'] if env=='pusht' else ['raw','symmetry']):
                key=rep+'__'+kind;r=read(folder/(key+'.json'));counts['configurations']+=1;assert r['data_lock_sha256']==sha256(OUT/'data_lock.json')
                for name,h in r['files'].items():assert sha256(folder/name)==h,name
                y=ytr[:,:9 if 'history' in rep else 6] if env=='pusht' else ytr.copy()
                if kind=='symmetry':
                    # Independently reproduce identity-nearest symmetry representative and lexicographic tie break.
                    matrices=proper_rotation(y)[:,None]@symmetries()[None];traces=np.trace(matrices,axis1=2,axis2=3);chosen=[]
                    for mats,scores in zip(matrices,traces):
                        tied=np.flatnonzero(scores>=scores.max()-1e-10);index=min(tied,key=lambda j:tuple(np.round(mats[j].reshape(-1),10)));chosen.append(mats[index])
                    y[:,3:12]=np.asarray(chosen).reshape(-1,9)
                xm=x.mean(0);xs=x.std(0);xs[xs==0]=1.;ym=y.mean(0);ys=y.std(0);ys[ys==0]=1.;xt=(x-xm)/xs;yt=(y-ym)/ys
                devx,devy,_,_=data['development'];devtruth=devy[:,:y.shape[1]];curves=[]
                for a in cfg['alphas'][env]:
                    estimator=Ridge(alpha=a).fit(xt,yt);pred=estimator.predict((devx[rep]-xm)/xs)*ys+ym;e=metric(env,pred,devtruth);value=float(np.mean(e['position_px']/2+e['rotation_rad']/np.deg2rad(2))) if env=='pusht' else float(np.mean(e['position_m']/.002+e['rotation_mod_cube_symmetry_rad' if kind=='symmetry' else 'rotation_rad']/np.deg2rad(2)))
                    recorded=next(t for t in r['curves'] if t['alpha']==a);check('development_alpha_score',value,recorded['development_score'],1e-6);curves.append((value,a));counts['alpha_trials']+=1
                assert min(curves,key=lambda z:z[0])[1]==r['selected_alpha']
                for learner in r['learner_info']:
                    counts['learners']+=1;stem=key+'__'+learner
                    with np.load(folder/(stem+'.npz'),allow_pickle=False) as raw:m=dict(raw)
                    if learner!='mean':
                        check('training_x_mean',m['x_mean'],xm,1e-10);check('training_x_scale',m['x_scale'],xs,1e-10);check('training_y_mean',m['y_mean'],ym,1e-10);check('training_y_scale',m['y_scale'],ys,1e-10)
                    if learner in ['ridge','label_shuffle']:
                        labels=y[shuffle] if learner=='label_shuffle' else y;expected=Ridge(alpha=r['selected_alpha']).fit(xt,(labels-ym)/ys)
                        check('ridge_refit_coef',m['coef'],expected.coef_,1e-7);check('ridge_refit_intercept',m['intercept'],expected.intercept_,1e-8)
                    for split in ['development','calibration']:
                        dx,dy,ff,ids=data[split];truth=dy[:,:y.shape[1]];pred=infer(m,dx[rep]);e=metric(env,pred,truth)
                        with np.load(folder/(stem+'__'+split+'.npz'),allow_pickle=False) as saved:
                            check('inference',pred,saved['predicted'],1e-8);check('truth_alignment',truth,saved['truth'],0.);np.testing.assert_array_equal(ff,saved['families']);np.testing.assert_array_equal(ids,saved['sample_ids'])
                            row=next(z for z in r['metrics'] if z['readout']==learner and z['split']==split)
                            for name,values in e.items():
                                check('metric_'+name,values,saved[name],1e-7);mean,ci=cluster(values,ff);check('family_mean',mean,row[name]['mean'],1e-7);check('family_ci',ci,row[name]['ci95'],1e-7);assert row[name]['families']==len(np.unique(ff))
                        counts['evaluations']+=1;counts['physical_metric_rows']+=len(truth)
    assert counts['configurations']==25 and counts['learners']==76 and counts['evaluations']==152 and counts['alpha_trials']==125,counts
    assert len(analysis['metrics'])==152 and done['configurations']==25 and done['metric_rows']==152
    for p,h in {**protocol['inputs'],**lock['inputs']}.items():assert sha256(resolve(p))==h,p
    write_json(OUT/'raw_audit.json',{'counts':counts,'max_error':errors,'source_sha256':sha256(Path(__file__)),'inputs':{name:sha256(OUT/name) for name in ['protocol.json','data_lock.json','selection.json','analysis.json','fit_complete.json']},'scope':'All raw-feature inference, heldout identities and physical errors; independent 24 cube symmetries/canonicalization, training-only scalers, all125 ridge alpha trials, shuffled-label ridge refit and family bootstrap. MLP optimizer not independently retrained; stored arrays/inference audited, sklearn parity checked at fit.'})
    print('AUDITED',counts,errors,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-pid',type=int);args=p.parse_args();start=time.monotonic()
    while not (OUT/'fit_complete.json').exists():
        assert args.dependency_pid,'Fit incomplete; specify live dependency'
        os.kill(args.dependency_pid,0)
        if time.monotonic()-start>7200:raise RuntimeError('Observation window elapsed; revalidate parent')
        time.sleep(5)
    audit()
if __name__=='__main__':main()
