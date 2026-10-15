"""Independent subset, fit, raw-inference and geometry checks for probe repetitions."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from zj_wam.io import resolve,sha256,write_json
from scripts.audit_retrained_dinowm_readouts import features,proper_rotation,symmetries,infer,metric,cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=BASE/('probe_stability_smoke' if args.smoke else 'probe_stability');p=read(out/'protocol.json');cfg=p['config'];pc=p['parent_config'];done=read(out/'fit_complete.json');assert done['smoke']==args.smoke and done['protocol_sha256']==sha256(out/'protocol.json');lock=read(BASE/'dinowm_readouts/data_lock.json')
    for name,h in {**p['inputs'],**lock['inputs']}.items():assert sha256(resolve(name))==h,name
    errs={};rows=[];models=0;trials=0;reused=0;subsets={}
    def check(name,x,y,tol=1e-8):
        x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,(name,x.shape,y.shape);e=float(abs(x.astype(float)-y.astype(float)).max()) if x.size else 0.;errs[name]=max(errs.get(name,0.),e);assert e<=tol,(name,e,tol)
    for env in cfg['representations']:
        data={s:features(env,s,pc) for s in ['probe_train','development','calibration']};xall,yall,families,keys=data['probe_train'];kind=cfg['targets'][env]
        for s in ['development','calibration']:assert not set(families)&set(data[s][2])
        assert not set(data['development'][2])&set(data['calibration'][2])
        records=sorted((out/env).glob('*/*.json'))
        for path in records:
            r=read(path);folder=path.parent;assert done['record_hashes'][str(path.relative_to(out))]==sha256(path) and r['protocol_sha256']==sha256(out/'protocol.json');rep=r['representation'];group=r['group'];fraction=group['fraction'];seed=group['subset_seed'];unique=np.unique(families);selected=unique if seed is None else np.random.default_rng(seed).permutation(unique)[:math.ceil(len(unique)*fraction)];mask=np.isin(families,selected)
            with np.load(folder/'subset.npz') as raw:
                np.testing.assert_array_equal(raw['families'],selected);np.testing.assert_array_equal(raw['mask'],mask);np.testing.assert_array_equal(raw['sample_ids'],keys[mask]);assert sha256(folder/'subset.npz')==r['subset_sha256']
            subsets[(env,seed,fraction)]=set(selected)
            for name,h in r['files'].items():assert sha256(folder/name)==h,name
            x=xall[rep][mask];y=yall[mask].copy()
            if env=='pusht':y=y[:,:9 if 'history' in rep else 6]
            else:
                matrices=proper_rotation(y)[:,None]@symmetries()[None];scores=np.trace(matrices,axis1=2,axis2=3);canonical=[]
                for mats,sc in zip(matrices,scores):
                    tied=np.flatnonzero(sc>=sc.max()-1e-10);chosen=min(tied,key=lambda j:tuple(np.round(mats[j].reshape(-1),10)));canonical.append(mats[chosen])
                y[:,3:12]=np.array(canonical).reshape(-1,9)
            xm=x.mean(0);xs=x.std(0);xs[xs==0]=1.;ym=y.mean(0);ys=y.std(0);ys[ys==0]=1.;xt=(x-xm)/xs;yt=(y-ym)/ys;dx,dy,_,_=data['development'];truth=dy[:,:y.shape[1]];alpha_scores=[]
            sourcecurves=r['trials']
            if not sourcecurves:sourcecurves=read(BASE/'dinowm_readouts'/env/(rep+'__'+kind+'.json'))['curves']
            for alpha in pc['alphas'][env]:
                model=Ridge(alpha=alpha).fit(xt,yt);prediction=model.predict((dx[rep]-xm)/xs)*ys+ym;met=metric(env,prediction,truth);sc=float((met['position_px']/2+met['rotation_rad']/np.deg2rad(2)).mean()) if env=='pusht' else float((met['position_m']/.002+met['rotation_mod_cube_symmetry_rad']/np.deg2rad(2)).mean());stored=next(t for t in sourcecurves if t['alpha']==alpha);check('alpha_score',sc,stored['development_score'],1e-6);alpha_scores.append((sc,alpha));trials+=1
            assert min(alpha_scores)[1]==r['selected_alpha']
            shuffle=np.random.default_rng(pc['seed']).permutation(len(y))
            for learner in r['learners']:
                mp=folder/(rep+'__'+learner+'.npz')
                with np.load(mp,allow_pickle=False) as raw:m=dict(raw)
                models+=1
                if learner in r['reused_sources']:
                    src=r['reused_sources'][learner];assert sha256(resolve(src['path']))==src['sha256'];reused+=1
                    with np.load(resolve(src['path'])) as raw:
                        assert set(raw.files)==set(m)
                        for k in m:check('reused_model_arrays',m[k],raw[k],0.)
                if learner!='mean':
                    for k,v in [('x_mean',xm),('x_scale',xs),('y_mean',ym),('y_scale',ys)]:check('training_'+k,m[k],v,1e-10)
                if learner in ['ridge','label_shuffle']:
                    labels=y[shuffle] if learner=='label_shuffle' else y;ref=Ridge(alpha=r['selected_alpha']).fit(xt,(labels-ym)/ys);check('ridge_coef',m['coef'],ref.coef_,1e-7);check('ridge_intercept',m['intercept'],ref.intercept_,1e-8)
                if learner=='mean':check('mean_target',m['y_mean'],ym,1e-12);check('mean_coef',m['coef'],np.zeros_like(m['coef']),0.)
                for split in ['development','calibration']:
                    xx,yy,ff,ids=data[split];truth=yy[:,:y.shape[1]];pred=infer(m,xx[rep]);em=metric(env,pred,truth);record=next(t for t in r['metrics'] if t['learner']==learner and t['split']==split);ep=folder/(rep+'__'+learner+'__'+split+'.npz');outrow={k:v for k,v in record.items() if k!='metrics'}
                    with np.load(ep) as saved:
                        check('inference',pred,saved['predicted'],1e-8);check('truth',truth,saved['truth'],0.);np.testing.assert_array_equal(ids,saved['sample_ids']);np.testing.assert_array_equal(ff,saved['families'])
                        for k,v in em.items():
                            check('physical_'+k,v,saved[k],1e-7);avg,ci=cluster(v,ff);check('metric_mean',avg,record['metrics'][k],1e-7);outrow[k]={'mean':avg,'ci95':ci.tolist(),'families':len(np.unique(ff))}
                    rows.append(outrow)
    if not args.smoke:
        for env in cfg['representations']:
            for seed in cfg['subset_seeds']:assert subsets[(env,seed,.25)]<=subsets[(env,seed,.5)]<=subsets[(env,None,1.)]
    assert models==done['models']==(20 if args.smoke else 180);assert reused==(8 if args.smoke else 72);assert models-reused==done['new_models']
    flat=[]
    for row in rows:
        for metricname,v in row.items():
            if isinstance(v,dict) and 'mean' in v:flat.append({k:row[k] for k in ['environment','group','fraction','subset_seed','representation','target','learner','split']}|{'metric':metricname,'value':v['mean']})
    f=pd.DataFrame(flat);f.to_csv(out/'capture_metrics.csv',index=False);variation=[]
    for key,g in f.groupby(['environment','split','representation','fraction','learner','metric'],dropna=False):
        if key[3]<1:variation.append(dict(zip(['environment','split','representation','fraction','learner','metric'],key))|{'axis':'training_subset','replicates':len(g),'mean':float(g.value.mean()),'min':float(g.value.min()),'max':float(g.value.max())})
    full=f[(f.fraction==1)&f.learner.str.startswith('mlp64_seed_')]
    for key,g in full.groupby(['environment','split','representation','metric']):variation.append(dict(zip(['environment','split','representation','metric'],key))|{'axis':'mlp_initialization','fraction':1.,'learner':'mlp64','replicates':len(g),'mean':float(g.value.mean()),'min':float(g.value.min()),'max':float(g.value.max())})
    write_json(out/'capture_analysis.json',{'smoke':args.smoke,'metrics':rows,'training_variation':variation,'scope':cfg['scope'],'note':'Each fitted model has its own evaluation-family interval. Range across overlapping training subsets/initializations is descriptive, not a confidence interval over independent datasets. Alpha selected on development; calibration only evaluated. Effects not yet completed.'});write_json(out/'capture_raw_audit.json',{'smoke':args.smoke,'models':models,'reused_models':reused,'evaluations':len(rows),'alpha_trials_recomputed':trials,'max_error':errs,'source_sha256':sha256(Path(__file__)),'protocol_sha256':sha256(out/'protocol.json'),'fit_complete_sha256':sha256(out/'fit_complete.json'),'analysis_sha256':sha256(out/'capture_analysis.json'),'scope':'Raw independent feature assembly, family splits/subset nesting, train-only scalers, all ridge alphas/refits, reused arrays exact, inference and physical geometry including Cube symmetry; MLP optimizer not independently retrained.'});print('AUDITED CAPTURE',models,len(rows),trials,flush=True)
if __name__=='__main__':main()
