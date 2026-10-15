"""F3: collect native representations and fit isolated physical-state readouts."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import DenseSimulator,angular_error
from zj_wam.model_audit import encode_representations
from zj_wam.io import resolve,write_json,sha256
from scripts.run_followup import initialize


def collect(spec,cfg,data,model,out):
    directory=out/'lewm_audit/representations';directory.mkdir(parents=True,exist_ok=True)
    key=f"{spec['split']}_f{spec['initial_pose_family']:03d}"
    if (directory/f'{key}.json').exists():return
    began=time.perf_counter();a,s=data.episode(spec['episode'])
    times=np.unique(np.linspace(10,len(a)-1,cfg['probe_frames_per_family']).astype(int))
    wanted=sorted(set(int(t-k) for t in times for k in [10,5,0]))
    sim=DenseSimulator(a,s,0,cfg['seed']+spec['episode']);sim.restore()
    images=[];labels=[];recorded_times=[]
    for t in range(max(wanted)+1):
        if t in wanted:
            images.append(sim.env.render().copy());labels.append(sim.full_observation());recorded_times.append(t)
        if t<max(wanted):sim.env.step(a[t])
    sim.close();features=encode_representations(model,images)
    index={t:i for i,t in enumerate(recorded_times)};current=[index[t] for t in times]
    history=np.array([[index[t-10],index[t-5],index[t]] for t in times])
    full=np.asarray(labels)[current]
    target=np.column_stack([full[:,2:4],np.sin(full[:,4]),np.cos(full[:,4]),full[:,:2],full[:,7:10]])
    values={k:v[current] for k,v in features.items()}
    values['history_projected']=features['projected_cls'][history].reshape(len(times),-1)
    np.savez_compressed(directory/f'{key}.npz',**values,targets=target,full_state=full,timesteps=times,
                        family=np.repeat(spec['initial_pose_family'],len(times)))
    record={**spec,'frames':len(times),'encoded_frames':len(images),'duration_seconds':time.perf_counter()-began,
            'source_sha256':sha256(Path(__file__))}
    write_json(directory/f'{key}.json',record);print(json.dumps(record),flush=True)


def load_split(out,split):
    paths=sorted((out/'lewm_audit/representations').glob(f'{split}_*.npz'))
    if not paths:raise ValueError(f'No representations for {split}')
    values=[dict(np.load(p)) for p in paths]
    return {k:np.concatenate([v[k] for v in values]) for k in values[0]}


def physical_errors(pred,true):
    pos=np.linalg.norm(pred[:,:2]-true[:,:2],axis=1)
    angle=angular_error(np.arctan2(pred[:,2],pred[:,3]),np.arctan2(true[:,2],true[:,3]))
    pusher=np.linalg.norm(pred[:,4:6]-true[:,4:6],axis=1)
    result={'position_mae':float(pos.mean()),'position_p90':float(np.quantile(pos,.9)),
            'rotation_mae_deg':float(np.rad2deg(angle.mean())),
            'rotation_p90_deg':float(np.rad2deg(np.quantile(angle,.9))),
            'pusher_mae':float(pusher.mean()),'within_2px_2deg':float(np.mean((pos<=2)&(angle<=np.deg2rad(2))))}
    if pred.shape[1]==9:
        result['velocity_mae']=float(np.linalg.norm(pred[:,6:8]-true[:,6:8],axis=1).mean())
        result['angular_velocity_mae_deg_s']=float(np.rad2deg(np.abs(pred[:,8]-true[:,8]).mean()))
    return result


def fit(cfg,out):
    destination=out/'lewm_audit/readouts';destination.mkdir(parents=True,exist_ok=True)
    if (destination/'selection.json').exists():
        print('Frozen readout selection already exists');return
    train=load_split(out,'probe_train');dev=load_split(out,'development')
    assert set(train['family']).isdisjoint(dev['family'])
    rows=[];selected={};predictions={};rng=np.random.default_rng(cfg['seed'])
    for name in ['projected_cls','encoder_cls','middle_cls','spatial_quadrants','history_projected']:
        dims=9 if name=='history_projected' else 6
        xs=StandardScaler().fit(train[name]);ys=StandardScaler().fit(train['targets'][:,:dims])
        x=xs.transform(train[name]);y=ys.transform(train['targets'][:,:dims]);xd=xs.transform(dev[name]);true=dev['targets'][:,:dims]
        candidates=[]
        for alpha in [.1,1.,10.,100.,1000.]:
            estimator=Ridge(alpha=alpha).fit(x,y)
            pred=ys.inverse_transform(estimator.predict(xd));errors=physical_errors(pred,true)
            score=errors['position_mae']/2+errors['rotation_mae_deg']/2
            rows.append({'representation':name,'readout':'ridge','alpha':alpha,'selection_score':score,**errors})
            candidates.append((score,alpha,estimator,pred))
        score,alpha,estimator,pred=min(candidates,key=lambda z:z[0])
        selected[name]={'alpha':alpha,'development_score':score,'native_future_available':name=='projected_cls'}
        joblib.dump({'input_scaler':xs,'output_scaler':ys,'estimator':estimator,'representation':name},destination/f'{name}_ridge.joblib')
        predictions[name+'_ridge']=pred
        shuffled=Ridge(alpha=alpha).fit(x,y[rng.permutation(len(y))])
        shuffled_pred=ys.inverse_transform(shuffled.predict(xd))
        predictions[name+'_shuffle']=shuffled_pred
        rows.append({'representation':name,'readout':'label_shuffle','alpha':alpha,**physical_errors(shuffled_pred,true)})
        mean=np.broadcast_to(train['targets'][:,:dims].mean(0),true.shape)
        rows.append({'representation':name,'readout':'mean','alpha':0.,**physical_errors(mean,true)})
        # Fixed small nonlinear capacity check; no test-driven architecture search.
        nonlinear=MLPRegressor(hidden_layer_sizes=(64,),activation='tanh',alpha=.01,batch_size=128,
                               max_iter=400,early_stopping=False,random_state=cfg['seed'])
        nonlinear.fit(x,y);npred=ys.inverse_transform(nonlinear.predict(xd))
        rows.append({'representation':name,'readout':'mlp64','alpha':.01,**physical_errors(npred,true),
                     'iterations':int(nonlinear.n_iter_)})
        predictions[name+'_mlp64']=npred
        joblib.dump({'input_scaler':xs,'output_scaler':ys,'estimator':nonlinear,'representation':name},destination/f'{name}_mlp64.joblib')
    pd.DataFrame(rows).to_csv(destination/'development_metrics.csv',index=False)
    np.savez_compressed(destination/'development_predictions.npz',**predictions,targets=dev['targets'],family=dev['family'])
    write_json(destination/'selection.json',{'scope':'Development-only capacity diagnosis. Diagnostic readouts; deployed use requires explicit supervised variant.',
                'train_families':len(np.unique(train['family'])),'development_families':len(np.unique(dev['family'])),
                'train_frames':len(train['family']),'development_frames':len(dev['family']),
                'selected_ridge':selected,'source_sha256':sha256(Path(__file__))})
    print(pd.DataFrame(rows)[['representation','readout','position_mae','rotation_mae_deg']].to_string(index=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','fit']);p.add_argument('--split',default='probe_train',choices=['probe_train','development','calibration','test']);p.add_argument('--limit',type=int)
    args=p.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text());out=resolve(cfg['output'])
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    if args.mode=='fit':fit(cfg,out);return
    if args.split=='test' and not (out/'protocol/final_selection_lock.json').exists():raise ValueError('Final selection lock required')
    data=ExpertDataset(cfg['dataset']);manifest=initialize(cfg,data,out);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
    episodes=[s for s in manifest['episodes'] if s['split']==args.split]
    if args.limit:episodes=episodes[:args.limit]
    for spec in episodes:collect(spec,cfg,data,model,out)

if __name__=='__main__':main()
