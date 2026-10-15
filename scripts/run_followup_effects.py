"""F4 real-effect resolution and F5 imagined-effect propagation, same frozen readouts."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch,joblib
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.model_audit import encode_representations,predict_candidates
from zj_wam.readouts import decode,effects,errors,direction_cosine,history_features
from zj_wam.io import resolve,write_json,sha256


def process(path,cfg,acfg,model,out):
    dest=out/'lewm_audit/effects';dest.mkdir(parents=True,exist_ok=True)
    key=path.stem
    if (dest/f'{key}.json').exists():return
    started=time.perf_counter();a=dict(np.load(path));frame=pd.read_csv(path.with_suffix('.csv'))
    n,blocks=a['endpoint_images'].shape[:2];h=blocks*5
    rep=encode_representations(model,a['endpoint_images'].reshape(-1,224,224,3))
    rep={k:v.reshape(n,blocks,-1) for k,v in rep.items()}
    _,context=model.predictor(a['history_images'],a['past_actions'])
    hist=context['embedding'].cpu().numpy()
    rep['history_projected']=history_features(hist,rep['projected_cls'])
    prediction=a['predictions'] if 'predictions' in a else predict_candidates(model,context['embedding'],context['past'],a['actions'])
    if prediction.ndim==2:prediction=prediction[:,None,:]
    predrep={'projected_cls':prediction,'history_projected':history_features(hist,prediction)}
    # control trace has three identical prefix controls, then h+3 controls.
    true=a['control'][:,3+np.arange(4,h,5),2:5];true_effect=effects(true)
    physical_safe=frame['preserved'].to_numpy(bool)
    idx=frame['index'].to_numpy(int)
    if not np.array_equal(idx,np.arange(1,n)):raise ValueError('Candidate rows and traces are not aligned')
    wrong=np.arange(1,n);rng=np.random.default_rng(cfg['seed']+int(frame.episode.iloc[0])+int(frame.timestep.iloc[0]))
    for _,g in frame.groupby(['relative_radius','sign']):
        ids=g['index'].to_numpy(int).copy();rng.shuffle(ids);wrong[ids-1]=np.roll(ids,1)
    if np.any(wrong==np.arange(1,n)):raise ValueError('Wrong-action control contains self assignment')
    rows=[];saved={'true':true,'true_effect':true_effect,'wrong_indices':wrong}
    readouts=out/'lewm_audit/readouts'
    for name in acfg['representations']:
      for kind in acfg['readouts']:
        probe=joblib.load(readouts/f'{name}_{kind}.joblib')
        real_pose=decode(probe,rep[name]);real_effect=effects(real_pose)
        methods={'real_encoded':(real_pose,real_effect),'zero_effect':(np.broadcast_to(real_pose[0:1],real_pose.shape),np.zeros_like(real_effect))}
        if name in predrep:
            imagined=decode(probe,predrep[name]);imagined_effect=effects(imagined)
            wrong_pose=imagined[np.r_[0,wrong]];wrong_effect=imagined_effect[np.r_[0,wrong]]
            if name=='projected_cls':current=decode(probe,hist[:,-1:])
            else:current=decode(probe,hist.reshape(1,1,-1))
            persistence=np.broadcast_to(current,(n,blocks,3))
            methods.update({'predicted':(imagined,imagined_effect),'wrong_action':(wrong_pose,wrong_effect),
                            'persistence':(persistence,np.zeros_like(imagined_effect))})
        for method,(pose,delta) in methods.items():
            pe,re=errors(delta,true_effect);ape,are=errors(pose,true)
            cosine=direction_cosine(delta,true_effect)
            if method in ['real_encoded','predicted']:saved[f'{name}_{kind}_{method}_pose']=pose
            for block in range(blocks):
                table=pd.DataFrame({'state_id':frame.state_id,'initial_pose_family':frame.initial_pose_family,'split':frame.split,
                    'horizon':h,'endpoint_step':(block+1)*5,'index':idx,'representation':name,'readout':kind,'method':method,
                    'direction_method':frame.method,'relative_radius':frame.relative_radius,'nominal_contact':frame.nominal_contact,
                    'preserved':physical_safe,'position_effect':np.linalg.norm(true_effect[1:,block,:2],axis=1),
                    'rotation_effect':np.abs(true_effect[1:,block,2]),'position_effect_error':pe[1:,block],
                    'rotation_effect_error':re[1:,block],'position_absolute_error':ape[1:,block],
                    'rotation_absolute_error':are[1:,block],'position_cosine':cosine[1:,block],
                    'rotation_sign_correct':np.sign(delta[1:,block,2])==np.sign(true_effect[1:,block,2]),
                    'predicted_position_magnitude':np.linalg.norm(delta[1:,block,:2],axis=1),
                    'predicted_rotation_magnitude':np.abs(delta[1:,block,2]),
                    'nominal_position_error':ape[0,block],'nominal_rotation_error':are[0,block]})
                rows.append(table)
    pd.concat(rows,ignore_index=True).to_csv(dest/f'{key}.csv',index=False)
    np.savez_compressed(dest/f'{key}.npz',**saved,**{'real_'+k:v for k,v in rep.items()},predicted_projected=prediction)
    write_json(dest/f'{key}.json',{'state_id':key,'candidates':n-1,'blocks':blocks,'split':frame.split.iloc[0],
                                 'duration_seconds':time.perf_counter()-started,'source_sha256':sha256(Path(__file__))})
    print(key,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',default='development',choices=['development','calibration','test']);parser.add_argument('--limit',type=int)
    args=parser.parse_args();cfg=json.loads(resolve('configs/followup_v1.json').read_text());acfg=json.loads(resolve('configs/followup_effect_audit.json').read_text());out=resolve(cfg['output'])
    if args.split=='test' and not (out/'protocol/final_selection_lock.json').exists():raise ValueError('Test lock required')
    torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
    paths=sorted((out/'interventions').glob(f'{args.split}_*_h05.npz'))
    if args.limit:paths=paths[:args.limit]
    for path in paths:
        if path.with_suffix('.json').exists():process(path,cfg,acfg,model,out)

if __name__=='__main__':main()
