"""F4 zero-intervention/repeated-execution observation and readout controls."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,joblib
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import DenseSimulator
from zj_wam.model_audit import encode_representations,predict_candidates
from zj_wam.readouts import decode,effects,history_features
from zj_wam.io import resolve,write_json
out=resolve('artifacts/followup_v1');cfg=json.loads(resolve('configs/followup_v1.json').read_text());torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);rows=[]
readouts={name:joblib.load(out/f'lewm_audit/readouts/{name}_ridge.joblib') for name in ['projected_cls','history_projected']}
selected=json.loads((out/'shift_preflight/protocol.json').read_text())['states'];dest=out/'lewm_audit/repeat_controls';dest.mkdir(exist_ok=True)
for sid in selected:
 spec=json.loads((out/'f1'/f'{sid}_h05.json').read_text());a,s=data.episode(spec['episode']);t=spec['timestep'];sim=DenseSimulator(a,s,t,cfg['seed']+spec['episode'])
 images,past=sim.restore(capture=True);first=sim.execute_dense(a[t:t+5],a[t+5:t+8],restore=False,capture_endpoints=True)
 second_images,second_past=sim.restore(capture=True);second=sim.execute_dense(a[t:t+5],a[t+5:t+8],restore=False,capture_endpoints=True)
 endpoints=np.concatenate([first['endpoint_images'],second['endpoint_images']]);features=encode_representations(model,endpoints)['projected_cls'][:,None,:]
 _,context=model.predictor(images,past);hist=context['embedding'].cpu().numpy();pred=predict_candidates(model,context['embedding'],context['past'],np.stack([a[t:t+5],a[t:t+5]]))
 row={'state_id':sid,'initial_pose_family':spec['initial_pose_family'],'physical_max':float(np.max(np.abs(first['physics_trace']-second['physics_trace']))),
      'history_pixel_max':int(np.max(np.abs(np.asarray(images).astype(int)-np.asarray(second_images).astype(int)))),
      'endpoint_pixel_max':int(np.max(np.abs(endpoints[0].astype(int)-endpoints[1].astype(int)))),
      'real_latent_max':float(np.max(np.abs(features[0]-features[1]))),'zero_intervention_prediction_max':float(np.max(np.abs(pred[0]-pred[1])))}
 for name in readouts:
  x=features if name=='projected_cls' else history_features(hist,features);delta=effects(decode(readouts[name],x))[1,0]
  row[name+'_readout_position_delta_px']=float(np.linalg.norm(delta[:2]));row[name+'_readout_rotation_delta_deg']=float(np.rad2deg(abs(delta[2])))
 rows.append(row);np.savez_compressed(dest/f'{sid}.npz',history_images=images,repeated_history_images=second_images,endpoints=endpoints,real_native=features,predicted_native=pred);sim.close()
summary={key:max(row[key] for row in rows) for key in rows[0] if key not in ['state_id','initial_pose_family']}
write_json(dest/'audit.json',{'states':len(rows),'families':len({r['initial_pose_family'] for r in rows}),'scope':'Deterministic nominal repeat and zero intervention control; does not establish physical effect resolution or observation-noise robustness.','maxima':summary,'rows':rows});print(summary)
