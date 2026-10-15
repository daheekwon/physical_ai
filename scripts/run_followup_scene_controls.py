"""F2: physically regenerated contact/near-miss and eccentric-push controls."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.followup import DenseSimulator,dense_metrics
from zj_wam.model_audit import predict_candidates
from zj_wam.io import resolve,write_json,sha256


def main():
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());out=resolve(cfg['output'])/'scene_controls';out.mkdir(exist_ok=True)
    design={'centers':[[220.,220.],[290.,200.]],'angles':[0.,.4,-.4],
            'local_x':{'center':0.,'eccentric':40.,'near_miss':76.},'local_y':-30.,
            'profiles':{'approach':[.2]*5,'release':[.35,.35,-.2,-.2,-.2],'recontact':[.35,.35,-.3,-.3,.35]},
            'signed_amplitudes':[-.08,-.04,-.02,.02,.04,.08],
            'history':'10 actual zero-action steps from each physically regenerated scene, context at steps 0,5,10',
            'scope':'Synthetic controlled diagnostics, not extra independent expert test families'}
    if (out/'design.json').exists():assert json.loads((out/'design.json').read_text())==design
    else:write_json(out/'design.json',design)
    torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
    for ci,center in enumerate(design['centers']):
      for ri,angle in enumerate(design['angles']):
       rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
       for location,x in design['local_x'].items():
        for profile,ys in design['profiles'].items():
         key=f'c{ci}_r{ri}_{location}_{profile}'
         if (out/f'{key}.json').exists():continue
         started=time.perf_counter();position=np.array(center)+rotation@np.array([x,design['local_y']])
         initial=np.r_[position,center,angle % (2*np.pi),0.,0.];states=np.tile(initial,(25,1));states[-1,2:4]+=rotation@np.array([0.,50.])
         actions=np.zeros((25,2));actions[10:15]=np.column_stack([np.zeros(5),ys])@rotation.T
         nominal=actions[10:15];suffix=actions[15:18]
         sim=DenseSimulator(actions,states,10,cfg['seed']+ci*10+ri);images,past=sim.restore(capture=True)
         root=sim.root_full_state.copy();ref=sim.execute_dense(nominal,suffix,restore=False,capture_endpoints=True)
         raw=[nominal]+[nominal+amp*np.tile(rotation@np.array([0.,1.]),(5,1)) for amp in design['signed_amplitudes']]
         _,context=model.predictor(images,past);pred=predict_candidates(model,context['embedding'],context['past'],raw)
         results=[ref]+[sim.execute_dense(a,suffix,capture_endpoints=True) for a in raw[1:]]
         repeat=sim.execute_dense(nominal,suffix)
         np.testing.assert_array_equal(ref['physics_trace'],repeat['physics_trace'])
         rows=[]
         for i,result in enumerate(results):
          contacts=result['physics_contacts'][:50]>0;on=np.flatnonzero(contacts)
          end=result['control'][4,2:5]-root[2:5]
          end[2]=np.arctan2(np.sin(end[2]),np.cos(end[2]))
          rows.append({'scene':key,'geometry_family':ci*3+ri,'location':location,'profile':profile,'index':i,
                       'amplitude':0. if i==0 else design['signed_amplitudes'][i-1],
                       'contact':bool(len(on)),'first_contact_seconds':float((on[0]+1)*.01) if len(on) else -1.,
                       'contact_seconds':float(contacts.sum()*.01),'contact_transitions':int(np.sum(contacts[1:]!=contacts[:-1])),
                       'object_translation':float(np.linalg.norm(end[:2])),'object_rotation_deg':float(np.rad2deg(abs(end[2]))),
                       'predicted_change':float(np.linalg.norm(pred[i,-1]-pred[0,-1])),**dense_metrics(result,ref,cfg['tolerances'])})
         pd.DataFrame(rows).to_csv(out/f'{key}.csv',index=False)
         np.savez_compressed(out/f'{key}.npz',actions=raw,physics=[r['physics_trace'] for r in results],
                             control=[r['control_trace'] for r in results],endpoint_images=[r['endpoint_images'] for r in results],
                             history_images=images,past_actions=past,root=root,predictions=pred,
                             contacts=[r['physics_contacts'] for r in results])
         write_json(out/f'{key}.json',{'scene':key,'rows':len(rows),'repeat_max':0.,'duration_seconds':time.perf_counter()-started,
                                      'source_sha256':sha256(Path(__file__))})
         sim.close();print(key,flush=True)
    frame=pd.concat([pd.read_csv(p) for p in out.glob('c*_*.csv')],ignore_index=True)
    frame.to_csv(out/'candidates.csv',index=False)
    summary=frame.groupby(['location','profile']).agg(n=('index','size'),contact=('contact','mean'),translation=('object_translation','mean'),rotation_deg=('object_rotation_deg','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    # Exact matched action/displacement support, relative to each scene's root.
    error=[]
    for ci in range(2):
     for ri in range(3):
      for profile in design['profiles']:
       paths=[]
       for location in design['local_x']:
        a=np.load(out/f'c{ci}_r{ri}_{location}_{profile}.npz')
        paths.append(a['physics'][:,30:,:2]-a['root'][:2])
       error.extend([float(np.max(np.abs(p-paths[0]))) for p in paths[1:]])
    write_json(out/'audit.json',{'scenes':54,'executions':len(frame),'max_matched_pusher_displacement_error':max(error),
                               'near_miss_contact_count':int(frame[frame.location=='near_miss'].contact.sum()),
                               'source_sha256':sha256(Path(__file__))})
    print(summary.to_string(index=False))

if __name__=='__main__':main()
