"""F10 preflight: verify that physical parameter interventions actually change dynamics."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.adapters import ExpertDataset
from zj_wam.followup import DenseSimulator,dense_metrics
from zj_wam.io import resolve,write_json
out=resolve('artifacts/followup_v1');cfg=json.loads(resolve('configs/followup_v1.json').read_text());data=ExpertDataset(cfg['dataset']);dest=out/'shift_preflight';dest.mkdir(exist_ok=True)
selected=[];families=set()
for p in sorted((out/'f1').glob('development*h05.json')):
 meta=json.loads(p.read_text());frame=pd.read_csv(p.with_suffix('.csv'),nrows=1)
 if bool(frame.nominal_contact.iloc[0]) and meta['initial_pose_family'] not in families:
  selected.append(meta);families.add(meta['initial_pose_family'])
 if len(selected)==12:break
conditions=[('repeat_nominal',{}),('density_half',{'density':.5}),('density_double',{'density':2.}),('object_shape_friction_only',{'object_friction':.8}),
            ('pair_friction_02',{'pair_friction':.2}),('pair_friction_08',{'pair_friction':.8}),('gain_08',{'kp_multiplier':.8}),('gain_12',{'kp_multiplier':1.2})]
write_json(dest/'protocol.json',{'scope':'Development mechanical preflight only; abrupt parameter change after the identical nominal prefix. Not smoothing adaptation or shift robustness evidence.',
 'states':[m['state_id'] for m in selected],'families':len(families),'conditions':conditions,
 'mass_rule':'Density scales both object mass and inertia; shape and center of gravity fixed.','friction_rule':'Pair friction value is implemented by setting both shape coefficients to sqrt(value). Object-only friction is an intentional null control when pusher friction=0.'})
rows=[]
for meta in selected:
 key=meta['state_id'];path=dest/f'{key}.json'
 if path.exists():rows.extend(json.loads(path.read_text())['rows']);continue
 a,s=data.episode(meta['episode']);t=meta['timestep'];nominal=a[t:t+5];suffix=a[t+5:t+8]
 sim=DenseSimulator(a,s,t,cfg['seed']+meta['episode']);ref=sim.execute_dense(nominal,suffix);physics=[ref['physics_trace']];current=[]
 for name,setting in conditions:
  sim.env.k_p=100;sim.env.k_v=20;sim.restore()
  if 'density' in setting:sim.env.block.mass*=setting['density'];sim.env.block.moment*=setting['density']
  if 'object_friction' in setting:
   for shape in sim.env.block.shapes:shape.friction=setting['object_friction']
  if 'pair_friction' in setting:
   for body in [sim.env.block,sim.env.agent]:
    for shape in body.shapes:shape.friction=np.sqrt(setting['pair_friction'])
  if 'kp_multiplier' in setting:sim.env.k_p*=setting['kp_multiplier']
  result=sim.execute_dense(nominal,suffix,restore=False)
  metrics=dense_metrics(result,ref,cfg['tolerances']);current.append({'state_id':key,'initial_pose_family':meta['initial_pose_family'],'condition':name,
    'actual_mass':sim.env.block.mass,'actual_moment':sim.env.block.moment,'shape_friction_product':next(iter(sim.env.block.shapes)).friction*next(iter(sim.env.agent.shapes)).friction,
    'kp':sim.env.k_p,**metrics});physics.append(result['physics_trace'])
 sim.close();np.savez_compressed(dest/f'{key}.npz',physics=physics,actions=nominal,suffix=suffix)
 write_json(path,{'state_id':key,'rows':current});rows.extend(current)
f=pd.DataFrame(rows);f.to_csv(dest/'results.csv',index=False)
summary=[]
for condition,g in f.groupby('condition'):
 summary.append({'condition':condition,'states':len(g),'max_position_delta':float(g.position_max.max()),'mean_position_delta':float(g.position_max.mean()),
                 'max_rotation_delta_deg':float(np.rad2deg(g.rotation_max.max())),'mean_rotation_delta_deg':float(np.rad2deg(g.rotation_max.mean())),
                 'object_changed_over_1e-6':int(((g.position_max>1e-6)|(g.rotation_max>1e-6)).sum()),'max_pusher_delta':float(g.pusher_max.max())})
write_json(dest/'analysis.json',{'scope':'Parameter-effect audit only; no deployed correction or independent shifted test','summary':summary})
for r in summary:print(r)
