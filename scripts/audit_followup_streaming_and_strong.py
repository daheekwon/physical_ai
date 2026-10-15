"""Recompute object trajectory, coverage and jerk from stored raw physics; no reruns/tuning."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,pymunk
from shapely.geometry import Polygon
from shapely.ops import unary_union
from zj_wam.adapters import ExpertDataset,ReplaySimulator
from zj_wam.io import resolve,write_json

def energy(q):return float(np.square(q[3:]-3*q[2:-1]+3*q[1:-2]-q[:-3]).sum()/.01**5)
def geometry(body,vertices):return unary_union([Polygon([body.local_to_world(v) for v in polygon]) for polygon in vertices])
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--banks',nargs='+',default=['strong_baselines','repeated_refinement','relinearization']);args=parser.parse_args()
 cfg=json.loads(resolve('configs/followup_v1.json').read_text());out=resolve(cfg['output']);data=ExpertDataset(cfg['dataset']);eps=cfg['tolerances'];reports={}
 for bank in args.banks:
  maxima={};states=0;rows=0;count=0
  def check(name,actual,expected,tol=1e-7):
   d=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))));maxima[name]=max(maxima.get(name,0),d)
   if d>tol:raise ValueError((bank,name,d,actual,expected))
  paths=sorted([* (out/bank).glob('development*.json'),* (out/bank).glob('calibration*.json')])
  assert len(paths)==(36 if bank=='strong_baselines_extended' else 210)
  for path in paths:
   m=json.loads(path.read_text());frame=pd.read_csv(path.with_suffix('.csv'));actions,recorded=data.episode(m['episode'])
   sim=ReplaySimulator(actions,recorded,m['timestep'],cfg['seed']+m['episode']);sim.env.reset(seed=cfg['seed']);vertices=[list(x.get_vertices()) for x in sim.env.block.shapes]
   body=pymunk.Body(body_type=pymunk.Body.KINEMATIC);body.center_of_gravity=sim.env.block.center_of_gravity
   body.angle=float(recorded[-1,4]);body.position=recorded[-1,2:4].tolist();target=geometry(body,vertices)
   def coverage(state):
    body.angle=float(state[4]);body.position=state[2:4].tolist();return geometry(body,vertices).intersection(target).area/target.area
   with np.load(path.with_suffix('.npz')) as a:
    h=5 if bank.startswith('strong_baselines') else 20;reference=a['physics'][0] if h==5 else a['nominal_physics'];nominal=a['actions'][0] if h==5 else a['nominal_actions']
    for row in frame.itertuples(index=False):
     trace=a['physics'][row.index] if h==5 else a[row.method+'_physics'];raw=a['actions'][row.index] if h==5 else a[row.method+'_actions'];check('identical_prefix',trace[:30],reference[:30],1e-9)
     check('action_norm',np.linalg.norm(((raw-nominal)/data.scaler.scale_).reshape(-1)),row.actual_norm,2e-6)
     for long in ([False] if h==5 else [False,True]):
      stop=len(trace) if long else 30+(h+3)*10;tr=trace[:stop];ref=reference[:stop];prefix='long_' if long else ''
      delta=tr[30:,2:4]-ref[30:,2:4];pos=float(np.linalg.norm(delta,axis=1).max());angles=tr[30:,4]-ref[30:,4];rot=float(np.abs(np.arctan2(np.sin(angles),np.cos(angles))).max())
      cover_t=max(0,coverage(ref[30+h*10-1])-coverage(tr[30+h*10-1]));cover_s=max(0,coverage(ref[-1])-coverage(tr[-1]))
      obj=max(pos/eps['position'],rot/eps['rotation'],cover_t/eps['coverage_loss'],cover_s/eps['coverage_loss'])
      if long and bank=='relinearization':obj=max(obj,row.object_error)
      ej=energy(tr[:,:2]);e0=energy(ref[:,:2]);gain=(e0-ej)/e0 if e0>1e-8 else 0
      for key,value in [('object_error',obj),('position_max',pos),('rotation_max',rot),('coverage_loss',cover_t),('coverage_suffix_loss',cover_s),('jerk_gain',gain)]:check(prefix+key,value,getattr(row,prefix+key))
      check(prefix+'physics_jerk_relative',ej/max(ej,1),getattr(row,prefix+'physics_jerk')/max(ej,1))
     if bank=='relinearization':
      basis=a[row.method+'_basis'];check('basis_orthonormal',np.swapaxes(basis,-2,-1)@basis,np.broadcast_to(np.eye(2),(4,2,2)))
      assert row.relinearizations=={'low_frozen':1,'low_refresh10':2,'low_refresh5':4}[row.method]
      if row.method=='low_frozen':check('frozen_basis',basis,np.broadcast_to(basis[0],basis.shape))
     rows+=1
   sim.close();states+=1
  reports[bank]={'states':states,'candidate_or_method_rows':rows,'max_absolute_recalculation_difference':maxima,'scope':'Full stored traces, coverage recomputed from raw poses and episode goal; no modified physics rerun. Long repeated_refinement raw E excludes core coverage; analysis uses max(core,long).'}
  write_json(out/bank/'raw_audit.json',reports[bank]);print(bank,states,rows,maxima,flush=True)
 combined=out/'protocol/streaming_and_strong_audit.json'
 previous=json.loads(combined.read_text()) if combined.exists() else {}
 previous.update(reports);write_json(combined,previous)
if __name__=='__main__':main()
