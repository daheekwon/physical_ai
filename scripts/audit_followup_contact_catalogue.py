"""Independently replay contacts, reconstruct event selection and calibration resolution."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from zj_wam.policy_refinement import make_scene,full_state
from scripts.audit_retrained_dinowm_readouts import infer
BASE=resolve('artifacts/followup_v1');OUT=BASE/'contact_events_catalogue';read=lambda p:json.loads(p.read_text())

def main():
    manifest=read(OUT/'manifest.json');cfg=manifest['config'];res=read(OUT/'resolution.json');inputs={};families=0;cases=0
    for p,h in {**manifest['inputs'],**res['inputs']}.items():assert sha256(resolve(p))==h,p
    for seed in cfg['development_seeds']:
        a=dict(np.load(BASE/'dinowm_policy_nominal/development'/f'{seed}_nominal.npz'));path=OUT/f'{seed}_nominal_contacts.npz';z=dict(np.load(path));env=make_scene(seed);contacts=[];trace=[];count=[0];step=env.space.step
        def callback(arbiter,space,data):
            env._handle_collision(arbiter,space,data)
            bodies=[s.body for s in arbiter.shapes]
            if env.agent in bodies and env.block in bodies:count[0]+=len(arbiter.contact_point_set.points)
        env.space.on_collision(0,0,post_solve=callback)
        def record(dt):old=count[0];step(dt);trace.append(full_state(env));contacts.append(count[0]-old)
        env.space.step=record
        for action in a['actions']:env.step(action)
        env.space.step=step;env.close();np.testing.assert_array_equal(trace,z['physics']);np.testing.assert_array_equal(trace,a['physics']);np.testing.assert_array_equal(contacts,z['contacts']);binary=np.array(contacts).reshape(300,10).sum(axis=1)>0;selected=[r for r in manifest['cases'] if r['seed']==seed]
        for kind in cfg['event_types']:
            new=kind=='onset';edges=[e for e in np.flatnonzero(binary[1:]!=binary[:-1])+1 if 15<=e<=291 and np.all(binary[e-2:e]==(not new)) and np.all(binary[e:e+2]==new)];rows=[r for r in selected if r['event_type']==kind]
            if not edges:assert not rows;continue
            e=min(edges);assert {r['event_control'] for r in rows}=={e};roles={r['role']:r for r in rows};assert all(k in roles for k in cfg['window_offsets'])
            for role,offset in cfg['window_offsets'].items():assert roles[role]['step']==e+offset
            def motion(t):
                q=a['states'][t:t+6,:2];return float(np.sqrt(np.sum((q-q[0])**2)/6))
            target=motion(e-2)
            for role in cfg['stable_controls']:
                phase=role=='stable_contact';valid=[]
                for t in range(10,293):
                    if np.all(binary[t:t+8]==phase) and (t+8<=e-5 or t>=e+9):valid.append((abs(np.log((motion(t)+1e-6)/(target+1e-6))),abs(t-(e-2)),t))
                if valid:assert roles[role]['step']==min(valid)[2]
                else:assert role not in roles
            for r in rows:
                t=r['step'];np.testing.assert_allclose(r['nominal_motion_rms'],motion(t),atol=1e-12,rtol=0);assert r['state_id']==f'{seed}_t{t:03d}';cases+=1
        families+=1;inputs[str(path.relative_to(resolve('.')))]=sha256(path)
    expected={(r['seed'],r['step'],r['state_id']) for r in manifest['cases']};assert expected=={(r['seed'],r['step'],r['state_id']) for r in manifest['unique_states']}
    thresholds={};checks=0
    for rep in cfg['representations']:
        p=BASE/'dinowm_readouts/pusht'/f'{rep}__raw__ridge.npz';model=dict(np.load(p));values=[];weights=[];inputs[str(p.relative_to(resolve('.')))]=sha256(p)
        for p in sorted((BASE/'output_replacement_lewm/pusht/calibration').glob('*.json')):
            m=read(p);src=resolve(m['source']);raw=dict(np.load(src.with_suffix('.npz')));f=pd.read_csv(src.with_suffix('.csv'));ids=f.loc[f.actual_norm>cfg['minimum_norm'],'index'].to_numpy(int);assert set(ids)==set(np.flatnonzero(np.linalg.norm((raw['actions']-raw['actions'][:1]).reshape(len(raw['actions']),-1),axis=1)>1e-12));features=dict(np.load(resolve(m['feature_source'])));x=features['actual'] if rep=='lewm_frame' else np.c_[np.broadcast_to(features['history'][-2:].reshape(1,-1),(len(features['actual']),384)),features['actual']];decoded=infer(model,x.astype(float));stored=np.load(p.with_suffix('.npz'))[rep+'__raw__real'];np.testing.assert_allclose(decoded,stored,atol=1e-8,rtol=0);gt=raw['physics'][:,79,2:5];dp=decoded[:,:2]-decoded[:1,:2]-(gt[:,:2]-gt[:1,:2]);theta=np.arctan2(decoded[:,2],decoded[:,3]);dr=(theta-theta[0])-(gt[:,2]-gt[0,2]);vals=np.c_[np.sqrt((dp*dp).sum(1)),abs((dr+np.pi)%(2*np.pi)-np.pi)];values.extend(vals[ids]);weights.extend([1/len(ids)]*len(ids));checks+=1
        v=np.array(values);w=np.array(weights);thresholds[rep]={}
        for j,axis in enumerate(['position','rotation']):
            order=np.argsort(v[:,j],kind='stable');mass=np.cumsum(w[order]);index=np.searchsorted(mass,cfg['resolution_quantile']*mass[-1],side='left');thresholds[rep][axis]=float(v[order[index],j]);np.testing.assert_allclose(thresholds[rep][axis],res['thresholds'][rep][axis],atol=1e-8,rtol=0)
    assert families==12 and cases==len(manifest['cases']);write_json(OUT/'raw_audit.json',{'families':families,'cases':cases,'unique_states':len(expected),'all_nominal_contacts_and_physics_freshly_replayed':True,'calibration_state_readout_checks':checks,'resolution':thresholds,'manifest_sha256':sha256(OUT/'manifest.json'),'resolution_sha256':sha256(OUT/'resolution.json'),'inputs':inputs,'source_sha256':sha256(Path(__file__)),'helper_sha256':sha256(resolve('scripts/audit_retrained_dinowm_readouts.py'))});print('AUDITED EVENT CATALOGUE',families,cases,len(expected),checks,flush=True)
if __name__=='__main__':main()
