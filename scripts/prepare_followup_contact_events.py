"""Replay nominal contact history and lock event windows before interventions."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,sha256,write_json
from zj_wam.policy_refinement import make_scene,full_state
from scripts.contact_event_core import windows,resolution
BASE=resolve('artifacts/followup_v1');OUT=BASE/'contact_events_catalogue';read=lambda p:json.loads(p.read_text())

def main():
    OUT.mkdir(exist_ok=True);cfg=read(resolve('configs/followup_contact_events.json'));parent=BASE/'dinowm_policy_nominal';audit=read(parent/'raw_audit.json');assert audit['episodes']==36 and audit['all_physics_and_RGB_replayed_exact'];assert read(parent/'pipeline_complete.json')['raw_audit_sha256']==sha256(parent/'raw_audit.json');cases=[];missing=[];inputs={}
    for seed in cfg['development_seeds']:
        src=parent/'development'/f'{seed}_nominal.npz';a=dict(np.load(src));env=make_scene(seed);physics=[];contact=[];counter=[0];old=env.space.step
        def collide(arbiter,space,data):
            env._handle_collision(arbiter,space,data)
            if {s.body for s in arbiter.shapes}=={env.agent,env.block}:counter[0]+=len(arbiter.contact_point_set.points)
        env.space.on_collision(0,0,post_solve=collide)
        def step(dt):before=counter[0];old(dt);physics.append(full_state(env));contact.append(counter[0]-before)
        env.space.step=step
        for command in a['actions']:env.step(command)
        env.space.step=old;env.close();np.testing.assert_array_equal(physics,a['physics']);p=OUT/f'{seed}_nominal_contacts.npz';np.savez_compressed(p,contacts=contact,physics=physics);rows,absent=windows(seed,a,np.array(contact),cfg);cases.extend(rows);missing.extend(absent)
        for q in [src,src.with_suffix('.json'),p]:inputs[str(q.relative_to(resolve('.')))]=sha256(q)
        print('EVENT CATALOGUE',seed,len(rows),len(absent),flush=True)
    unique={x['state_id']:{'seed':x['seed'],'step':x['step'],'state_id':x['state_id']} for x in cases};unique=[unique[k] for k in sorted(unique)];assert unique
    res=resolution(cfg);write_json(OUT/'resolution.json',res)
    files=[resolve(p) for p in ['configs/followup_contact_events.json','scripts/contact_event_core.py','scripts/prepare_followup_contact_events.py','zj_wam/policy_refinement.py']]+[parent/'manifest.json',parent/'raw_audit.json',parent/'pipeline_complete.json',OUT/'resolution.json'];inputs.update({str(p.relative_to(resolve('.'))):sha256(p) for p in files});write_json(OUT/'manifest.json',{'config':cfg,'cases':cases,'unique_states':unique,'missing':missing,'families':len(cfg['development_seeds']),'inputs':inputs,'scope':cfg['scope']});print('EVENT WINDOWS',len(cases),len(unique),'missing',len(missing),res['thresholds'],flush=True)
if __name__=='__main__':main()
