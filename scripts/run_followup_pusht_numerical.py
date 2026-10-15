"""Local integration sensitivity, preserving the original branch root and action clock."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,sha256,write_json
from zj_wam.policy_refinement import make_scene,full_state,coverage
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_pusht_numerical.json','scripts/run_followup_pusht_numerical.py','zj_wam/policy_refinement.py','external/stable-worldmodel/stable_worldmodel/envs/pusht/env.py']

def selected():
    cfg=read(resolve(SOURCES[0]));paths=sorted((BASE/'dinowm_policy_bank/development').glob('*.json'));out=[]
    for phase in cfg['phase_order']:
        seen=set();take=[]
        for p in paths:
            m=read(p)
            if m['phase']!=phase or m['seed'] in seen:continue
            seen.add(m['seed']);take.append(p)
            if len(take)==cfg['states_per_nominal_phase']:break
        assert len(take)==cfg['states_per_nominal_phase'];out.extend(take)
    return out

def rollout(seed,t,parent,action,rate):
    env=make_scene(seed)
    for x in parent['actions'][:t]:env.step(x)
    np.testing.assert_array_equal(full_state(env),parent['states'][t])
    assert env.control_hz==10 and env.k_p==100 and env.k_v==20
    env.dt=1/rate;assert int(1/(env.dt*env.control_hz))==rate//10
    trace=[];contacts=[];covers=[];counter=[0];original=env.space.step
    def collide(arbiter,space,data):
        env._handle_collision(arbiter,space,data)
        if {env.agent,env.block}=={shape.body for shape in arbiter.shapes}:counter[0]+=len(arbiter.contact_point_set.points)
    env.space.on_collision(0,0,post_solve=collide)
    def step(dt):
        before=counter[0];original(dt);trace.append(full_state(env));contacts.append(counter[0]-before)
    env.space.step=step
    for k,x in enumerate(np.r_[action,parent['actions'][t+5:t+8]]):
        env.step(x)
        if k in [4,7]:covers.append(coverage(env))
    env.space.step=original;env.close()
    assert len(trace)==rate*8//10
    return np.array(trace),np.array(contacts),np.array(covers)

def main(smoke):
    cfg=read(resolve(SOURCES[0]));paths=selected();paths=paths[:1] if smoke else paths
    out=BASE/('pusht_numerical_smoke' if smoke else 'pusht_numerical');out.mkdir(exist_ok=True)
    bank=BASE/'dinowm_policy_bank';a=read(bank/'raw_audit.json');assert a['states']==108 and read(bank/'pipeline_complete.json')['raw_audit_sha256']==sha256(bank/'raw_audit.json')
    files=[resolve(p) for p in SOURCES]+[bank/'manifest.json',bank/'raw_audit.json',bank/'pipeline_complete.json']
    for p in paths:
        files.extend([p,p.with_suffix('.npz'),resolve(read(p)['source']),resolve(read(p)['source']).with_suffix('.npz')])
    lock={'smoke':smoke,'config':cfg,'sources':[str(p.relative_to(resolve('.'))) for p in paths],'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files}}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==lock
    else:write_json(out/'protocol.json',lock)
    total=0
    for p in paths:
        m=read(p);a=dict(np.load(p.with_suffix('.npz')));parent=dict(np.load(resolve(m['source']).with_suffix('.npz')));dest=out/(p.stem+'.npz');raw={};start=time.time()
        for rate,rep in [(100,0),(100,1),(200,0),(400,0)]:
            trace=[];contacts=[];covers=[]
            for action in a['actions']:
                q,c,v=rollout(m['seed'],m['step'],parent,action,rate);trace.append(q);contacts.append(c);covers.append(v)
            key=f'{rate}_r{rep}';raw['physics_'+key]=np.array(trace);raw['contacts_'+key]=np.array(contacts);raw['coverage_'+key]=np.array(covers)
        for suffix in ['physics','contacts','coverage']:np.testing.assert_array_equal(raw[suffix+'_100_r0'],raw[suffix+'_100_r1'])
        np.testing.assert_array_equal(raw['physics_100_r0'],a['physics'][:,30:]);np.testing.assert_array_equal(raw['contacts_100_r0'],a['contacts']);np.testing.assert_array_equal(raw['coverage_100_r0'],a['coverage'])
        np.savez_compressed(dest,**raw);write_json(dest.with_suffix('.json'),{'source':str(p.relative_to(resolve('.'))),'seed':m['seed'],'state_id':m['state_id'],'phase':m['phase'],'candidates':len(a['actions']),'data_sha256':sha256(dest),'protocol_sha256':sha256(out/'protocol.json')});total+=len(a['actions']);print('NUMERICAL',p.stem,len(a['actions']),round(time.time()-start,2),flush=True)
    for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'states':len(paths),'candidates':total,'fresh_physical_branches':total*4,'smoke':smoke,'protocol_sha256':sha256(out/'protocol.json')})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.smoke)
