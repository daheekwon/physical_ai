"""Frozen nominal policy chunks with polynomial terminal-state constraints."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256
from zj_wam.adapters import ExpertDataset
from zj_wam.policy_refinement import make_scene,full_state,coverage
from zj_wam.followup import optimize_controller
from zj_wam.diagnostics import polynomial_basis
from zj_wam.correction_families import spline_basis,terminal_constrained
BASE=resolve('artifacts/followup_v1');SOURCE=BASE/'policy_refinement/development'
read=lambda p:json.loads(p.read_text())


def prepare(smoke):
    cfg=read(resolve('configs/followup_policy_terminal.json')); main=read(resolve('configs/followup_v1.json'))
    out=BASE/('policy_terminal_smoke' if smoke else 'policy_terminal');out.mkdir(exist_ok=True)
    paths=[resolve(p) for p in ['configs/followup_policy_terminal.json','configs/followup_v1.json','scripts/run_followup_policy_terminal.py','scripts/analyze_followup_policy_terminal.py','scripts/analyze_followup_policy_final.py','scripts/analyze_followup_response_selection.py','zj_wam/policy_refinement.py','zj_wam/correction_families.py','zj_wam/followup.py','zj_wam/diagnostics.py','zj_wam/adapters.py']]
    paths += [SOURCE/'raw_audit.json',SOURCE/'manifest.json']
    paths += [SOURCE/f'{seed}_nominal{ext}' for seed in cfg['development_seeds'] for ext in ['.json','.npz']]
    m={'config':cfg,'main':main,'smoke':smoke,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==m
    else:write_json(out/'manifest.json',m)
    return cfg,main,out,m


def replay(seed,actions):
    env=make_scene(seed);initial=full_state(env);trace=[];cov=[];old=env.space.step
    def step(dt):old(dt);trace.append(full_state(env))
    env.space.step=step
    for action in actions:env.step(action);cov.append(coverage(env))
    env.space.step=old;env.close()
    return initial,np.asarray(trace),np.asarray(cov)


def process(seed,h,cfg,main,out,scale):
    path=out/f'{seed}_h{h:02d}.json'
    if path.exists():assert read(path)['manifest_sha256']==sha256(out/'manifest.json');return
    start=time.perf_counter();t=cfg['correction_step'];chunk=cfg['chunk_actions']
    with np.load(SOURCE/f'{seed}_nominal.npz') as a:old={k:a[k].copy() for k in ['actions','physics','coverage','initial_state','states']}
    assert len(old['actions'])==cfg['max_actions'];initial,ref,refcov=replay(seed,old['actions'])
    np.testing.assert_array_equal(initial,old['initial_state']);np.testing.assert_array_equal(ref,old['physics']);np.testing.assert_array_equal(refcov,old['coverage'])
    root=old['states'][t];prefix=ref[t*10-30:t*10];nominal=old['actions'][t:t+h];knownsuffix=old['actions'][t+h:t+chunk]
    bases={'quintic':polynomial_basis(h),'spline':spline_basis(h)}
    for name in list(bases):bases[name+'_terminal'],_=terminal_constrained(bases[name],nominal,knownsuffix,root,prefix,scale)
    traces=[ref];coverages=[refcov];allactions=[old['actions']];rows=[]
    for name in cfg['families']:
        basis=bases[name]
        for radius in cfg['relative_radii']:
            if basis.shape[1]:chosen,info=optimize_controller(nominal,knownsuffix,root,prefix,scale,radius*np.sqrt(2*h),basis=basis)
            else:chosen=nominal.copy();info={'solver_success':True,'iterations':0,'feasible':True}
            delivered=old['actions'].copy();delivered[t:t+h]=chosen
            ini,physics,cov=replay(seed,delivered);np.testing.assert_array_equal(ini,initial);np.testing.assert_array_equal(physics[:t*10],ref[:t*10])
            d=physics-ref;j=(t+h)*10-1;q=float(np.linalg.norm(d[j,:2]));v=float(np.linalg.norm(d[j,5:7]))
            if name.endswith('_terminal'):assert max(q,v)<=cfg['terminal_tolerance'],(seed,h,name,q,v)
            norm=float(np.linalg.norm((chosen-nominal)/scale))
            rows.append({'seed':seed,'horizon':h,'method':name,'radius':radius,'index':len(traces),'dimension':basis.shape[1],'norm':norm,'terminal_q':q,'terminal_v':v,'solver_success':info['solver_success'],'solver_iterations':info['iterations']})
            traces.append(physics);coverages.append(cov);allactions.append(delivered)
    np.savez_compressed(path.with_suffix('.npz'),initial=initial,physics=traces,coverage=coverages,actions=allactions,scale=scale,**{'basis_'+k:v for k,v in bases.items()})
    pd.DataFrame(rows).to_csv(path.with_suffix('.csv'),index=False)
    write_json(path,{'seed':seed,'horizon':h,'candidates':len(rows),'dimensions':{k:v.shape[1] for k,v in bases.items()},'seconds':time.perf_counter()-start,'manifest_sha256':sha256(out/'manifest.json')})
    print(seed,h,len(rows),'branches',round(time.perf_counter()-start,2),'seconds',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args();cfg,main,out,m=prepare(args.smoke)
    data=ExpertDataset(main['dataset'])
    for seed in cfg['development_seeds'][:1] if args.smoke else cfg['development_seeds']:
        for h in cfg['horizons']:process(seed,h,cfg,main,out,data.scaler.scale_)
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'seed_horizons':2 if args.smoke else 24,'manifest_sha256':sha256(out/'manifest.json')})
if __name__=='__main__':main()
