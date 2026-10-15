"""Finite physical tests for low-response/jerk-gradient overlap on development."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset
from zj_wam.followup import DenseSimulator,dense_metrics
from scripts.direction_utility_core import generate
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=read(resolve('configs/followup_direction_utility.json'));maincfg=read(resolve('configs/followup_v1.json'));out=BASE/('direction_utility_smoke' if args.smoke else 'direction_utility');out.mkdir(exist_ok=True);paths=sorted((BASE/'interventions').glob('development*_h05.json'));assert len(paths)==105
    sources=[resolve(p) for p in ['configs/followup_direction_utility.json','configs/followup_v1.json','scripts/direction_utility_core.py','scripts/run_followup_direction_utility.py','zj_wam/followup.py','zj_wam/adapters.py','zj_wam/diagnostics.py']]+[BASE/'interventions/raw_audit.json',BASE/'selection/selector_rules.json',BASE/'protocol/manifest.json'];manifest={'smoke':args.smoke,'config':cfg,'main':maincfg,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in sources},'parents':{str(p.with_suffix(ext).relative_to(resolve('.'))):sha256(p.with_suffix(ext)) for p in paths for ext in ['.json','.npz']}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
    else:write_json(out/'manifest.json',manifest)
    data=ExpertDataset(maincfg['dataset']);rule=read(BASE/'selection/selector_rules.json')['rules']['controller_pusher']
    for path in paths[:1] if args.smoke else paths:
        meta=read(path);dest=out/path.name
        if dest.exists():assert read(dest)['manifest_sha256']==sha256(out/'manifest.json');continue
        started=time.perf_counter();aa,states=data.episode(meta['episode']);t=meta['timestep'];nominal=aa[t:t+5];suffix=aa[t+5:t+8]
        with np.load(path.with_suffix('.npz')) as z:parent={k:z[k] for k in ['root','physics','actions','jacobian','basis']}
        sim=DenseSimulator(aa,states,t,maincfg['seed']+meta['episode']);reference=sim.execute_dense(nominal,suffix);prefix=sim.prefix_physics.copy();root=reference['root_full_state'];np.testing.assert_array_equal(root,parent['root']);np.testing.assert_array_equal(reference['physics_trace'],parent['physics'][0]);np.testing.assert_array_equal(nominal,parent['actions'][0]);g=generate(nominal,suffix,root,prefix,data.scaler.scale_,parent['jacobian'],cfg,meta['episode'],t);np.testing.assert_array_equal(g['basis'],parent['basis'])
        norm=np.linalg.norm(g['coefficients'],axis=1);allowed=(norm>1e-6)&(norm<=rule['relative_radius']*np.sqrt(10)+1e-7)&(g['scores']<=rule['tau']);selected={}
        for name in cfg['methods']:
            ids=np.array([i for i,r in enumerate(g['records']) if r['method']==name]);ids=ids[allowed[ids]];chosen=int(ids[np.lexsort((norm[ids],g['costs'][ids]))[0]]) if len(ids) else -1
            if chosen>=0 and (g['cost0']-g['costs'][chosen])/max(g['cost0'],1e-8)<=1e-8:chosen=-1
            selected[name]=chosen+1
        rows=[];traces=[reference['physics_trace']];controls=[reference['control_trace']];cover=[[reference['coverage_t'],reference['coverage_suffix']]];contacts=[reference['physics_contacts']];maxcontroller=0.
        for i,(raw,record) in enumerate(zip(g['actions'],g['records'])):
            assert np.max(abs(raw))<=1+1e-6;result=sim.execute_dense(raw,suffix);metrics=dense_metrics(result,reference,maincfg['tolerances']);ce=float(np.max(abs(result['physics'][:,:2]-g['predicted'][i])));maxcontroller=max(maxcontroller,ce);assert ce<=1e-3
            rows.append({**{k:meta[k] for k in ['state_id','initial_pose_family','episode','timestep','split']},'index':i+1,**record,'dimension':g['bases'][record['method']].shape[1],'eta':g['eta'][record['method']],'gradient_valid':g['gradient_valid'],'norm':float(norm[i]),'controller_score':float(g['scores'][i]),'predicted_jerk':float(g['costs'][i]),'nominal_predicted_jerk':g['cost0'],'approved':bool(allowed[i]),'selected':selected[record['method']]==i+1,'nominal_contact':meta['nominal_contact'],'controller_max_error':ce,**metrics})
            traces.append(result['physics_trace']);controls.append(result['control_trace']);cover.append([result['coverage_t'],result['coverage_suffix']]);contacts.append(result['physics_contacts'])
        sim.close();arrays={'actions':np.array([nominal,*g['actions']]),'physics':np.array(traces),'control':np.array(controls),'coverage':np.array(cover),'contacts':np.array(contacts),'root':root,'suffix':suffix,'scale':data.scaler.scale_,'jacobian':parent['jacobian'],'coefficients':g['coefficients'],**{k:g[k] for k in ['gradient','relative_gradient','basis','base','response','design','j0','costs','scores','predicted','controller_eigenvalues','lewm_eigenvalues']}}
        for name,b in g['bases'].items():arrays['subspace_'+name]=b;arrays['direction_'+name]=g['directions'][name]
        np.savez_compressed(dest.with_suffix('.npz'),**arrays);pd.DataFrame(rows).to_csv(dest.with_suffix('.csv'),index=False);write_json(dest,{**{k:meta[k] for k in ['state_id','initial_pose_family','episode','timestep','split']},'candidates':len(rows),'eta':g['eta'],'gradient_valid':g['gradient_valid'],'controller_selected':selected,'common_bound':g['common_bound'],'max_controller_error':maxcontroller,'nominal_jerk':reference['physics_jerk'],'manifest_sha256':sha256(out/'manifest.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'csv_sha256':sha256(dest.with_suffix('.csv')),'seconds':time.perf_counter()-started});print(meta['state_id'],len(rows),round(time.perf_counter()-started,2),flush=True)
    for p,h in {**manifest['inputs'],**manifest['parents']}.items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'states':1 if args.smoke else 105,'candidates_per_state':60,'manifest_sha256':sha256(out/'manifest.json')})
if __name__=='__main__':main()
