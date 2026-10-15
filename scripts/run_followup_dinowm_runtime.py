"""Warmed randomized fresh CPU runtime on the audited common verifier bank."""
import argparse,json,os,platform,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits,threadpool_info
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from scripts.run_followup_cube import Model
from scripts.model_runtime_imports import load_dino
from scripts.dinowm_runtime_core import run
from scripts.dinowm_runtime_context import load_context
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_dinowm_runtime.json','scripts/run_followup_dinowm_runtime.py','scripts/dinowm_runtime_context.py','scripts/dinowm_runtime_core.py','scripts/model_runtime_imports.py','scripts/retrained_dinowm_core.py','scripts/run_followup_cube.py','configs/followup_v1.json','configs/followup_cube.json','configs/followup_response_models.json','zj_wam/followup.py','zj_wam/cube_experiment.py','zj_wam/diagnostics.py','zj_wam/response_models.py','zj_wam/repeated_refinement.py','zj_wam/model_audit.py','zj_wam/adapters.py']

def main(smoke):
    rcfg=read(resolve('configs/followup_dinowm_runtime.json'));torch.set_num_threads(rcfg['torch_threads']);cfg0=read(resolve('configs/followup_v1.json'));cfg1=read(resolve('configs/followup_cube.json'));response=read(resolve('configs/followup_response_models.json'));data=ExpertDataset(cfg0['dataset']);out=BASE/('dinowm_runtime_warmed_smoke' if smoke else 'dinowm_runtime');out.mkdir(exist_ok=True)
    capture=BASE/('dinowm_interventions_smoke' if smoke else 'dinowm_interventions');manifest=read(capture/'manifest.json');rules={(r['environment'],r['method']):r['rule'] for r in read(BASE/'dinowm_selection/rules.json')['rules']};parent=pd.read_csv(BASE/'dinowm_selection/candidates.csv');oldchoices=pd.read_csv(BASE/'dinowm_selection/selected.csv')
    files=[resolve(p) for p in SOURCES]+[capture/'manifest.json',BASE/'dinowm_selection/rules.json',BASE/'dinowm_selection/raw_audit.json',BASE/'dinowm_selection/candidates.csv',BASE/'dinowm_selection/selected.csv',BASE/'cube/manifest.json']
    files += [resolve(cfg0['checkpoint'])/'weights.pt',resolve(cfg1['checkpoint'])/'weights.pt']
    files += sorted((BASE/'protocol/dinowm_retrained_inventory').glob('*.json'))
    protocol={'smoke':smoke,'config':rcfg,'states':manifest['states'],'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'platform':platform.platform(),'python':sys.version,'threadpools':threadpool_info(),'scope':rcfg['scope']}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write_json(out/'protocol.json',protocol)
    write_json(out/'process.json',{'pid':os.getpid(),'started_unix':time.time()});count=0
    for env,splits in manifest['states'].items():
        cfg=cfg0 if env=='pusht' else {**cfg1,'device':'cpu'};norm=read(BASE/'cube/manifest.json')['normalization'];mean=data.scaler.mean_ if env=='pusht' else np.array(norm['mean']);scale=data.scaler.scale_ if env=='pusht' else np.array(norm['scale']);lewm=FrozenLeWM(cfg0['checkpoint'],data.scaler,'cpu') if env=='pusht' else Model(cfg,mean,scale);dino=load_dino(env)
        def execute(c,method):return run(env,c,cfg,method,rules[(env,method)],lewm if method=='lewm_global' else None,dino if method.startswith('dino') else None,response)
        _,warm,_,_=load_context(env,resolve(splits['calibration'][0]),data,mean,scale)
        for method in rcfg['methods']:
            for _ in range(rcfg['warmup_per_method_per_environment']):execute(warm,method)
        for split in ['calibration','development']:
            for name in splits[split]:
                src=resolve(name);folder=out/env/split;folder.mkdir(parents=True,exist_ok=True);dest=folder/src.name
                if dest.exists():assert read(dest)['protocol_sha256']==sha256(out/'protocol.json');count+=1;continue
                meta,c,lookup,raw=load_context(env,src,data,mean,scale);stateparent=parent[(parent.environment==env)&(parent.state_id==meta['state_id'])];order=[(m,r) for r in range(rcfg['repetitions']) for m in rcfg['methods']];rng=np.random.default_rng(np.random.SeedSequence([rcfg['seed'],meta['episode'],meta['timestep']]));rng.shuffle(order);rows=[];saved={};first={};errors={};solvers={}
                for method,repetition in order:
                    result=execute(c,method);ids=np.array([lookup[tuple(label)] for label in result['labels']]);error=float(abs(result['actions']-raw['actions'][ids]).max());assert error<=rcfg['action_reproduction_atol'],(env,meta['state_id'],method,error,'candidate reproduction failed');indexmap={int(k):i for i,k in enumerate(ids)};expected=set(stateparent.candidate);assert expected<=set(ids);selected=int(ids[result['selected']]);old=int(oldchoices[(oldchoices.environment==env)&(oldchoices.state_id==meta['state_id'])&(oldchoices.method==method)].candidate.iloc[0]);scoreerror=float(abs(result['scores'][[indexmap[int(k)] for k in stateparent.candidate]]-stateparent[method].to_numpy()).max());errors[method]={'actions':error,'score_vs_parent':scoreerror}
                    assert np.all(np.isfinite(result['costs'])) and not np.any(np.isnan(result['scores']))
                    if method in first:
                        assert selected==first[method]
                        for field in ['actions','scores','costs','norms']:np.testing.assert_array_equal(result[field],saved[method+'_'+field])
                    else:
                        first[method]=selected;solvers[method]=result['solver'];saved[method+'_candidate_ids']=ids
                        for field in ['actions','scores','costs','norms']:saved[method+'_'+field]=result[field]
                    row={'environment':env,'split':split,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'phase':stateparent.phase.iloc[0],'method':method,'repetition':repetition,'candidate':selected,'parent_candidate':old,'parent_choice_match':selected==old,'candidate_count':len(ids),'optimizer_calls':len(result['solver']),'solver_failures':sum(not x['success'] for x in result['solver']),**{k:v for k,v in result.items() if k.endswith('_seconds') or k.endswith('_calls')}};rows.append(row)
                pd.DataFrame(rows).to_csv(dest.with_suffix('.csv'),index=False);np.savez_compressed(dest.with_suffix('.npz'),**saved);write_json(dest,{'environment':env,'split':split,'state_id':meta['state_id'],'protocol_sha256':sha256(out/'protocol.json'),'inputs':{str(src.with_suffix(e).relative_to(resolve('.'))):sha256(src.with_suffix(e)) for e in ['.json','.csv','.npz']},'errors':errors,'solvers':solvers,'timing_rows':len(rows),'load_average':os.getloadavg(),'threadpools':threadpool_info()});count+=1;print('RUNTIME',env,split,meta['state_id'],count,'rows',len(rows),flush=True)
        del lewm,dino
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    assert count==(4 if smoke else 402)
    write_json(out/'execution_complete.json',{'states':count,'timings':count*len(rcfg['methods'])*rcfg['repetitions'],'protocol_sha256':sha256(out/'protocol.json'),'unix':time.time(),'scope':'Timing collection complete; independent action, scores, choices, physical outcome and family statistics audit required.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=read(resolve('configs/followup_dinowm_runtime.json'))
    with threadpool_limits(limits=cfg['blas_threads'],user_api='blas'):main(args.smoke)
