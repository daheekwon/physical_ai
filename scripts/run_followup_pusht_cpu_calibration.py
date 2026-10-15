"""Refresh only native CPU LeWM scores and calibration before fresh expert test."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from zj_wam.io import resolve,sha256,write_json
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.model_audit import predict_candidates
from scripts.run_followup_dinowm_selection import make_frame
from scripts.output_replacement_core import fit,selections
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def refreshed(src,out,cfg):
    f,_=make_frame('pusht',read(src)['split'],src,BASE/'dinowm_interventions',cfg)
    with np.load(out/'scores'/src.with_suffix('.npz').name) as a:
        mapping={int(k):i for i,k in enumerate(a['candidate_ids'])};ids=[mapping[int(k)] for k in f.candidate];f['lewm_global']=a['scores'][ids].astype(np.float64)
    return f

def main(smoke):
    cfg=read(BASE/'dinowm_selection/protocol.json')['config'];main=read(resolve('configs/followup_v1.json'));capture=read(BASE/'dinowm_interventions/manifest.json')['states']['pusht'];states=[resolve(n) for split in ['development','calibration'] for n in (capture[split][:1] if smoke else capture[split])];out=BASE/('pusht_cpu_calibration_smoke' if smoke else 'pusht_cpu_calibration');(out/'scores').mkdir(parents=True,exist_ok=True);files=[resolve(p) for p in ['scripts/run_followup_pusht_cpu_calibration.py','scripts/run_followup_dinowm_selection.py','scripts/output_replacement_core.py','zj_wam/model_audit.py','zj_wam/adapters.py','configs/followup_v1.json']]+[BASE/'dinowm_selection'/p for p in ['protocol.json','raw_audit.json','pipeline_complete.json','rules.json','selected.csv']]+[resolve(main['checkpoint'])/'weights.pt'];frames={}
    for src in states:
        f,paths=make_frame('pusht',read(src)['split'],src,BASE/'dinowm_interventions',cfg);frames[src.stem]=f;files.extend(paths)
    protocol={'smoke':smoke,'config':cfg,'sources':[str(p.relative_to(resolve('.'))) for p in states],'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'scope':'Same210oldexpert development/calibration families/candidates; refresh nativeCPU LeWM scores, recalibrate LeWM only. Controller/contact/nativeCPU DINO rules stay parent-locked. No newfamily outcomes, policydata, or computationcost claim. Smoke2states only tests execution/audit, does not provide finalrules.'}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write_json(out/'protocol.json',protocol)
    torch.set_num_threads(2);data=ExpertDataset(main['dataset']);model=FrozenLeWM(main['checkpoint'],data.scaler,'cpu')
    for src in states:
        dest=out/'scores'/src.name
        if dest.exists():assert read(dest)['protocol_sha256']==sha256(out/'protocol.json');continue
        start=time.perf_counter();f=frames[src.stem];ids=np.array(sorted(set([0]+f.candidate.to_list())),int)
        with np.load(src.with_suffix('.npz')) as a:actions=a['actions'][ids];images=a['history_images'];past=a['past_actions']
        _,context=model.predictor(images,past)
        with torch.no_grad():z=predict_candidates(model,context['embedding'],context['past'],actions,batch_size=64)[:,0]
        scores=np.linalg.norm(z-z[:1],axis=1);old=f.lewm_global.to_numpy();np.savez_compressed(dest.with_suffix('.npz'),candidate_ids=ids,latents=z,scores=scores);mapping={int(k):i for i,k in enumerate(ids)};error=float(abs(scores[[mapping[int(k)] for k in f.candidate]]-old).max());write_json(dest,{'state_id':src.stem,'protocol_sha256':sha256(out/'protocol.json'),'data_sha256':sha256(dest.with_suffix('.npz')),'max_score_difference_from_parent':error,'candidates':len(ids),'seconds':time.perf_counter()-start});print('CPU CAL SCORE',src.stem,len(ids),error,flush=True)
    f=pd.concat([refreshed(p,out,cfg) for p in states],ignore_index=True);f.to_csv(out/'candidates.csv',index=False);rule,curves=fit(f[f.split=='calibration'],'lewm_global',cfg,'pusht');write_json(out/'rule.json',rule);pd.DataFrame(curves).to_csv(out/'calibration_curves.csv',index=False);rules={r['method']:r['rule'] for r in read(BASE/'dinowm_selection/rules.json')['rules'] if r['environment']=='pusht'};rules['lewm_global']=rule;chosen=[]
    for method in cfg['methods']:
        for split,g in f.groupby('split'):
            x=selections(g,method,rules[method],cfg);x['method']=method;x['split']=split;chosen.append(x)
    chosen=pd.concat(chosen,ignore_index=True);chosen.to_csv(out/'selected.csv',index=False);write_json(out/'rules.json',{'rules':rules});summary=[]
    for (split,method),g in chosen.groupby(['split','method']):summary.append({'split':split,'method':method,'states':len(g),**{k:mean_ci(g,k) for k in ['joint','active','jerk_gain']},'selected_risk':mean_ci(g[g.active],'violation')})
    write_json(out/'analysis.json',{'smoke':smoke,'states':len(states),'candidate_rows':len(f),'selections':len(chosen),'summary':summary,'scope':protocol['scope']});write_json(out/'execution_complete.json',{'smoke':smoke,'states':len(states),'protocol_sha256':sha256(out/'protocol.json'),'outputs':{p:sha256(out/p) for p in ['analysis.json','candidates.csv','selected.csv','calibration_curves.csv','rule.json','rules.json']}});print('CPU CAL EXECUTED',len(states),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();main(args.smoke)
