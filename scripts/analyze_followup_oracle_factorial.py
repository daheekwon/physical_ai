"""Separate acceptance and executed-jerk ranking with a fixed finite action bank."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.run_followup_dinowm_selection import make_frame
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def sources(smoke):
    capture=BASE/'dinowm_interventions';manifest=read(capture/'manifest.json')
    return capture,[(env,split,resolve(name)) for env,splits in manifest['states'].items() for split,names in splits.items() for name in (names[:1] if smoke else names)]

def choose(g,method,rule,cell,cfg):
    active=g.norm.to_numpy()>cfg['minimum_norm'];approved=(g[method].to_numpy()<=rule['tau']) if rule['enabled'] else np.zeros(len(g),bool)
    if cell in ['C','D']:approved=g.dense_full.to_numpy()<=1
    allowed=active&approved;actual=cell in ['B','D'];gain=g.jerk_gain.to_numpy() if actual else g.predicted_gain.to_numpy();ids=np.flatnonzero(allowed&(gain>(0 if actual else cfg['minimum_predicted_gain'])))
    rank=-gain if actual else g.predicted_cost.to_numpy();i=int(ids[np.lexsort((g.candidate.to_numpy()[ids],g.norm.to_numpy()[ids],rank[ids]))[0]]) if len(ids) else None;r=g.iloc[i] if i is not None else None
    useful=active&(g.dense_full.to_numpy()<=1)&(g.jerk_gain.to_numpy()>=cfg['meaningful_gain'])
    return {'state_id':g.state_id.iloc[0],'initial_pose_family':int(g.initial_pose_family.iloc[0]),'phase':g.phase.iloc[0],'candidate':int(r.candidate) if r is not None else 0,'active':r is not None,'violation':bool(r is not None and r.dense_full>1),'joint':bool(r is not None and r.dense_full<=1 and r.jerk_gain>=cfg['meaningful_gain']),'jerk_gain':float(r.jerk_gain) if r is not None else 0.,'object_error':float(r.dense_full) if r is not None else 0.,'useful_candidate_exists':bool(useful.any()),'approved_useful_exists':bool((useful&approved).any()),'approved_count':int((active&approved).sum())}

def main(smoke):
    cfg=read(BASE/'dinowm_selection/protocol.json')['config'];fcfg=read(resolve('configs/followup_oracle_factorial.json'));capture,srcs=sources(smoke);out=BASE/('oracle_factorial_smoke' if smoke else 'oracle_factorial');out.mkdir(exist_ok=True);parent=read(BASE/'dinowm_selection/pipeline_complete.json');assert parent['raw_audit_sha256']==sha256(BASE/'dinowm_selection/raw_audit.json');rules={(r['environment'],r['method']):r['rule'] for r in read(BASE/'dinowm_selection/rules.json')['rules']};old=pd.read_csv(BASE/'dinowm_selection/selected.csv');files=[resolve(p) for p in ['configs/followup_oracle_factorial.json','scripts/analyze_followup_oracle_factorial.py','scripts/run_followup_dinowm_selection.py','scripts/output_replacement_core.py','scripts/analyze_followup_response_selection.py']]+[BASE/'dinowm_selection'/p for p in ['protocol.json','pipeline_complete.json','raw_audit.json','rules.json','selected.csv']];inputs={str(p.relative_to(resolve('.'))):sha256(p) for p in files};rows=[];candidate_rows=0
    for env,split,src in srcs:
        g,paths=make_frame(env,split,src,capture,cfg)
        for p in paths:inputs[str(p.relative_to(resolve('.')))]=sha256(p)
        candidate_rows+=len(g)
        for method in cfg['methods']:
            for cell in fcfg['cells']:
                r=choose(g,method,rules[(env,method)],cell,cfg);rows.append({'environment':env,'split':split,'method':method,'cell':cell,**r})
                if cell=='A':
                    oldrow=old[(old.environment==env)&(old.split==split)&(old.method==method)&(old.state_id==r['state_id'])].iloc[0]
                    assert r['candidate']==int(oldrow.candidate) and r['joint']==oldrow.joint and r['violation']==oldrow.violation
        print('FACTORIAL',env,split,src.stem,flush=True)
    f=pd.DataFrame(rows);f.to_csv(out/'selected.csv',index=False);summary=[];paired=[]
    for (env,split,method,cell),g in f.groupby(['environment','split','method','cell']):
        summary.append({'environment':env,'split':split,'method':method,'cell':cell,'states':len(g),**{k:mean_ci(g,k) for k in ['active','joint','jerk_gain','useful_candidate_exists','approved_useful_exists']},'selected_risk':mean_ci(g[g.active],'violation')})
    for (env,split,method),g in f.groupby(['environment','split','method']):
        for later,earlier in [('B','A'),('C','A'),('D','C'),('D','B')]:
            x=g[g.cell==later].merge(g[g.cell==earlier],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
            for key in ['joint','active','jerk_gain']:
                x['difference']=x[key].astype(float)-x[key+'_base'].astype(float);paired.append({'environment':env,'split':split,'method':method,'later':later,'earlier':earlier,'metric':key,**mean_ci(x,'difference')})
    protocol={'smoke':smoke,'config':fcfg,'parent_config':cfg,'inputs':inputs,'sources':[[env,split,str(p.relative_to(resolve('.')))] for env,split,p in srcs]};write_json(out/'protocol.json',protocol);write_json(out/'analysis.json',{'smoke':smoke,'states':len(srcs),'candidate_rows':candidate_rows,'selection_rows':len(f),'summary':summary,'paired':paired,'scope':fcfg['scope']});write_json(out/'execution_complete.json',{'smoke':smoke,'states':len(srcs),'protocol_sha256':sha256(out/'protocol.json'),'analysis_sha256':sha256(out/'analysis.json'),'selected_sha256':sha256(out/'selected.csv')});print('FACTORIAL EXECUTED',len(srcs),candidate_rows,len(f),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();main(args.smoke)
