"""Native DINO verifier calibration on identical non-WM candidate banks."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.output_replacement_core import fit,selections,physical
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())

def make_frame(env,split,src,capture,cfg):
    meta=read(src);parentcsv=src.with_suffix('.csv');f=pd.read_csv(parentcsv);sourcepaths=[src,parentcsv,src.with_suffix('.npz')]
    with np.load(src.with_suffix('.npz'),allow_pickle=False) as z:raw=dict(z)
    tp=capture/env/split/src.with_suffix('.npz').name;tm=read(tp.with_suffix('.json'));assert tm['data_sha256']==sha256(tp);sourcepaths.extend([tp,tp.with_suffix('.json')])
    with np.load(tp,allow_pickle=False) as z:pred=z['predicted'].astype(np.float64)
    diff=pred-pred[:1];dino_rms=np.sqrt(np.mean(np.sum(diff**2,axis=2),axis=1));dino_mean=np.linalg.norm(diff.mean(axis=1),axis=1)
    truth=physical(env,raw)
    if env=='pusht':
        sp=BASE/'response_scores/f1'/src.with_suffix('.csv').name;sourcepaths.append(sp);f=pd.read_csv(sp);assert set(f.method.str.split('_').str[0])<={'optimizer','random'};ids=f['index'].to_numpy(int);radius=f.relative_radius.to_numpy();cost=f.predicted_jerk.to_numpy();nominal=f.nominal_predicted_jerk.to_numpy();norm=f.actual_norm.to_numpy();controller=f.controller_pusher_score.to_numpy();contact=f.vision_contact_score.to_numpy();lewm=f.lewm_global_score.to_numpy();phase='contact' if bool(f.nominal_contact.iloc[0]) else 'free';nonwm=np.ones(len(f),bool)
    else:
        ids=f.candidate.to_numpy(int);radius=f.radius.to_numpy();cost=f.predicted_jerk_cost.to_numpy();norm=f.norm.to_numpy();nominal=np.full(len(f),cost[np.flatnonzero(ids==0)[0]]);lewm=f.lewm_score.to_numpy();phase=meta['phase'];nonwm=~f.method.isin(['lewm_low','lewm_high']).to_numpy();motion=raw['predicted_robot']-raw['predicted_robot'][:1];pos=np.linalg.norm(motion[:,:,:3],axis=2);angle=abs(np.arctan2(np.sin(motion[:,:,3]),np.cos(motion[:,:,3])));controller=np.maximum(pos.max(1)/.002,angle.max(1)/np.deg2rad(2))[ids];coupled=bool(float(raw['root_gripper_contact'])>.1 or np.any(np.r_[raw['nominal_actions'][:,4],raw['suffix_actions'][:,4]]>.1));contact=np.maximum((pos+np.sqrt(3)*.02*angle).max(1)/.002,angle.max(1)/np.deg2rad(2))[ids] if coupled else np.zeros(len(f))
    valid=nonwm&(radius<=cfg['maximum_radius']+1e-12)&(norm>cfg['minimum_norm']);assert len(pred)==len(raw['actions'])
    np.testing.assert_allclose(truth['dense_full'][ids],f.object_error,atol=2e-5);np.testing.assert_allclose(truth['jerk_gain'][ids],f.jerk_gain,atol=1e-8)
    frame=pd.DataFrame({'environment':env,'split':split,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'phase':phase,'candidate':ids,'generator':f.method.to_numpy(),'radius':radius,'norm':norm,'predicted_cost':cost,'predicted_gain':(nominal-cost)/np.maximum(nominal,1e-12),'dense_full':truth['dense_full'][ids],'jerk_gain':truth['jerk_gain'][ids],'controller':controller,'contact':contact,'lewm_global':lewm,'dino_patch_rms':dino_rms[ids],'dino_patch_mean':dino_mean[ids]})
    # Keep a nominal row only when no active candidates exist, so state denominators persist.
    frame=frame[valid].copy()
    if frame.empty:
        frame=pd.DataFrame([{'environment':env,'split':split,'state_id':meta['state_id'],'initial_pose_family':meta['initial_pose_family'],'phase':phase,'candidate':0,'generator':'nominal_fallback','radius':0.,'norm':0.,'predicted_cost':float(nominal[0]),'predicted_gain':0.,'dense_full':0.,'jerk_gain':0.,**{m:0. for m in cfg['methods']}}])
    return frame,sourcepaths


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();cfg=read(resolve('configs/followup_dinowm_selection.json'));capture=BASE/('dinowm_interventions_smoke' if args.smoke else 'dinowm_interventions');out=BASE/('dinowm_selection_smoke' if args.smoke else 'dinowm_selection');audit=read(capture/'raw_audit.json');assert audit['states']==(4 if args.smoke else 402)
    if not args.smoke:assert read(capture/'pipeline_complete.json')['raw_audit_sha256']==sha256(capture/'raw_audit.json')
    out.mkdir(exist_ok=True);manifest=read(capture/'manifest.json');inputs={str(p.relative_to(resolve('.'))):sha256(p) for p in [resolve('configs/followup_dinowm_selection.json'),Path(__file__),resolve('scripts/output_replacement_core.py'),resolve('scripts/analyze_followup_response_selection.py'),capture/'manifest.json',capture/'raw_audit.json']};frames=[]
    for env,splits in manifest['states'].items():
        for split,names in splits.items():
            for name in names[:1] if args.smoke else names:
                frame,paths=make_frame(env,split,resolve(name),capture,cfg);frames.append(frame)
                for p in paths:inputs[str(p.relative_to(resolve('.')))]=sha256(p)
    frame=pd.concat(frames,ignore_index=True);states=frame.groupby(['environment','split']).state_id.nunique().to_dict();protocol={'smoke':args.smoke,'config':cfg,'inputs':inputs,'state_counts':{env+'/'+split:int(v) for (env,split),v in states.items()}}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write_json(out/'protocol.json',protocol)
    frame.to_csv(out/'candidates.csv',index=False);rules=[];curves=[];chosen=[]
    for env,g in frame.groupby('environment',sort=True):
        for method in cfg['methods']:
            rule,cc=fit(g[g.split=='calibration'],method,cfg,env);rules.append({'environment':env,'method':method,'rule':rule});curves.extend({'environment':env,'method':method,**r} for r in cc)
            for split in ['development','calibration']:
                rows=selections(g[g.split==split],method,rule,cfg);rows['environment']=env;rows['method']=method;rows['split']=split;chosen.append(rows)
    chosen=pd.concat(chosen,ignore_index=True);chosen.to_csv(out/'selected.csv',index=False);pd.DataFrame(curves).to_csv(out/'calibration_curves.csv',index=False);write_json(out/'rules.json',{'rules':rules});summary=[];paired=[]
    for (env,split,method),g in chosen.groupby(['environment','split','method']):
        for phase in ['all']+sorted(g.phase.unique().tolist()):
            v=g if phase=='all' else g[g.phase==phase];a=v[v.active];summary.append({'environment':env,'split':split,'method':method,'phase':phase,'states':len(v),**{k:mean_ci(v,k) for k in ['active','joint','jerk_gain','predicted_gain','candidate_coverage']},'selected_risk':mean_ci(a,'violation')})
    for (env,split),g in chosen.groupby(['environment','split']):
        for method in ['dino_patch_rms','dino_patch_mean']:
            for baseline in ['controller','contact','lewm_global']:
                a=g[g.method==method];b=g[g.method==baseline];v=a.merge(b,on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
                for metric in ['joint','active','jerk_gain']:
                    v['difference']=v[metric].astype(float)-v[metric+'_base'].astype(float);paired.append({'environment':env,'split':split,'method':method,'baseline':baseline,'metric':metric,**mean_ci(v,'difference')})
    write_json(out/'analysis.json',{'smoke':args.smoke,'states':sum(states.values()),'candidate_rows':len(frame),'selected_rows':len(chosen),'summary':summary,'paired':paired,'scope':cfg['scope']})
    for p,h in inputs.items():assert sha256(resolve(p))==h,p
    write_json(out/'execution_complete.json',{'smoke':args.smoke,'states':sum(states.values()),'protocol_sha256':sha256(out/'protocol.json'),'files':{n:sha256(out/n) for n in ['candidates.csv','selected.csv','calibration_curves.csv','rules.json','analysis.json']},'scope':'Independent physical/native-score/calibration audit required.'});print('DINO SELECTION EXECUTED',sum(states.values()),len(frame),len(chosen),flush=True)

if __name__=='__main__':main()
