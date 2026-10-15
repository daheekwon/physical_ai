"""Stream frozen readout diagnostics; publish only after full collection audit."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.retrained_dinowm_effect_readout_core import decode_all
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');READOUT=BASE/'dinowm_readouts';read=lambda p:json.loads(p.read_text())


def prepare(smoke):
    capture=BASE/('dinowm_interventions_smoke' if smoke else 'dinowm_interventions');out=BASE/('dinowm_effect_readouts_smoke' if smoke else 'dinowm_effect_readouts');out.mkdir(exist_ok=True);cfg=read(resolve('configs/followup_dinowm_effect_readouts.json'));m=read(capture/'manifest.json');paths=[resolve(p) for p in ['configs/followup_dinowm_effect_readouts.json','scripts/retrained_dinowm_effect_readout_core.py','scripts/analyze_retrained_dinowm_effect_readouts.py','scripts/retrained_dinowm_readout_core.py','zj_wam/cube_readouts.py','scripts/analyze_followup_response_selection.py']]+[capture/'manifest.json',READOUT/'selection.json',READOUT/'raw_audit.json',READOUT/'projection.npy'];models={}
    for env in cfg['targets']:
        models[env]={}
        for rep in cfg['representations']:
            for kind in cfg['targets'][env]:
                for learner in (cfg['cube_raw_learners'] if env=='ogb-cube' and kind=='raw' else cfg['learners']):
                    key=rep+'__'+kind+'__'+learner;p=READOUT/env/(key+'.npz');paths.append(p)
                    with np.load(p,allow_pickle=False) as a:models[env][key]=dict(a)
        paths.append(resolve('data/checkpoints/dinowm-retrained-fafraob')/env/'swm_h5_norm_stats.json')
    record={'smoke':smoke,'config':cfg,'collection_manifest_sha256':sha256(capture/'manifest.json'),'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==record
    else:write_json(out/'protocol.json',record)
    return out,capture,cfg,m,models,record


def process(env,split,src,capture,out,cfg,models,q):
    folder=out/env/split;folder.mkdir(parents=True,exist_ok=True);dest=folder/src.name;tokenspath=capture/env/split/src.with_suffix('.npz').name;tokenmeta=read(tokenspath.with_suffix('.json'))
    assert tokenmeta['data_sha256']==sha256(tokenspath) and tokenmeta['manifest_sha256']==sha256(capture/'manifest.json')
    if dest.exists():
        r=read(dest);assert r['protocol_sha256']==sha256(out/'protocol.json') and r['tokens_sha256']==sha256(tokenspath)
        for name,h in r['files'].items():assert sha256(folder/name)==h
        return
    started=time.perf_counter();old=pd.read_csv(src.with_suffix('.csv'));parentmeta=read(src);scale=np.array(read(resolve('data/checkpoints/dinowm-retrained-fafraob')/env/'swm_h5_norm_stats.json')['action_std']);ids=old['index' if env=='pusht' else 'candidate'].to_numpy(int);norm=old['actual_norm' if env=='pusht' else 'norm'].to_numpy();frames=[];arrays={}
    with np.load(tokenspath,allow_pickle=False) as z,np.load(src.with_suffix('.npz'),allow_pickle=False) as p:
        tokens={k:z[k] for k in ['predicted','actual','history']};parent={k:p[k] for k in (['physics','actions'] if env=='pusht' else ['candidate_object_pose','actions'])}
        for key,model in models[env].items():
            rep,kind,learner=key.split('__');methods,saved=decode_all(env,rep,kind,model,tokens,q,parent,cfg,scale);arrays.update({key+'__'+name:value for name,value in saved.items()})
            posmag=np.linalg.norm(saved['true_dp'],axis=1);rotmag=np.linalg.norm(saved['true_dr'],axis=1)
            for method,values in methods.items():
                common={'environment':env,'split':split,'state_id':parentmeta['state_id'],'initial_pose_family':parentmeta['initial_pose_family'],'representation':rep,'target':kind,'readout':learner,'method':method,'candidate':ids,'active':norm>1e-6,'physical_E':old.object_error.to_numpy(),'jerk_gain':old.jerk_gain.to_numpy(),'true_position_magnitude':posmag[ids],'true_rotation_magnitude':rotmag[ids],'wrong_index':saved['wrong_index'][ids],'wrong_norm_ratio':saved['wrong_norm_ratio'][ids],'wrong_cosine':saved['wrong_cosine'][ids]}
                if env=='pusht':common['phase']=np.where(old.nominal_contact,'contact','free')
                else:common['phase']=parentmeta['phase']
                frames.append(pd.DataFrame({**common,**{k:v[ids] for k,v in values.items()}}))
    f=pd.concat(frames,ignore_index=True);csv=dest.with_suffix('.csv.gz');npz=dest.with_suffix('.npz');f.to_csv(csv,index=False);np.savez_compressed(npz,**arrays)
    write_json(dest,{'environment':env,'split':split,'state_id':parentmeta['state_id'],'source':str(src.relative_to(resolve('.'))),'rows':len(f),'candidate_rows':len(old),'readouts':len(models[env]),'protocol_sha256':sha256(out/'protocol.json'),'tokens_sha256':sha256(tokenspath),'files':{p.name:sha256(p) for p in [csv,npz]},'seconds':time.perf_counter()-started});print('ANALYZED',env,split,src.stem,len(f),round(time.perf_counter()-started,2),flush=True)


def weighted_quantile(frame,col,q):
    v=frame[col].to_numpy();weights=1/frame.groupby('state_id').state_id.transform('size').to_numpy();order=np.argsort(v,kind='stable');cumulative=np.cumsum(weights[order]);return float(v[order[min(np.searchsorted(cumulative,q*cumulative[-1]),len(v)-1)]])


def summarize(out,cfg):
    parts=[]
    for p in sorted(out.glob('*/*/*.csv.gz')):
        f=pd.read_csv(p);f=f[f.active].copy()
        for k in ['environment','split','representation','target','readout','method','phase']:f[k]=f[k].astype('category')
        parts.append(f)
    all_data=pd.concat(parts,ignore_index=True);groupkeys=['environment','representation','target','readout'];thresholds=[]
    cal=all_data[(all_data.split=='calibration')&(all_data.method=='real_encoded')]
    for key,g in cal.groupby(groupkeys,observed=True):
        thresholds.append({**dict(zip(groupkeys,key)),'position':max(1e-9,weighted_quantile(g,'position_effect_error',cfg['resolution_quantile'])),'rotation':max(1e-9,weighted_quantile(g,'rotation_effect_error',cfg['resolution_quantile'])),'states':int(g.state_id.nunique()),'families':int(g.initial_pose_family.nunique())})
    lookup={tuple(r[k] for k in groupkeys):r for r in thresholds};states=[];paired=[]
    for key,g in all_data.groupby(['environment','split','state_id','representation','target','readout'],observed=True,sort=False):
        env,split,sid,rep,kind,learner=key;threshold=lookup[(env,rep,kind,learner)];common=dict(zip(['environment','split','state_id','representation','target','readout'],key));common['initial_pose_family']=int(g.initial_pose_family.iloc[0]);common['phase']=str(g.phase.iloc[0])
        for method,v in g.groupby('method',observed=True):
            valid=v[np.isfinite(v.position_effect_error)&np.isfinite(v.rotation_effect_error)];rp=valid.true_position_magnitude>threshold['position'];rr=valid.true_rotation_magnitude>threshold['rotation']
            row={**common,'method':str(method),'active_candidates':len(v),'valid_candidates':len(valid),'control_support':len(valid)/len(v),'physical_preservation':float((v.physical_E<=1).mean()),'position_resolvable_fraction':float(rp.mean()) if len(valid) else None,'rotation_resolvable_fraction':float(rr.mean()) if len(valid) else None,'position_direction_agreement':float((valid.loc[rp,'position_cosine']>=cfg['position_direction_cosine']).mean()) if rp.any() else None,'rotation_direction_agreement':float((valid.loc[rr,'rotation_cosine']>=cfg['rotation_direction_cosine']).mean()) if rr.any() else None}
            for col in ['position_effect_error','rotation_effect_error','position_absolute_error','rotation_absolute_error','predicted_position_magnitude','predicted_rotation_magnitude','true_position_magnitude','true_rotation_magnitude']:row[col]=float(valid[col].mean()) if len(valid) else None
            states.append(row)
        predicted=g[g.method=='predicted']
        for control in ['real_encoded','zero_effect','persistence','wrong_action']:
            matched=predicted.merge(g[g.method==control],on='candidate',suffixes=('','_control'),validate='one_to_one');valid=matched[np.isfinite(matched.position_effect_error_control)&np.isfinite(matched.rotation_effect_error_control)]
            paired.append({**common,'control':control,'matched_candidates':len(valid),'position_difference':float((valid.position_effect_error-valid.position_effect_error_control).mean()) if len(valid) else None,'rotation_difference':float((valid.rotation_effect_error-valid.rotation_effect_error_control).mean()) if len(valid) else None})
    s=pd.DataFrame(states);p=pd.DataFrame(paired);summary=[];contrasts=[];metrics=['control_support','physical_preservation','position_resolvable_fraction','rotation_resolvable_fraction','position_direction_agreement','rotation_direction_agreement','position_effect_error','rotation_effect_error','position_absolute_error','rotation_absolute_error','predicted_position_magnitude','predicted_rotation_magnitude','true_position_magnitude','true_rotation_magnitude']
    for key,g in s.groupby(['environment','split','representation','target','readout','method'],observed=True):
        for phase in ['all']+sorted(g.phase.unique().tolist()):
            h=g if phase=='all' else g[g.phase==phase];summary.append({**dict(zip(['environment','split','representation','target','readout','method'],key)),'phase':phase,'states':len(h),'families':int(h.initial_pose_family.nunique()),'valid_candidates':int(h.valid_candidates.sum()),**{k:mean_ci(h.dropna(subset=[k]),k) for k in metrics}})
    for key,g in p.groupby(['environment','split','representation','target','readout','control'],observed=True):contrasts.append({**dict(zip(['environment','split','representation','target','readout','control'],key)),'matched_candidates':int(g.matched_candidates.sum()),**{k:mean_ci(g.dropna(subset=[k]),k) for k in ['position_difference','rotation_difference']}})
    s.to_csv(out/'state_metrics.csv',index=False);p.to_csv(out/'paired_state_metrics.csv',index=False);write_json(out/'resolution.json',{'thresholds':thresholds,'config':cfg,'scope':'Calibration real-encoded active effects; equal state weight. Diagnostic resolution only; calibration directional results are resubstitution.'});write_json(out/'analysis.json',{'summary':summary,'paired_controls':contrasts,'candidate_method_rows':len(all_data),'state_method_rows':len(s),'scope':cfg['scope'],'note':'Summary averages candidate errors within each state, then bootstraps families of states. Unsupported wrong-action controls remain unavailable. Direction accuracy treats zero predicted direction as failure on resolvable true effects. No effect-resolution filtering of physical preservation.'})
    print('SUMMARIZED',len(all_data),len(s),len(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--dependency-pid',type=int);args=p.parse_args();out,capture,cfg,m,models,protocol=prepare(args.smoke);q=np.load(READOUT/'projection.npy');pending=[(env,split,resolve(name)) for env,splits in m['states'].items() for split,paths in splits.items() for name in paths];start=time.monotonic()
    while pending:
        remaining=[]
        for env,split,src in pending:
            path=capture/env/split/src.name
            if not path.exists():remaining.append((env,split,src));continue
            try:read(path)
            except json.JSONDecodeError:remaining.append((env,split,src));continue
            process(env,split,src,capture,out,cfg,models,q)
        pending=remaining
        if pending:
            assert args.dependency_pid,'Collection incomplete; provide live parent PID';os.kill(args.dependency_pid,0)
            if time.monotonic()-start>21600:raise RuntimeError('Observation window elapsed; inspect running collection')
            time.sleep(5)
    while not (capture/'raw_audit.json').exists():
        assert args.dependency_pid,'Collection audit missing';os.kill(args.dependency_pid,0);time.sleep(5)
    audit=read(capture/'raw_audit.json');assert audit['states']==(4 if args.smoke else 402) and audit['manifest_sha256']==sha256(capture/'manifest.json')
    for name,h in protocol['inputs'].items():assert sha256(resolve(name))==h,name
    summarize(out,cfg);write_json(out/'analysis_complete.json',{'smoke':args.smoke,'states':audit['states'],'analysis_sha256':sha256(out/'analysis.json'),'resolution_sha256':sha256(out/'resolution.json'),'collection_audit_sha256':sha256(capture/'raw_audit.json'),'scope':'Analysis complete; independent audit required before research reporting.'})
if __name__=='__main__':main()
