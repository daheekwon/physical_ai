"""Cube capture/resolution/propagation with history and symmetry-aware diagnostic probes."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib
from zj_wam.io import resolve,write_json,sha256
from zj_wam.cube_readouts import target,fit,predict,capture_errors,effects,canonical_target,current_features,future_features
from scripts.analyze_followup_response_selection import mean_ci


def signature():return {p:sha256(resolve(p)) for p in ['configs/followup_cube_probes.json','zj_wam/cube_readouts.py','scripts/analyze_followup_cube_probes.py']}


def training(out,manifest,representation):
    xs=[];ys=[];families=[];paths=sorted((out/'probe_train').glob('e*.npz'));assert len(paths)==4*len(manifest['splits']['probe_train'])
    for path in paths:
        m=json.loads(path.with_suffix('.json').read_text());assert m['signature']==manifest['signature']
        with np.load(path) as a:
            h=a['history_embedding'];np.testing.assert_array_equal(h[-1],a['current_embedding']);x=[current_features(h,representation),future_features(h,a['future_embedding'][None],representation)[0]]
            for index,when in enumerate(['current','future']):xs.append(x[index]);ys.append(target(a[when+'_object_pose'][None],a[when+'_object_velocity'][None],a[when+'_eef_position'][None])[0]);families.append(m['episode'])
    return np.asarray(xs),np.asarray(ys),np.asarray(families)


def captures(folder,manifest,representation,partial=False):
    xs=[];ys=[];rows=[];paths=sorted(folder.glob('e*.npz'))
    if not partial:assert len(paths)==4*len(manifest['splits'][folder.name]),folder
    for path in paths:
        m=json.loads(path.with_suffix('.json').read_text());assert m['signature']==manifest['signature']
        with np.load(path) as a:
            h=a['history_embedding'];x=[current_features(h,representation),future_features(h,a['actual_endpoint_latent'][:1],representation)[0]];y=[target(a['root_object_pose'][None],a['root_velocity'][None],a['root_eef_position'][None])[0],target(a['candidate_object_pose'][0,124][None],a['candidate_object_velocity'][0,124][None],a['candidate_eef_position'][0,124][None])[0]]
        xs.extend(x);ys.extend(y)
        for when in ['current','future']:rows.append({'state_id':m['state_id'],'initial_pose_family':m['episode'],'split':folder.name,'phase':m['phase'],'when':when})
    return np.asarray(xs),np.asarray(ys),pd.DataFrame(rows)


def fit_models(out,cfg,pcfg,manifest,smoke=False):
    started=time.perf_counter();folder=out/('pilot' if smoke else 'development');dest=out/('probe_smoke_v2' if smoke else 'probes_v2');dest.mkdir(exist_ok=True);selection={};curves=[];captures_all=[];archive={};seed=cfg['seed']
    for representation in pcfg['representations']:
        x,y,families=training(out,manifest,representation);cy=canonical_target(y);vx,vy,meta=captures(folder,manifest,representation,smoke);archive[representation+'_train_x']=x;archive[representation+'_development_x']=vx;archive['train_y']=y;archive['canonical_train_y']=cy;archive['train_family']=families;archive['development_y']=vy;archive['development_family']=meta.initial_pose_family.to_numpy();shuffled=np.random.default_rng(seed).permutation(len(y));archive['shuffle_indices']=shuffled;assert not set(families)&set(meta.initial_pose_family)
        chosen={}
        for kind,labels,error_key in [('ridge_raw',y,'rotation_rad'),('ridge_symmetry',cy,'rotation_mod_cube_symmetry_rad')]:
            trial=[];models={}
            for alpha in pcfg['alphas']:
                model=fit(x,labels,'ridge',alpha,seed);e=capture_errors(predict(model,vx),vy);score=float(np.mean(e['position_m']/cfg['position_tolerance_m']+e[error_key]/cfg['rotation_tolerance_rad']));trial.append({'representation':representation,'kind':kind,'alpha':alpha,'score':score,'position_m':float(e['position_m'].mean()),'full_rotation_rad':float(e['rotation_rad'].mean()),'symmetry_rotation_rad':float(e['rotation_mod_cube_symmetry_rad'].mean())});models[alpha]=model
            best=min(trial,key=lambda z:z['score']);chosen[kind]=models[best['alpha']];curves.extend(trial)
        chosen['label_shuffle_symmetry']=fit(x,cy[shuffled],'ridge',chosen['ridge_symmetry']['alpha'],seed);chosen['mlp_symmetry']=fit(x,cy,'mlp',0.,seed)
        for kind,model in chosen.items():
            name=representation+'__'+kind;prediction=predict(model,vx);archive[name+'_development_pred']=prediction;e=capture_errors(prediction,vy);record=meta.copy();record['readout']=name;record['representation']=representation;record['kind']=kind
            for key,value in e.items():record[key]=value
            captures_all.append(record);joblib.dump(model,dest/(name+'.joblib'));selection[name]={'representation':representation,'kind':kind,'alpha':model['alpha'],'input_dimension':x.shape[1],'target':'raw_rotation' if kind=='ridge_raw' else 'cube_symmetry_canonical','model_sha256':sha256(dest/(name+'.joblib'))};print('FIT',name,'alpha',model['alpha'],flush=True)
    pd.concat(captures_all).to_csv(dest/'development_capture.csv',index=False);np.savez_compressed(dest/'fit_audit.npz',**archive);write_json(dest/'selection.json',{'bank_signature':manifest['signature'],'probe_signature':signature(),'config':pcfg,'readouts':selection,'curves':curves,'training_rows':len(y),'training_families':len(np.unique(families)),'development_rows':len(vy),'smoke':smoke,'seconds':time.perf_counter()-started,'scope':pcfg['scope']});print('FIT COMPLETE',len(selection),flush=True)


def evaluate(out,cfg,pcfg,manifest,smoke=False):
    dest=out/('probe_smoke_v2' if smoke else 'probes_v2');selection=json.loads((dest/'selection.json').read_text());assert selection['bank_signature']==manifest['signature'] and selection['probe_signature']==signature();models={name:joblib.load(dest/(name+'.joblib')) for name in selection['readouts']};records=[];capture=[];audit={};splits=['pilot'] if smoke else ['development','calibration']
    for split in splits:
        folder=out/split
        for representation in pcfg['representations']:
            x,y,meta=captures(folder,manifest,representation,smoke)
            for name,model in models.items():
                spec=selection['readouts'][name]
                if spec['representation']!=representation:continue
                e=capture_errors(predict(model,x),y);r=meta.copy();r['readout']=name;r['representation']=representation;r['kind']=spec['kind']
                for k,v in e.items():r[k]=v
                capture.append(r)
        audit[split]={'states':0,'active_candidates':0,'max_velocity_frame_error':0.}
        for path in sorted(folder.glob('e*.npz')):
            m=json.loads(path.with_suffix('.json').read_text());raw=pd.read_csv(path.with_suffix('.csv'));audit[split]['states']+=1
            with np.load(path) as a:
                true=target(a['candidate_object_pose'][:,124],a['candidate_object_velocity'][:,124],a['candidate_eef_position'][:,124]);count=len(true);order=np.roll(np.arange(count),1);active=np.flatnonzero(raw.norm.to_numpy()>1e-6);audit[split]['active_candidates']+=len(active)
                # Independently verify native angular velocity frame on executed nominal physics.
                from scipy.spatial.transform import Rotation
                q=a['candidate_object_pose'][0,:,3:];r=Rotation.from_quat(q[:,[1,2,3,0]]);world_fd=(r[1:]*r[:-1].inv()).as_rotvec()/cfg['physics_dt'];world_v=r[1:].apply(a['candidate_object_velocity'][0,1:,3:]);frame_error=float(np.abs(world_fd-world_v).max());assert frame_error<1e-6,(path,frame_error);audit[split]['max_velocity_frame_error']=max(audit[split]['max_velocity_frame_error'],frame_error)
                for name,model in models.items():
                    spec=selection['readouts'][name];representation=spec['representation'];real=predict(model,future_features(a['history_embedding'],a['actual_endpoint_latent'],representation));imagined=predict(model,future_features(a['history_embedding'],a['predicted_latent'],representation));align=spec['kind']!='ridge_raw';rp,tp,rr,tr=effects(real,true,align);ip,_,ir,_=effects(imagined,true,align);truthangle=tr.magnitude();magnitude=np.linalg.norm(tp,axis=1)
                    for control,dp,dr in [('real_endpoint',rp,rr),('predicted',ip,ir),('zero',np.zeros_like(tp),Rotation.identity()),('wrong_action',ip[order],ir[order])]:
                        pos_error=np.linalg.norm(dp-tp,axis=1);rot_error=(dr*tr.inv()).magnitude();predmag=np.linalg.norm(dp,axis=1);direction=np.sum(dp*tp,axis=1)/np.maximum(predmag*magnitude,1e-15)
                        for i in active:
                            records.append({'split':split,'state_id':m['state_id'],'initial_pose_family':m['episode'],'phase':m['phase'],'readout':name,'representation':representation,'kind':spec['kind'],'control':control,'candidate':int(i),'position_effect_error_m':float(pos_error[i]),'rotation_effect_error_rad':float(rot_error[i]),'true_position_effect_m':float(magnitude[i]),'predicted_position_effect_m':float(predmag[i]),'true_rotation_effect_rad':float(truthangle[i]),'direction_cosine':float(direction[i]),'object_error':float(raw.object_error.iloc[i]),'action_norm':float(raw.norm.iloc[i])})
    f=pd.DataFrame(records);capture=pd.concat(capture,ignore_index=True);thresholds={}
    for name in models:
        g=f[(f.split==splits[0])&(f.readout==name)&(f.control=='real_endpoint')];thresholds[name]={'position_m':max(1e-6,float(g.position_effect_error_m.quantile(pcfg['resolution_quantile']))),'rotation_rad':max(1e-6,float(g.rotation_effect_error_rad.quantile(pcfg['resolution_quantile'])))}
    f['prediction_has_direction']=f.predicted_position_effect_m>1e-8
    f['position_resolved']=[r.true_position_effect_m>thresholds[r.readout]['position_m'] for r in f.itertuples()];f['rotation_resolved']=[r.true_rotation_effect_rad>thresholds[r.readout]['rotation_rad'] for r in f.itertuples()];summary=[]
    for (split,name,control),g in f.groupby(['split','readout','control']):
        resolved=g[g.position_resolved & g.prediction_has_direction];summary.append({'split':split,'readout':name,'control':control,'candidate_rows':len(g),**{k:mean_ci(g,k) for k in ['position_effect_error_m','rotation_effect_error_rad','position_resolved','rotation_resolved']},'resolved_position_direction':mean_ci(resolved,'direction_cosine'),'direction_available_when_resolved':mean_ci(g[g.position_resolved],'prediction_has_direction')})
    capture_summary=[]
    for (split,name),g in capture.groupby(['split','readout']):capture_summary.append({'split':split,'readout':name,**{k:mean_ci(g,k) for k in ['position_m','rotation_rad','rotation_mod_cube_symmetry_rad','linear_velocity_m_s','angular_velocity_rad_s','eef_position_m']}})
    f.to_csv(dest/'effects.csv',index=False);capture.to_csv(dest/'capture.csv',index=False);write_json(dest/'analysis.json',{'selection':selection,'resolution_thresholds':thresholds,'capture':capture_summary,'effects':summary,'scope':pcfg['scope']+' Resolution thresholds from development90thpercentile real-effect error. Symmetry-trained effect orientation aligned to decoded nominal only; physical E_obj and true relative effect retain full SO3 rotation. Unresolved cases retained in physical preservation statistics.'});write_json(dest/'input_audit.json',audit);print('EVALUATED',len(f),'effectrows',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','evaluate'],required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args();cfg=json.loads(resolve('configs/followup_cube.json').read_text());pcfg=json.loads(resolve('configs/followup_cube_probes.json').read_text());out=resolve(cfg['output']);manifest=json.loads((out/'manifest.json').read_text());(fit_models if args.phase=='fit' else evaluate)(out,cfg,pcfg,manifest,args.smoke)
if __name__=='__main__':main()
