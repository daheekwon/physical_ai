"""Exploratory action-norm plus realized EEF amplitude/frequency matching."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.fft import dct
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_response_selection import mean_ci

BASE=resolve('artifacts/followup_v1/cube')


def read(p):return json.loads(p.read_text())


def robot_features(a,cfg):
    xyz=a['candidate_eef_position'];quat=a['candidate_eef_quaternion'];n,t,_=xyz.shape
    dp=xyz-xyz[:1];r=Rotation.from_quat(quat[:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(n,t,3,3)
    relative=r@r[:1].transpose(0,1,3,2);rotvec=Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(n,t,3)
    distance=np.linalg.norm(dp,axis=-1);angle=np.linalg.norm(rotvec,axis=-1)
    waveform=np.concatenate([dp,cfg['yaw_length_scale_m']*rotvec],axis=-1);power=(dct(waveform,axis=1,norm='ortho')**2).sum(-1);frequencies=np.arange(t)/(2*t*cfg['physics_dt'])
    bands=np.stack([power[:,(frequencies>=lo)&(frequencies<hi)].sum(1) for lo,hi in [(0,2.5),(2.5,5),(5,10),(10,np.inf)]],axis=1)
    total=bands.sum(1);normalized=bands/np.maximum(total[:,None],1e-30);normalized[total<1e-20]=np.array([1.,0.,0.,0.])
    return {'translation_rms':np.sqrt((distance**2).mean(1)),'translation_max':distance.max(1),'rotation_rms':np.sqrt((angle**2).mean(1)),'rotation_max':angle.max(1),'band_power':normalized,'total_power':total}


def main():
    cfg=read(resolve('configs/followup_cube.json'));mcfg=read(resolve('configs/followup_cube_motion_matching.json'));manifest=read(BASE/'manifest.json');audit=read(BASE/'raw_audit.json');out=BASE/'motion_matching';out.mkdir(exist_ok=True)
    spec={'config':mcfg,'config_sha256':sha256(resolve('configs/followup_cube_motion_matching.json')),'source_sha256':sha256(Path(__file__)),'parent_raw_audit_sha256':sha256(BASE/'raw_audit.json'),'parent_manifest_sha256':sha256(BASE/'manifest.json')}
    path=out/'manifest.json'
    if path.exists():assert read(path)==spec
    else:write_json(path,spec)
    records=[];normmax=0.;rawrows=0;quaternion_error=0.
    for split in mcfg['splits']:
        paths=sorted((BASE/split).glob('e*.csv'));assert len(paths)==audit[split]['states']==96
        for path in paths:
            meta=read(path.with_suffix('.json'));assert meta['signature']==manifest['signature'];f=pd.read_csv(path)
            with np.load(path.with_suffix('.npz')) as a:
                feat=robot_features(a,cfg);rawrows+=len(f)
                # Independent scalar quaternion dot-product check of angular maxima.
                q=a['candidate_eef_quaternion'];q=q/np.linalg.norm(q,axis=-1,keepdims=True);angle=2*np.arccos(np.clip(np.abs((q*q[:1]).sum(-1)),0,1));e=float(np.max(np.abs(angle.max(1)-feat['rotation_max'])));assert e<1e-6;quaternion_error=max(quaternion_error,e)
                actions=a['actions'];norm=np.linalg.norm(((actions-a['nominal_actions'])/np.asarray(manifest['normalization']['scale'])).reshape(len(f),-1),axis=1);np.testing.assert_allclose(norm,f.norm,atol=1e-10,rtol=0)
            direction=f[f.method.isin([*mcfg['directions'],*mcfg['random_controls']])]
            for (radius,sign),g in direction.groupby(['radius','sign']):
                if set(g.method)!=set([*mcfg['directions'],*mcfg['random_controls']]):continue
                indexed=g.set_index('method')
                for name in mcfg['directions']:
                    left=indexed.loc[name];i=int(left.candidate)
                    for control in mcfg['random_controls']:
                        right=indexed.loc[control];j=int(right.candidate);ne=abs(float(left.norm-right.norm));assert ne<1e-10;normmax=max(normmax,ne)
                        row={'split':split,'state_id':meta['state_id'],'initial_pose_family':meta['episode'],'phase':meta['phase'],'direction':name,'control':control,'radius':radius,'sign':sign,'left_candidate':i,'right_candidate':j,
                            'object_error_difference':float(left.object_error-right.object_error),'position_max_difference_m':float(left.object_position_max_m-right.object_position_max_m),'rotation_max_difference_rad':float(left.object_rotation_max_rad-right.object_rotation_max_rad),
                            'preserved_difference':float(left.object_error<=1)-float(right.object_error<=1),'jerk_gain_difference':float(left.jerk_gain-right.jerk_gain),'left_E':float(left.object_error),'right_E':float(right.object_error)}
                        mismatch=[]
                        for key in ['translation_rms','translation_max','rotation_rms','rotation_max']:
                            x,y=feat[key][i],feat[key][j];floor=mcfg['translation_floor_m'] if key.startswith('translation') else mcfg['rotation_floor_rad'];d=abs(x-y)/max(x,y,floor);row[key+'_left']=float(x);row[key+'_right']=float(y);row[key+'_mismatch']=float(d);mismatch.append(d)
                        row['amplitude_mismatch']=float(max(mismatch));row['frequency_distance']=float(np.abs(feat['band_power'][i]-feat['band_power'][j]).sum()/2)
                        row['informative_motion']=bool(max(feat['translation_rms'][i],feat['translation_rms'][j])>1e-5 or max(feat['rotation_rms'][i],feat['rotation_rms'][j])>1e-4)
                        records.append(row)
    f=pd.DataFrame(records);summary=[];phase_support=[];state_rows=[]
    keys=['object_error_difference','position_max_difference_m','rotation_max_difference_rad','preserved_difference','jerk_gain_difference']
    for (split,direction),pool in f.groupby(['split','direction']):
        for caliper in [None,*mcfg['calipers']]:
            for mode in (['action_norm'] if caliper is None else ['amplitude','amplitude_frequency']):
                selected=pool if caliper is None else pool[pool.amplitude_mismatch<=caliper]
                if mode=='amplitude_frequency':selected=selected[selected.frequency_distance<=caliper]
                for phase in cfg['phases']:
                    before=pool[pool.phase==phase];after=selected[selected.phase==phase];phase_support.append({'split':split,'direction':direction,'mode':mode,'caliper':caliper,'phase':phase,'pool_pairs':len(before),'matched_pairs':len(after),'matched_states':after.state_id.nunique(),'matched_families':after.initial_pose_family.nunique()})
                states=selected.groupby(['state_id','initial_pose_family'],as_index=False)[keys].mean()
                record={'split':split,'direction':direction,'mode':mode,'caliper':caliper,'pool_pairs':len(pool),'matched_pairs':len(selected),'unmatched_pairs':len(pool)-len(selected),'matched_states':len(states),'matched_families':states.initial_pose_family.nunique(),'informative_pairs':int(selected.informative_motion.sum()),'max_amplitude_mismatch':float(selected.amplitude_mismatch.max()) if len(selected) else None,'max_frequency_distance':float(selected.frequency_distance.max()) if len(selected) else None,**{k:mean_ci(states,k) for k in keys}}
                summary.append(record);states['split']=split;states['direction']=direction;states['mode']=mode;states['caliper']=caliper;state_rows.append(states)
    f.to_csv(out/'candidate_pairs.csv',index=False);pd.concat(state_rows,ignore_index=True).to_csv(out/'matched_state_means.csv',index=False);pd.DataFrame(phase_support).to_csv(out/'phase_support.csv',index=False)
    write_json(out/'analysis.json',{'summary':summary,'phase_support':phase_support,'primary_caliper':mcfg['primary_caliper'],'scope':mcfg['scope']})
    write_json(out/'raw_audit.json',{'raw_candidates':rawrows,'pair_rows':len(f),'max_pair_action_norm_difference':normmax,'max_rotation_matrix_vs_quaternion_error':quaternion_error,'source_sha256':sha256(Path(__file__)),'parent_raw_audit_sha256':sha256(BASE/'raw_audit.json'),'matching_uses_object_results':False,'summary_weights':'Equal matchedstate mean, initialfamily bootstrap','scope':'All192parentstates read; robotfeatures500Hz,actionnorm independentlychecked; physicalobjectoutcomes inheritedfromparentfullrawaudit. Pair selection based only declaredstate/actionfeasibility and robot features, not object effects.'})
    for r in summary:
        if r['split']=='development' and r['direction']=='lewm_low':print(r['mode'],r['caliper'],'pairs',r['matched_pairs'],'states',r['matched_states'],'families',r['matched_families'],'E difference',r['object_error_difference'],flush=True)
if __name__=='__main__':main()
