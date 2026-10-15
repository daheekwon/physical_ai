"""Independently audit frozen final Cube effects using matrix angular errors."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,sha256,write_json
from zj_wam.cube_readouts import predict,rotations,align_rotations,future_features
from scripts.audit_followup_cube_probes import matrix_angle

BASE=resolve('artifacts/followup_v1/cube')
read=lambda p:json.loads(p.read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validation-split',choices=['development','calibration']);args=parser.parse_args()
    split=args.validation_split or 'test';dest=BASE/('final_validation_'+split if args.validation_split else 'final_analysis')
    analysis=read(dest/'analysis.json');audit=read(dest/'raw_audit.json');assert analysis['split']==split and audit['no_fit_or_calibration']
    if split=='test':
        from scripts.run_followup_cube_final import verify,LOCK
        lock=verify();assert analysis['lock_sha256']==sha256(LOCK) and not audit['partial']
        assert audit['states']==lock['planned_states']
    source=BASE/'probes_v2';selection=read(source/'selection.json');resolution=read(source/'analysis.json')['resolution_thresholds'];models={}
    for name,spec in selection['readouts'].items():
        path=source/(name+'.joblib');assert sha256(path)==spec['model_sha256'];models[name]=joblib.load(path)
    f=pd.read_csv(dest/'effects.csv');assert not f.duplicated(['split','state_id','readout','control','candidate']).any()
    assert set(f.split)=={split};groups={key:g.set_index('candidate').sort_index() for key,g in f.groupby(['state_id','readout','control'],sort=False)}
    cfg=read(resolve('configs/followup_cube.json'));endpoint=cfg['horizon']*cfg['substeps']-1;checked=0;errors={};inputs={}
    def check(name,x,y,atol=1e-6):
        error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),error);assert error<=atol,(name,error)
    for state in sorted(f.state_id.unique()):
        path=BASE/split/(state+'.npz');meta=read(path.with_suffix('.json'));raw=pd.read_csv(path.with_suffix('.csv')).set_index('candidate').sort_index();inputs[str(path.relative_to(resolve('.')))]=sha256(path)
        with np.load(path) as a:
            position=a['candidate_object_pose'][:,endpoint,:3];quat=a['candidate_object_pose'][:,endpoint,3:];actual=Rotation.from_quat(quat[:,[1,2,3,0]]).as_matrix();true_rot=actual@actual[0].T;tp=position-position[:1];order=np.roll(np.arange(len(position)),1);ids=np.flatnonzero(raw.norm.to_numpy()>1e-6)
            for name,model in models.items():
                spec=selection['readouts'][name];decoded={}
                for control,key in [('real_endpoint','actual_endpoint_latent'),('predicted','predicted_latent')]:
                    value=predict(model,future_features(a['history_embedding'],a[key],spec['representation']));rot=rotations(value)
                    if spec['kind']!='ridge_raw':rot=align_rotations(rot,rot[0])
                    matrices=rot.as_matrix();decoded[control]=(value[:,:3]-value[:1,:3],matrices@matrices[0].T)
                decoded['zero']=(np.zeros_like(tp),np.broadcast_to(np.eye(3),(len(tp),3,3)))
                decoded['wrong_action']=(decoded['predicted'][0][order],decoded['predicted'][1][order])
                for control,(dp,dr) in decoded.items():
                    g=groups[(state,name,control)];np.testing.assert_array_equal(g.index.to_numpy(),ids)
                    assert set(g.initial_pose_family)=={meta['episode']} and set(g.phase)=={meta['phase']}
                    true_mag=np.linalg.norm(tp,axis=1);pred_mag=np.linalg.norm(dp,axis=1);true_angle=matrix_angle(true_rot)
                    values={'position_effect_error_m':np.linalg.norm(dp-tp,axis=1),'rotation_effect_error_rad':matrix_angle(dr@np.swapaxes(true_rot,1,2)),'true_position_effect_m':true_mag,'predicted_position_effect_m':pred_mag,'true_rotation_effect_rad':true_angle,'direction_cosine':np.sum(dp*tp,axis=1)/np.maximum(pred_mag*true_mag,1e-15)}
                    for key,v in values.items():check(key,v[ids],g[key])
                    np.testing.assert_array_equal(g.position_resolved,true_mag[ids]>resolution[name]['position_m'])
                    np.testing.assert_array_equal(g.rotation_resolved,true_angle[ids]>resolution[name]['rotation_rad'])
                    np.testing.assert_array_equal(g.prediction_has_direction,pred_mag[ids]>1e-8)
                    checked+=len(g)
    assert checked==len(f)==analysis['probes']['effect_rows']
    assert resolution==analysis['probes']['resolution_thresholds']
    for row in analysis['probes']['effects']:
        if row['control']=='zero':assert row['resolved_position_direction']['mean'] is None
    paths=[dest/'analysis.json',dest/'raw_audit.json',dest/'effects.csv',source/'selection.json',source/'analysis.json',Path(__file__),resolve('scripts/audit_followup_cube_probes.py')]
    inputs.update({str(p.relative_to(resolve('.'))):sha256(p) for p in paths})
    record={'split':split,'validation_only':bool(args.validation_split),'states':int(f.state_id.nunique()),'effect_rows':checked,'max_error':errors,'inputs':inputs,'scope':'Frozen readout, real/predicted/zero/wrong action effects independently reconstructed from physical poses; rotation via matrix trace/skew, fixed development resolution, no final-data fitting. Physical candidate audit remains separate.'}
    if split=='test':verify();record['lock_sha256']=sha256(LOCK)
    write_json(dest/'probe_effect_raw_audit.json',record);print('PASS independent Cube',split,'effects',checked,'states',record['states'],flush=True)

if __name__=='__main__':main()
