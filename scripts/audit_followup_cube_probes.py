"""Reconstruct Cube probe labels and effect errors from saved physics and model files."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,joblib
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,write_json,sha256
from zj_wam.cube_readouts import predict,rotations,align_rotations,future_features


def matrix_angle(matrix):
    m=np.asarray(matrix);s=np.stack([m[...,2,1]-m[...,1,2],m[...,0,2]-m[...,2,0],m[...,1,0]-m[...,0,1]],axis=-1)
    return np.arctan2(np.linalg.norm(s,axis=-1)/2,np.clip((np.trace(m,axis1=-2,axis2=-1)-1)/2,-1,1))


def main():
    out=resolve('artifacts/followup_v1/cube');dest=out/'probes_v2';selection=json.loads((dest/'selection.json').read_text());analysis=json.loads((dest/'analysis.json').read_text());f=pd.read_csv(dest/'effects.csv');groups={key:g.set_index('candidate').sort_index() for key,g in f.groupby(['split','state_id','readout','control'],sort=False)};models={n:joblib.load(dest/(n+'.joblib')) for n in selection['readouts']};errors={};checked=0
    def check(name,x,y,tol=1e-6):
        e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),e);assert e<=tol,(name,e)
    # Reconstruct training labels independently, including native->world angular velocity.
    targets=[];families=[]
    for path in sorted((out/'probe_train').glob('e*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as a:
            for when in ['current','future']:
                pose=a[when+'_object_pose'];velocity=a[when+'_object_velocity'];r=Rotation.from_quat(pose[[4,5,6,3]]);targets.append(np.r_[pose[:3],r.as_matrix().reshape(-1),velocity[:3],r.apply(velocity[3:]),a[when+'_eef_position']]);families.append(meta['episode'])
    with np.load(dest/'fit_audit.npz') as a:
        check('training_world_labels',targets,a['train_y'],1e-10);check('training_family_alignment',families,a['train_family'],0.);assert not set(families)&set(a['development_family'])
        for name,model in models.items():
            rep=selection['readouts'][name]['representation'];check('saved_development_prediction',predict(model,a[rep+'_development_x']),a[name+'_development_pred'],1e-9)
    for split in ['development','calibration']:
        for path in sorted((out/split).glob('e*.npz')):
            with np.load(path) as a:
                position=a['candidate_object_pose'][:,124,:3];q=a['candidate_object_pose'][:,124,3:];actual=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();true_effect=actual@actual[0].T;tp=position-position[:1];true_angle=matrix_angle(true_effect);order=np.roll(np.arange(len(position)),1)
                raw=pd.read_csv(path.with_suffix('.csv')).set_index('candidate')
                for name,model in models.items():
                    spec=selection['readouts'][name];rep=spec['representation'];decoded={}
                    for kind,latent in [('real_endpoint',a['actual_endpoint_latent']),('predicted',a['predicted_latent'])]:
                        value=predict(model,future_features(a['history_embedding'],latent,rep));r=rotations(value)
                        if spec['kind']!='ridge_raw':r=align_rotations(r,r[0])
                        matrix=r.as_matrix();decoded[kind]=(value[:,:3]-value[:1,:3],matrix@matrix[0].T)
                    decoded['zero']=(np.zeros_like(tp),np.broadcast_to(np.eye(3),(len(tp),3,3)));decoded['wrong_action']=(decoded['predicted'][0][order],decoded['predicted'][1][order])
                    for control,(dp,dr) in decoded.items():
                        g=groups[(split,path.stem,name,control)];ids=g.index.to_numpy();assert np.array_equal(ids,np.flatnonzero(raw.norm.to_numpy()>1e-6));position_error=np.linalg.norm(dp-tp,axis=1);rotation_error=matrix_angle(dr@np.swapaxes(true_effect,1,2));true_mag=np.linalg.norm(tp,axis=1);pred_mag=np.linalg.norm(dp,axis=1);cosine=np.sum(tp*dp,axis=1)/np.maximum(true_mag*pred_mag,1e-15)
                        for key,values in [('position_effect_error_m',position_error),('rotation_effect_error_rad',rotation_error),('true_position_effect_m',true_mag),('predicted_position_effect_m',pred_mag),('true_rotation_effect_rad',true_angle),('direction_cosine',cosine)]:check(key,values[ids],g[key])
                        check('physical_preservation_label_unchanged',raw.loc[ids,'object_error'],g.object_error,1e-10);check('physical_action_norm_unchanged',raw.loc[ids,'norm'],g.action_norm,1e-10);checked+=len(g)
    assert checked==len(f)
    assert np.array_equal(f.prediction_has_direction.to_numpy(),(f.predicted_position_effect_m>1e-8).to_numpy())
    for row in analysis['effects']:
        if row['control']=='zero':assert row['resolved_position_direction']['mean'] is None
    for name,threshold in analysis['resolution_thresholds'].items():
        g=f[(f.split=='development')&(f.readout==name)&(f.control=='real_endpoint')];check('resolution_position',max(1e-6,g.position_effect_error_m.quantile(.9)),threshold['position_m'],1e-10);check('resolution_rotation',max(1e-6,g.rotation_effect_error_rad.quantile(.9)),threshold['rotation_rad'],1e-10)
    write_json(dest/'raw_audit.json',{'training_rows':len(targets),'effect_rows':checked,'max_recalculation_error':errors,'source_sha256':sha256(Path(__file__)),'scope':'Physicalworldlabels,disjointfamilies,savedreadoutpredictions,real/predicted/zero/wrongaction effectrows; angularerrors independently reconstructed via rotationmatrix trace/skew rather than scipyrotation.magnitude. Samefullphysical preservationlabels andactionnorm. Developmentresolutionthresholds reconstructed.'});print('PASS',checked,'Cube probe effectrows',errors,flush=True)
if __name__=='__main__':main()
