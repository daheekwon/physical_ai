"""Numerical native-inference and action-Jacobian audit on a fixed development subset."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.model_audit import encode_representations,predict_candidates
from zj_wam.io import resolve,write_json,sha256


def main():
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());out=resolve(cfg['output'])
    acfg=json.loads(resolve('configs/followup_f2_analysis.json').read_text());m=json.loads((out/'protocol/manifest.json').read_text())
    data=ExpertDataset(cfg['dataset']);torch.set_num_threads(cfg['threads']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device'])
    assert not model.model.training and all(not p.requires_grad for p in model.model.parameters())
    records=[]
    specs=[s for s in m['states'] if s['split']=='development'][:acfg['finite_difference_audit_states']]
    for spec in specs:
        path=out/'interventions'/f"{spec['state_id']}_h05.npz"
        with np.load(path) as a:
            images=a['history_images'];past=a['past_actions'];raw=a['actions'][:4];jac=a['jacobian'];directions=a['directions'][:3]
        predict,context=model.predictor(images,past);x=torch.as_tensor(model.normalize(raw[0]).reshape(-1),device=model.device)
        latent=encode_representations(model,images)['projected_cls']
        native_error=float(np.max(np.abs(latent-context['embedding'].cpu().numpy()[0])))
        batch=predict_candidates(model,context['embedding'],context['past'],raw)
        with torch.no_grad():single=np.stack([predict(torch.as_tensor(model.normalize(v).reshape(-1),device=model.device)).cpu().numpy() for v in raw])
        batch_error=float(np.max(np.abs(batch[:,-1]-single)))
        for i,d in enumerate(directions):
            d=torch.as_tensor(d,dtype=x.dtype,device=model.device);step=acfg['finite_difference_step']
            with torch.no_grad():fd=((predict(x+step*d)-predict(x-step*d))/(2*step)).cpu().numpy()
            exact=jac@d.cpu().numpy();rel=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1e-8))
            records.append({**spec,'direction':i,'relative_fd_error':rel,'native_encoding_error':native_error,'batch_prediction_error':batch_error})
    if max(r['native_encoding_error'] for r in records)>2e-5:raise AssertionError('Different native encoder path')
    if max(r['batch_prediction_error'] for r in records)>2e-5:raise AssertionError('Batch/scalar mismatch')
    write_json(out/'protocol/model_numerical_audit.json',{'rows':records,'parameters_frozen':True,'max_relative_fd_error':max(r['relative_fd_error'] for r in records),
               'max_encoding_error':max(r['native_encoding_error'] for r in records),'max_batch_error':max(r['batch_prediction_error'] for r in records),
               'checkpoint_unchanged':sha256(resolve(cfg['checkpoint'])/'weights.pt')==m['checkpoint_sha256']})
    print(json.dumps({'directions':len(records),'max_fd_error':max(r['relative_fd_error'] for r in records)}))

if __name__=='__main__':main()
