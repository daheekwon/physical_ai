"""Native Cube Jacobian: step sweep, float64 reference and signed differences."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from zj_wam.io import resolve,write_json,sha256
from scripts.run_followup_cube import Model


def main():
 cfg=json.loads(resolve('configs/followup_cube.json').read_text());out=resolve(cfg['output']);manifest=json.loads((out/'manifest.json').read_text());mean=np.asarray(manifest['normalization']['mean']);scale=np.asarray(manifest['normalization']['scale']);torch.set_num_threads(4);model=Model(cfg,mean,scale);rows=[];comparisons=[]
 for path in sorted((out/'pilot').glob('e*.npz')):
  with np.load(path) as a:
   embedding,predict=model.predictor(a['history_images'],a['past_actions']);x0=((a['nominal_actions']-mean)/scale).reshape(-1);x=torch.tensor(x0,dtype=torch.float32,device=cfg['device']);reference=predict(x).detach().cpu().numpy();replay=float(np.max(np.abs(reference-a['predicted_latent'][0])));assert replay<2e-5,(path,replay)
   q=a['polynomial_basis'];rng=np.random.default_rng(cfg['seed']);directions=[q[:,0],q[:,-1]]
   for _ in range(2):z=rng.normal(size=q.shape[1]);directions.append(q@(z/np.linalg.norm(z)))
   for index,d in enumerate(directions):
    expected=a['jacobian']@d
    for epsilon in [.1,.03,.01,.003,.001,.0003]:
     delta=torch.tensor(epsilon*d,dtype=torch.float32,device=cfg['device'])
     with torch.no_grad():fd=((predict(x+delta)-predict(x-delta))/(2*epsilon)).cpu().numpy()
     error=float(np.linalg.norm(fd-expected)/max(np.linalg.norm(expected),1e-12));rows.append({'state_id':path.stem,'precision':'float32','direction':index,'epsilon':epsilon,'relative_error':error,'absolute_error':float(np.linalg.norm(fd-expected)),'expected_norm':float(np.linalg.norm(expected)),'saved_prediction_max_error':replay})
   # Predictor-only float64 comparison retains exactly the native float32 encoded history/input values.
   model.model.double();emb64=embedding.double();old=torch.tensor(((a['past_actions']-mean)/scale).astype(np.float32).reshape(1,2,25),device=cfg['device']).double();x64=x.double()
   def pred64(z):return model.model.predict(emb64,model.model.action_encoder(torch.cat([old,z.reshape(1,1,25)],1)))[:,-1].reshape(-1)
   j64=torch.autograd.functional.jacobian(pred64,x64,vectorize=True).detach().cpu().numpy();restricted32=a['jacobian']@q;restricted64=j64@q;relative=float(np.linalg.norm(restricted32-restricted64)/max(np.linalg.norm(restricted64),1e-12));comparisons.append({'state_id':path.stem,'restricted_jacobian_float32_vs_float64_relative_error':relative});assert relative<.01,(path,relative)
   for index,d in enumerate(directions):
    expected=j64@d;errors=[]
    for epsilon in [.01,.001,.0001]:
     delta=torch.tensor(epsilon*d,dtype=torch.float64,device=cfg['device'])
     with torch.no_grad():fd=((pred64(x64+delta)-pred64(x64-delta))/(2*epsilon)).cpu().numpy()
     error=float(np.linalg.norm(fd-expected)/max(np.linalg.norm(expected),1e-12));errors.append(error);rows.append({'state_id':path.stem,'precision':'float64','direction':index,'epsilon':epsilon,'relative_error':error,'absolute_error':float(np.linalg.norm(fd-expected)),'expected_norm':float(np.linalg.norm(expected))})
    assert min(errors)<.001,(path,index,errors)
   model.model.float()
 write_json(out/'derivative_audit.json',{'rows':rows,'comparisons':comparisons,'source_sha256':sha256(Path(__file__)),'scope':'Fourpilotphases x fourpolynomialdirections. Nativefloat32step sweep exposes cancellation at small sensitivities. Predictorfloat64 at identicalencodedhistory/input verifies autodiff and signedfinite difference. Float32/native remains actualexperiment; no silentprecisionchange. Priorfloat32only5percent gate failed intransportdirection2 atstep.01; retained /tmp/zj_cube_derivatives_initial.log.'});print('PASS',len(rows),'finite difference comparisons',comparisons,flush=True)
if __name__=='__main__':main()
