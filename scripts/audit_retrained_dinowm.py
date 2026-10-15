"""CPU FP32 implementation audit, not a prediction-quality evaluation."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from zj_wam.io import resolve,write_json,sha256
from scripts.retrained_dinowm_core import FrozenRetrainedDinoWM
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_retrained_preflight'


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);torch.manual_seed(8142);records=[]
    paths={'pusht':sorted((BASE/'family_ablation').glob('development*_h05.npz'))[0],'ogb-cube':sorted((BASE/'cube/development').glob('*.npz'))[0]}
    for env,path in paths.items():
        start=time.perf_counter();model=FrozenRetrainedDinoWM(env);wm=model.model.wm
        with np.load(path) as a:
            images=a['history_images'];past=a['past_actions'];future=a['actions'][0] if env=='pusht' else a['nominal_actions']
        assert past.shape[0]==10 and len(future)==5
        pixels=model.pixels(images);encoded=model.encode(images);prop=torch.zeros(1,3,1);normalized=model.normalize_group(past,future)
        expected=((np.concatenate([past,future]).astype(np.float32)-model.mean.cpu().numpy())/model.scale.cpu().numpy()).reshape(1,3,-1)
        np.testing.assert_array_equal(normalized.cpu().numpy(),expected)
        with torch.no_grad():
            manual=(torch.from_numpy(images).permute(0,3,1,2).float()/255.-.5)/.5;torch.testing.assert_close(pixels[0],manual,rtol=0,atol=3e-7)
            native=wm.rollout({'visual':pixels,'proprio':prop},normalized)[0]['visual'][:,-1]
            own=model.predict(encoded,past,future);repeat=model.predict(encoded,past,future);changedprop=model.predict(encoded,past,future,proprio=np.ones((3,10))*999)
            one_error=float((own-native).abs().max());assert one_error<2e-5
            assert torch.equal(own,repeat) and torch.equal(own,changedprop)
            # Numerical two-step fixture; no two-step physical truth implied.
            future2=np.concatenate([future,future]);native2=wm.rollout({'visual':pixels,'proprio':prop},model.normalize_group(past,future2))[0]['visual'][:,-1];own2=model.predict(encoded,past,future2)
            two_error=float((own2-native2).abs().max());assert two_error<2e-5
        future_tensor=torch.tensor(future,requires_grad=True,dtype=torch.float32);projection=torch.randn_like(own);projection=projection/projection.norm()
        value=(model.predict(encoded,past,future_tensor)*projection).sum();gradient=torch.autograd.grad(value,future_tensor)[0];assert torch.isfinite(gradient).all() and gradient.norm()>0
        rng=np.random.default_rng(8142);direction=torch.tensor(rng.normal(size=future.shape),dtype=torch.float32);direction/=direction.norm();automatic=float((gradient*direction).sum());sweep=[]
        for step in [.01,.003,.001,.0003]:
            with torch.no_grad():
                plus=(model.predict(encoded,past,future_tensor.detach()+step*direction)*projection).sum();minus=(model.predict(encoded,past,future_tensor.detach()-step*direction)*projection).sum()
            finite=float((plus-minus)/(2*step));relative=abs(finite-automatic)/max(abs(finite),abs(automatic),1e-7);sweep.append({'step':step,'finite':finite,'autodiff':automatic,'relative_error':relative})
        assert min(x['relative_error'] for x in sweep)<.03,(env,sweep)
        np.savez_compressed(OUT/(env+'.npz'),images=images,past=past,future=future,encoded=encoded.cpu().numpy(),prediction=own.cpu().numpy(),native_prediction=native.cpu().numpy(),two_step_prediction=own2.cpu().numpy(),gradient=gradient.cpu().numpy())
        records.append({'environment':env,'checkpoint_epoch':model.epoch,'encoded_shape':list(encoded.shape),'prediction_shape':list(own.shape),'native_one_step_max_error':one_error,'native_two_step_max_error':two_error,'repeat_exact':True,'external_proprio_invariant':True,'normalization_exact':True,'gradient_norm':float(gradient.norm()),'directional_derivative_sweep':sweep,'fixture':str(path.relative_to(resolve('.'))),'fixture_sha256':sha256(path),'seconds':time.perf_counter()-start})
        print('AUDITED',env,records[-1],flush=True);del model
    paths=[resolve('scripts/retrained_dinowm_core.py'),Path(__file__)]+list((BASE/'protocol/dinowm_retrained_inventory').glob('*.json'))
    write_json(OUT/'audit.json',{'models':records,'device':'cpu','precision':'FP32; released config uses bf16 on GPU, not reproduced here','restricted_loading':'weights_only=True with fixed reviewed class allowlist, no unrestricted fallback','inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'scope':'Native rollout equivalence for one physical history per task, two-step numerical fixture, normalization, deterministic output, no externalproprio, action gradient/finite differences. No capture/resolution/propagation accuracy or finaltest claim.'})
if __name__=='__main__':main()
