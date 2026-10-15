"""Separate real executed object effects from image-only changes with frozen readouts."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch,joblib
from zj_wam.adapters import ExpertDataset,FrozenLeWM
from zj_wam.model_audit import encode_representations
from zj_wam.readouts import decode,effects,errors
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_response_selection import mean_ci


def transform(images,name,noise,tint):
    x=images.copy()
    if name=='background_tint':x[np.all(x==255,axis=-1)]=tint
    elif name.startswith('rgb_noise_'):x=np.clip(np.rint(x.astype(float)+float(name.rsplit('_',1)[-1])*noise[None]),0,255).astype(np.uint8)
    return x


def process(path,cfg,ncfg,model,readouts,out):
    key=path.stem;done=out/f'{key}.json'
    if done.exists():return
    start=time.perf_counter();frame=pd.read_csv(path.with_suffix('.csv'));meta=json.loads(path.read_text());keep=frame[frame.method.isin(ncfg['candidate_methods'])].sort_values('index');idx=np.r_[0,keep['index'].to_numpy(int)]
    with np.load(path.with_suffix('.npz')) as a:
        images=a['endpoint_images'][idx,0].copy();truth=a['control'][idx,7,2:5].copy();actions=a['actions'][idx].copy()
    noise=np.random.default_rng(np.random.SeedSequence([cfg['seed'],meta['episode'],meta['timestep'],404])).normal(size=images.shape[1:]);views=np.stack([transform(images,name,noise,ncfg['background_tint']) for name in ncfg['appearances']]);reps=encode_representations(model,views.reshape(-1,224,224,3));reps={k:v.reshape(len(views),len(idx),-1) for k,v in reps.items() if k in ncfg['representations']}
    rows=[];saved={'truth':truth,'indices':idx,'actions':actions,'clean_images':images,'noise':noise.astype(np.float32)};delta=effects(truth)
    for rep,features in reps.items():
        saved['features_'+rep]=features
        for kind in ncfg['readouts']:
            poses=decode(readouts[rep,kind],features);saved[f'pose_{rep}_{kind}']=poses
            for j,appearance in enumerate(ncfg['appearances']):
                contrasts=[('physical_pair',features[j]-features[j,:1],effects(poses[j]),delta,range(1,len(idx))),('appearance_only_nominal',features[j]-features[0],effects(np.stack([poses[0,0],poses[j,0]]))[1:2],np.zeros((1,3)),[0]),('combined',features[j]-features[0,:1],poses[j]-poses[0,:1],delta,range(1,len(idx)))]
                for contrast,zchange,pchange,target,indices in contrasts:
                    if contrast=='appearance_only_nominal':zchange=zchange[:1]
                    pchange=pchange.copy();pchange[:,2]=np.arctan2(np.sin(pchange[:,2]),np.cos(pchange[:,2]));pe,re=errors(pchange,target)
                    for i in indices:
                        k=0 if contrast=='appearance_only_nominal' else i
                        rows.append({'state_id':meta['state_id'],'split':meta['split'],'initial_pose_family':meta['initial_pose_family'],'nominal_contact':bool(frame.nominal_contact.iloc[0]),'index':int(idx[i]),'appearance':appearance,'representation':rep,'readout':kind,'contrast':contrast,'latent_distance':float(np.linalg.norm(zchange[k])),'position_error':float(pe[k]),'rotation_error_deg':float(np.rad2deg(re[k])),'true_position_effect':float(np.linalg.norm(target[k,:2])),'true_rotation_effect_deg':float(np.rad2deg(abs(target[k,2]))),'within_2px_2deg':bool(pe[k]<=2 and re[k]<=np.deg2rad(2))})
    pd.DataFrame(rows).to_csv(done.with_suffix('.csv'),index=False);np.savez_compressed(done.with_suffix('.npz'),**saved)
    write_json(done,{'source':str(path),'source_sha256':sha256(path),'source_npz_sha256':sha256(path.with_suffix('.npz')),'state_id':meta['state_id'],'split':meta['split'],'initial_pose_family':meta['initial_pose_family'],'candidates':len(idx)-1,'source_code_sha256':sha256(Path(__file__)),'seconds':time.perf_counter()-start});print(key,'candidates',len(idx)-1,'seconds',round(time.perf_counter()-start,2),flush=True)


def analyze(out,ncfg):
    files=sorted(out.glob('*_f*_e*_t*_h05.json'));assert len(files)==2*ncfg['families_per_split'];f=pd.concat([pd.read_csv(p.with_suffix('.csv')) for p in files],ignore_index=True)
    # Deterministic clean identity must have exactly zero nuisance effect.
    identity=f[(f.appearance=='clean')&(f.contrast=='appearance_only_nominal')];assert identity[['latent_distance','position_error','rotation_error_deg']].to_numpy().max()==0
    for p in files:
        with np.load(p.with_suffix('.npz')) as a:
            m=json.loads(p.read_text());source=Path(m['source'])
            with np.load(source.with_suffix('.npz')) as b:
                np.testing.assert_array_equal(a['truth'],b['control'][a['indices'],7,2:5]);np.testing.assert_array_equal(a['clean_images'],b['endpoint_images'][a['indices'],0])
    # Equal family/state contribution; do not count eight correlated candidates as families.
    group=['state_id','initial_pose_family','split','appearance','representation','readout','contrast'];metrics=['latent_distance','position_error','rotation_error_deg','within_2px_2deg'];states=f.groupby(group)[metrics].mean().reset_index();summary=[]
    for keys,g in states.groupby(['split','appearance','representation','readout','contrast']):summary.append({**dict(zip(['split','appearance','representation','readout','contrast'],keys)),**{m:mean_ci(g,m) for m in metrics}})
    paired=[]
    physical=states[states.contrast=='physical_pair']
    for keys,g in physical.groupby(['split','representation','readout']):
        clean=g[g.appearance=='clean']
        for appearance in ncfg['appearances'][1:]:
            x=g[g.appearance==appearance].merge(clean,on='state_id',suffixes=('','_clean'))
            for m in metrics:x[m+'_difference']=x[m]-x[m+'_clean']
            paired.append({**dict(zip(['split','representation','readout'],keys)),'appearance':appearance,**{m:mean_ci(x,m+'_difference') for m in metrics}})
    states.to_csv(out/'state_metrics.csv',index=False);write_json(out/'analysis.json',{'config':ncfg,'states':len(files),'families_per_split':ncfg['families_per_split'],'rows':len(f),'summary':summary,'paired_physical_error_changes':paired});write_json(out/'raw_audit.json',{'states':len(files),'identity_error':0.,'verified':'Physical labels and clean images exactly equal saved executed endpoints. Clean nominal identity effects zero. Input transforms alter only observations; no new physics. All readouts frozen, source hashes saved.'})


def main():
    cfg=json.loads(resolve('configs/followup_v1.json').read_text());ncfg=json.loads(resolve('configs/followup_visual_nuisance.json').read_text());base=resolve(cfg['output']);out=base/'visual_nuisance';out.mkdir(exist_ok=True);rng=np.random.default_rng(cfg['seed']+404);paths=[]
    for split in ncfg['splits']:
        groups={}
        for path in sorted((base/'f1').glob(split+'*h05.json')):
            m=json.loads(path.read_text());groups.setdefault(m['initial_pose_family'],[]).append(path)
        for family in sorted(rng.choice(sorted(groups),ncfg['families_per_split'],replace=False)):
            available=groups[family];paths.append(available[int(rng.integers(len(available)))])
    torch.set_num_threads(cfg['threads']);data=ExpertDataset(cfg['dataset']);model=FrozenLeWM(cfg['checkpoint'],data.scaler,cfg['device']);readouts={};hashes={}
    for rep in ncfg['representations']:
        for kind in ncfg['readouts']:
            p=base/'lewm_audit/readouts'/f'{rep}_{kind}.joblib';readouts[rep,kind]=joblib.load(p);hashes[f'{rep}_{kind}']=sha256(p)
    write_json(out/'manifest.json',{'config':ncfg,'selected_source_paths':[str(p) for p in paths],'readout_hashes':hashes,'source_sha256':sha256(Path(__file__)),'config_sha256':sha256(resolve('configs/followup_visual_nuisance.json'))})
    for path in paths:process(path,cfg,ncfg,model,readouts,out)
    analyze(out,ncfg)
if __name__=='__main__':main()
