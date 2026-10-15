"""Budget-matched Cube comparison with explicitly model-free baseline paths."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_response_selection import mean_ci

BASE=resolve('artifacts/followup_v1/cube');read=lambda p:json.loads(p.read_text())

def freeze(smoke):
    dest=BASE/('runtime_pure_smoke' if smoke else 'runtime_pure');dest.mkdir(exist_ok=True)
    paths=[resolve(p) for p in ['configs/followup_cube_runtime_pure.json','configs/followup_cube_runtime.json','configs/followup_cube.json','scripts/analyze_followup_cube_runtime_pure.py','scripts/analyze_followup_response_selection.py']]
    m={'smoke':smoke,'config':read(paths[0]),'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths}}
    if (dest/'manifest.json').exists():assert read(dest/'manifest.json')==m,'Pure pipeline definition changed'
    else:write_json(dest/'manifest.json',m)
    return dest,m


def analyze(smoke,dest,manifest):
    src=BASE/('runtime_smoke' if smoke else 'runtime');raw=read(src/'raw_audit.json');assert raw['smoke']==smoke and raw['states']==(2 if smoke else 192)
    original=read(src/'manifest.json');assert raw['manifest_sha256']==sha256(src/'manifest.json')
    for p,h in original['inputs'].items():assert sha256(resolve(p))==h,p
    for p,h in manifest['inputs'].items():assert sha256(resolve(p))==h,p
    f=pd.read_csv(src/'evaluation.csv');t=pd.read_csv(src/'timings.csv');cfg=original['cube_config'];rcfg=original['config'];summary=read(src/'analysis.json')['summary'];settings=[]
    assert len(f)==raw['outcomes'] and len(t)==raw['timings']
    for name,pipeline in manifest['config']['pipelines'].items():
        profiles=pipeline['profiles'];method=pipeline['method']
        if name.endswith('_no_wm'):
            rows=t[(t.method==method)&t.profile.isin(profiles)]
            assert len(rows)==raw['states']*len(profiles)*rcfg['repetitions']
            assert not rows[['encoder_calls','jacobian_calls','finite_prediction_calls']].to_numpy().any()
        for cap in rcfg['budgets_seconds']:
            options=[r for r in summary if r['split']=='calibration' and r['method']==method and r['profile'] in profiles and r['p95_seconds']<=cap and r['risk']['mean'] is not None and r['contact_risk']['mean'] is not None and max(r['risk']['mean'],r['contact_risk']['mean'])<=cfg['target_risk'] and r['active']['mean']>=cfg['minimum_active_fraction']]
            chosen=max(options,key=lambda r:(r['joint']['mean'],r['jerk_gain']['mean'],-r['p95_seconds'])) if options else None
            settings.append({'pipeline':name,'method':method,'cap_seconds':cap,'enabled':chosen is not None,'profile':chosen['profile'] if chosen else None,'calibration':chosen})
    write_json(dest/'calibrated_settings.json',{'settings':settings,'smoke':smoke,'scope':'Calibration only. Contact-free implementation smoke cannot qualify profiles; no performance conclusion.'})
    rows=[]
    for setting in settings:
        method=setting['method'];profile=setting['profile'];g=f[(f.split=='development')&(f.method==method)&(f.profile==(profile or 'nonwm_full'))].copy()
        g['pipeline']=setting['pipeline'];g['cap_seconds']=setting['cap_seconds'];g['enabled']=setting['enabled']
        if profile is None:
            g['candidate']=0;g['active']=False;g['joint']=False;g['violation']=False;g['object_error']=0.;g['jerk_gain']=0.;g['norm']=0.;g['seconds']=0.;g['profile']='disabled';g['deadline_exceedance']=0.
        else:
            times=t[(t.split=='development')&(t.method==method)&(t.profile==profile)].copy();times['miss']=times.seconds>setting['cap_seconds'];g['deadline_exceedance']=g.state_id.map(times.groupby('state_id').miss.mean())
        rows.append(g)
    evaluation=pd.concat(rows,ignore_index=True);output=[];paired=[]
    for (pipeline,cap),g in evaluation.groupby(['pipeline','cap_seconds']):
        selected=g[g.active];output.append({'pipeline':pipeline,'cap_seconds':cap,'enabled':bool(g.enabled.iloc[0]),'profile':g.profile.iloc[0], **{key:mean_ci(g,key) for key in ['joint','active','jerk_gain','deadline_exceedance']},'risk':mean_ci(selected,'violation')})
    for cap,g in evaluation.groupby('cap_seconds'):
        baseline=g[g.pipeline==manifest['config']['primary_baseline']]
        for pipeline in manifest['config']['pipelines']:
            if pipeline==manifest['config']['primary_baseline']:continue
            x=g[g.pipeline==pipeline].merge(baseline,on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one');x['delta_U']=x.joint.astype(float)-x.joint_base.astype(float);x['delta_gain']=x.jerk_gain-x.jerk_gain_base
            paired.append({'pipeline':pipeline,'baseline':manifest['config']['primary_baseline'],'cap_seconds':cap,'U_difference':mean_ci(x,'delta_U'),'jerk_gain_difference':mean_ci(x,'delta_gain')})
    evaluation.to_csv(dest/'evaluation.csv',index=False)
    write_json(dest/'analysis.json',{'smoke':smoke,'summary':output,'paired':paired,'scope':manifest['config']['scope']})
    write_json(dest/'raw_audit.json',{'smoke':smoke,'parent_states':raw['states'],'parent_outcomes':len(f),'parent_timings':len(t),'zero_model_call_baselines':['controller_no_wm','contact_no_wm'],'manifest_sha256':sha256(dest/'manifest.json'),'inputs':{name:sha256(src/name) for name in ['manifest.json','raw_audit.json','analysis.json','evaluation.csv','timings.csv']},'budget_rows':len(evaluation),'scope':'Eligibility locked separately before full measurements; pure baseline model calls asserted from all three measured repetitions. Original parent action/physics audit inherited unchanged.'})
    print('AUDITED pure Cube budget comparison',raw['states'],'parent states',len(evaluation),'budget rows',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--dependency-pid',type=int);args=p.parse_args();dest,m=freeze(args.smoke)
    if args.dependency_pid:
        assert not args.smoke;start=time.monotonic();print('WAITING for full Cube runtime audit; pure pipeline eligibility locked',flush=True)
        while not (BASE/'runtime/raw_audit.json').exists():
            try:os.kill(args.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError('Runtime pipeline ended before full audit; inspect logs')
            if time.monotonic()-start>21600:raise RuntimeError('Observation window elapsed; revalidate process')
            time.sleep(10)
    analyze(args.smoke,dest,m)
if __name__=='__main__':main()
