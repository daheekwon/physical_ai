"""Paired family contrasts and full-path latency figure for the completed runtime study."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.analyze_followup_response_selection import mean_ci
from zj_wam.io import resolve,write_json

base=resolve('artifacts/followup_v1');directory=base/'runtime';f=pd.read_csv(directory/'evaluation.csv');rows=[]
for split in ['development','calibration']:
    full=f[(f.split==split)&(f.profile=='full')]
    for baseline in ['gaussian_action','controller_pusher','vision_contact']:
        b=full[full.method==baseline]
        for method,g in full.groupby('method'):
            if method==baseline:continue
            p=g.merge(b,on=['state_id','initial_pose_family'],validate='one_to_one',suffixes=('','_base'));assert len(p)==105
            for name in ['joint','jerk_gain','mean_seconds']:
                p['delta']=p[name].astype(float)-p[name+'_base'].astype(float)
                rows.append({'split':split,'method':method,'baseline':baseline,'metric':name,'difference':mean_ci(p,'delta')})
write_json(directory/'contrasts.json',{'rows':rows,'scope':'Same105states and full candidate profile. Family paired differences; observed risks differ, so not exact risk-matched superiority. Gaussian uses broader correction family. Shared GPU/host latency, not isolated production timing.'})
a=json.loads((directory/'analysis.json').read_text());variants=[r for r in a['variants'] if r['split']=='development' and r['profile']=='full'];variants.sort(key=lambda r:r['latency_p95_seconds'])
labels={'controller_pusher':'Controller','gaussian_action':'Gaussian','lewm_global':'LeWM verifier','vision_adaptive':'Online contact','vision_contact':'Fixed contact'}
fig,axes=plt.subplots(1,2,figsize=(10,4.7),layout='constrained');names=[labels[r['method']] for r in variants];x=np.arange(len(names))
axes[0].barh(x,[r['latency_p95_seconds']*1000 for r in variants],color='#4886ad');axes[0].set_yticks(x,names);axes[0].set_xlabel('Measured p95 full-path latency (ms)');axes[0].invert_yaxis()
for i,r in enumerate(variants):axes[0].text(r['latency_p95_seconds']*1000+1,i,f"{r['latency_p95_seconds']*1000:.1f}",va='center',fontsize=9)
axes[0].set_xlim(0,108)
means=np.array([r['joint']['mean']*100 for r in variants]);cis=np.array([r['joint']['ci95'] for r in variants])*100
axes[1].barh(x,means,color='#5b9d79');axes[1].errorbar(means,x,xerr=np.stack([means-cis[:,0],cis[:,1]-means]),fmt='none',ecolor='#222222',capsize=3);axes[1].set_yticks(x,names);axes[1].invert_yaxis();axes[1].set_xlabel('Preservation + executed jerk reduction ≥10% (%)');axes[1].set_xlim(0,50)
for i,r in enumerate(variants):axes[1].text(49,i,f"R {r['risk']['mean']*100:.1f}%",ha='right',va='center',fontsize=8)
fig.suptitle('PushT development: measured cost and useful corrections\n105 states / 35 families; full profile; family bootstrap 95% intervals',fontsize=11)
fig.supxlabel('Gaussian has broader boundary freedom. R: violations among applied corrections. Shared hardware; no actuation delay injected.',fontsize=8)
for ax in axes:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
folder=base/'figures';folder.mkdir(exist_ok=True)
for ext in ['png','pdf']:fig.savefig(folder/f'runtime_development.{ext}',dpi=180)
for r in rows:
    if r['split']=='development' and r['baseline']=='gaussian_action' and r['metric']=='joint':print(r)
