"""Development comparisons; scientific figure with explicit information/family limits."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zj_wam.io import resolve
out=resolve('artifacts/followup_v1');fig,axes=plt.subplots(1,3,figsize=(16,5.3),gridspec_kw={'width_ratios':[1.25,1.2,1]})
def load(name):return json.loads((out/(name+'.json')).read_text())
def rows(name):return {x['method']:x for x in load(name)['summary'] if x['split']=='development' and x['stratum']=='all'}
a=rows('strong_baselines/analysis');b=rows('repeated_refinement/analysis');c=rows('relinearization/analysis')
names=[('gaussian_action','Gaussian / full actions'),('groove_full_command','GROOVE adapter / full'),('lipo_full_command','LiPo adapter / full'),('lipo_polynomial_executed','LiPo adapter / polynomial'),('nonlinear_wm','Nonlinear LeWM / polynomial'),('soft_sensitivity','Soft LeWM / polynomial')]
for i,(key,label) in enumerate(names):
 r=a[key]['joint'];mu=100*r['mean'];lo,hi=np.array(r['ci95'])*100;color='#168579' if i<3 else '#476db2'
 axes[0].errorbar(mu,i,xerr=[[mu-lo],[hi-mu]],fmt='o',color=color,capsize=3);axes[0].text(43,i,f"R={100*a[key]['risk']['mean']:.1f}%",va='center',fontsize=8)
axes[0].set_yticks(range(len(names)),[label for _,label in names]);axes[0].invert_yaxis();axes[0].set_xlim(-1,59);axes[0].set_xlabel('Joint preservation + jerk reduction U (%)');axes[0].set_title('Strong baselines: 105 states')
names=[('open20_piecewise_vision','One-shot / piecewise'),('repeat5_lewm','Repeat / LeWM'),('repeat5_observed_pose','Repeat / visual contact'),('repeat5_online_gains','Repeat / online gains')]
for i,(key,label) in enumerate(names):
 for offset,metric,color in [(-.13,'risk','#426ba7'),(.13,'long_risk','#c65c3f')]:
  r=b[key][metric];mu=100*r['mean'];lo,hi=np.array(r['ci95'])*100
  axes[1].errorbar(mu,i+offset,xerr=[[mu-lo],[hi-mu]],fmt='o',color=color,capsize=3,label='Core suffix 3 actions' if i==0 and metric=='risk' else 'Long suffix up to 10' if i==0 else None)
axes[1].set_yticks(range(len(names)),[label for _,label in names]);axes[1].invert_yaxis();axes[1].set_xlim(0,104);axes[1].set_xlabel('Violation among active corrections (%)');axes[1].set_title('Repeated expert: accumulated error');axes[1].legend(loc='lower left',fontsize=8)
for i,(key,label) in enumerate([('low_frozen','Once'),('low_refresh10','Every 10 actions'),('low_refresh5','Every 5 actions')]):
 r=c[key]['joint'];mu=100*r['mean'];lo,hi=np.array(r['ci95'])*100
 axes[2].errorbar(i,mu,yerr=[[mu-lo],[hi-mu]],fmt='o',capsize=4,color='#476db2');axes[2].text(i,mu+1,f'{mu:.1f}%',ha='center',fontsize=9)
axes[2].set_xticks(range(3),['Once','Every 10','Every 5']);axes[2].set_ylim(0,23);axes[2].set_xlim(-.35,2.35);axes[2].set_xlabel('Jacobian refresh interval (actions)');axes[2].set_ylabel('Joint rate U (%)');axes[2].set_title('Matched rank 2: 20-action execution')
for ax in axes:ax.grid(alpha=.2,axis='x' if ax is not axes[2] else 'y');ax.set_axisbelow(True)
fig.suptitle('Development diagnostics: strong baselines, accumulated drift, and relinearization',fontsize=13)
fig.text(.5,.012,'95% family bootstrap intervals; final test remains locked. Left: correction families differ. Middle: reused H5 rules, no recalibration. Right: common rank and verifier.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.055,1,.95]);dest=out/'figures';dest.mkdir(exist_ok=True)
for ext in ['png','pdf']:fig.savefig(dest/f'streaming_strong_development.{ext}',dpi=180)
print(dest/'streaming_strong_development.png')
