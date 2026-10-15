"""Development outcome/risk plot with episode-family uncertainty."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zj_wam.io import resolve

base=resolve('artifacts/followup_v1');d=json.loads((base/'cumulative_allowance/analysis.json').read_text());methods=['local','constant_shrink','remaining_margin','reference_forecast'];labels=['Local rule','Fixed shrink','Remaining margin','Reference forecast'];fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
for col,tier in enumerate(['vision','state']):
 rows={r['method']:r for r in d['summary'] if r['split']=='development' and r['tier']==tier}
 for row,keys in enumerate([('joint','long_joint'),('risk','long_risk')]):
  ax=axes[row,col]
  for j,(key,color,label) in enumerate(zip(keys,['#206b9b','#d77d26'],['20 actions + 3 suffix','20 actions + up to 10 suffix'])):
   values=np.array([rows[m][key]['mean'] for m in methods])*100;ci=np.array([rows[m][key]['ci95'] for m in methods])*100;x=np.arange(4)+(j-.5)*.18
   ax.errorbar(x,values,yerr=np.vstack([values-ci[:,0],ci[:,1]-values]),fmt='o',color=color,capsize=3,label=label)
  ax.set_xticks(np.arange(4),labels,rotation=18,ha='right');ax.grid(axis='y',alpha=.2);ax.set_ylim((-2,35) if row==0 else (-2,103));ax.set_ylabel('Joint success U (%)' if row==0 else 'Violation rate among modified states (%)')
  if row==1:ax.axhline(10,color='#ad3333',ls='--',lw=1,label='Calibration risk target')
  if row==0:ax.set_title('Image-based pose' if tier=='vision' else 'True pose (privileged diagnostic)')
axes[0,0].legend(fontsize=8);fig.suptitle('Cumulative allowance: lower deviation, little useful smoothing',fontsize=13);fig.text(.5,.012,'U: preservation + >=10% jerk reduction. 105 states / 35 families; 95% family intervals. New rules failed calibration qualification.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.045,1,.96]);out=base/'figures';out.mkdir(exist_ok=True)
for extension in ['png','pdf']:fig.savefig(out/f'cumulative_allowance_development.{extension}',dpi=170)
print(out/'cumulative_allowance_development.png')
