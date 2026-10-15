import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zj_wam.io import resolve
base=resolve('artifacts/followup_v1');out=base/'figures';out.mkdir(exist_ok=True)
data=json.loads((base/'tolerance_sweep/analysis.json').read_text())['summary'];names={'lewm_global':'LeWM global','vision_contact':'Visual contact','vision_adaptive':'Visual online gains','finite_bank_oracle':'Finite-bank oracle'}
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for method,label in names.items():
 variant='oracle' if method=='finite_bank_oracle' else 'shared_tolerance_input';items=[next(x for x in data if x['split']=='development' and x['method']==method and x['variant']==variant and x['tolerance']==tol) for tol in ['p0.5_r0.5','p1_r1','p2_r2']]
 for ax,metric in zip(axes,['joint','risk']):
  y=np.array([x[metric]['mean'] for x in items])*100;ci=np.array([x[metric]['ci95'] for x in items])*100;ax.errorbar([0,1,2],y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),marker='o',capsize=3,label=label,linestyle='--' if method=='finite_bank_oracle' else '-')
for ax in axes:ax.set_xticks([0,1,2],['1 px / 1°','2 px / 2°','4 px / 4°']);ax.set_xlabel('Position / rotation tolerance; coverage loss ≤ 0.02');ax.grid(alpha=.2)
axes[0].set_ylabel('Preserved + jerk reduction ≥10% (%)');axes[1].set_ylabel('Violation among applied corrections (%)');axes[0].legend(fontsize=8);fig.suptitle('One tolerance-input rule per method • development 35 families\nFamily bootstrap 95% intervals; controller rule had no feasible joint calibration',fontsize=10)
for ext in ['png','pdf']:fig.savefig(out/f'tolerance_sweep_development.{ext}',dpi=180)
plt.close(fig)
data=json.loads((base/'visual_nuisance/analysis.json').read_text())['summary'];names=['background_tint','rgb_noise_5','rgb_noise_15'];items=[next(x for x in data if x['split']=='development' and x['representation']=='projected_cls' and x['readout']=='ridge' and x['contrast']=='appearance_only_nominal' and x['appearance']==name) for name in names]
fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
for ax,key,label in zip(axes,['position_error','rotation_error_deg'],['Apparent object displacement (px)','Apparent object rotation (degrees)']):
 y=np.array([x[key]['mean'] for x in items]);ci=np.array([x[key]['ci95'] for x in items]);ax.bar(np.arange(3),y,color=['#4c78a8','#59a14f','#e15759']);ax.errorbar(np.arange(3),y,yerr=np.maximum(0,np.vstack([y-ci[:,0],ci[:,1]-y])),fmt='none',color='black',capsize=4);ax.axhline(2,color='gray',ls='--',lw=1);ax.set_xticks(np.arange(3),['Background tint','RGB noise 5','RGB noise 15']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
fig.suptitle('Same physical endpoint, changed image • native LeWM + frozen linear probe\n12 development families; diagnostic only, not correction-selection failure',fontsize=10)
for ext in ['png','pdf']:fig.savefig(out/f'visual_nuisance_development.{ext}',dpi=180)
