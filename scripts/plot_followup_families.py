import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zj_wam.io import resolve
out=resolve('artifacts/followup_v1');rows=json.loads((out/'family_ablation/analysis.json').read_text())['oracle']
methods=['quintic_full','lewm_low2','cubic_spline_terminal','interior_values_terminal'];labels=['Quintic (4D)','LeWM low (2D)','Cubic spline + terminal constraints','All interior values + terminal constraints'];colors=['#718096','#527cb8','#58a78d','#278266'];horizons=[5,10,20]
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');width=.19
for i,(method,label,color) in enumerate(zip(methods,labels,colors)):
 r=[next(r for r in rows if r['split']=='development' and r['stratum']=='all' and r['horizon']==h and r['method']==method)['joint'] for h in horizons]
 mean=np.array([a['mean'] for a in r])*100;ci=np.array([a['ci95'] for a in r])*100;x=np.arange(3)+(i-1.5)*width
 ax.bar(x,mean,width,label=label,color=color);ax.errorbar(x,mean,yerr=np.stack([mean-ci[:,0],ci[:,1]-mean]),fmt='none',ecolor='#222',capsize=2)
ax.set_xticks(np.arange(3),['H=5 (0.5 s)','H=10 (1 s)','H=20 (2 s)']);ax.set_ylabel('Object preserved + >=10% executed jerk reduction (%)');ax.set_title('Four-radius family oracle: development only, 35 families / 105 states')
ax.legend(loc='upper left',fontsize=8);ax.set_ylim(0,100);ax.spines[['top','right']].set_visible(False)
fig.supxlabel('Different correction spaces and horizons; not deployed selection performance.\nFamily bootstrap 95% intervals.',fontsize=8)
fig.savefig(out/'figures/family_oracle_development.png',dpi=180,bbox_inches='tight');fig.savefig(out/'figures/family_oracle_development.pdf',bbox_inches='tight')
