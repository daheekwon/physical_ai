import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zj_wam.io import resolve
out=resolve('artifacts/followup_v1');data=json.loads((out/'selection/evaluation_summary.json').read_text())
methods=['controller_pusher','lewm_global','lewm_object','vision_contact','vision_adaptive'];names=['Controller','LeWM latent','LeWM + pose readout','Vision + contact','Vision + online gains']
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for ax,metric,title in zip(axes,['joint','conditional_violation'],['Preserved object trajectory + >=10% jerk reduction','Violation among applied corrections']):
 rows=[next(r for r in data['selection'] if r['split']=='development' and r['stratum']=='all' and r['rule_source']=='selector' and r['method']==m)[metric] for m in methods]
 values=np.array([r['mean'] for r in rows])*100;bounds=np.array([r['ci95'] for r in rows])*100
 ax.barh(names,values,color=['#718096','#527cb8','#315d9e','#58a78d','#278266']);ax.errorbar(values,np.arange(len(names)),xerr=np.stack([values-bounds[:,0],bounds[:,1]-values]),fmt='none',ecolor='#222222',capsize=3)
 ax.invert_yaxis();ax.set_xlabel('Percent (family bootstrap 95% interval)');ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Development diagnostics: fixed common candidate bank, 35 families / 105 states',fontsize=11)
d=out/'figures';d.mkdir(exist_ok=True);fig.savefig(d/'selection_development.png',dpi=180);fig.savefig(d/'selection_development.pdf');print(d/'selection_development.png')
