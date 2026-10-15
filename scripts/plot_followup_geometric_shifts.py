"""Compare locked smoothing and model-based selectors under effective shifts."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zj_wam.io import resolve
base=resolve('artifacts/followup_v1');a=json.loads((base/'geometric_shifts/analysis.json').read_text());assert not a['partial']
conditions=['nominal','friction_02','friction_08','gain_08_unknown','gain_12_unknown','rgb_noise_5','rgb_noise_15','friction_02_noise_15']
labels=['Nominal','Friction .2','Friction .8','Gain ×.8','Gain ×1.2','RGB noise 5','RGB noise 15','Friction .2\n+ noise 15']
methods=['gaussian_action','lipo_full_command','groove_full_command','lipo_polynomial_executed','groove_polynomial_executed','controller_pusher','vision_contact','vision_adaptive','lewm_global']
names=['Gaussian','LiPo full command','GROOVE full command','LiPo polynomial','GROOVE polynomial','Controller','Fixed contact','Online contact','LeWM verifier']
lookup={(r['condition'],r['method']):r for r in a['summary'] if r['split']=='development' and r['stratum']=='all'}
fig,axes=plt.subplots(1,2,figsize=(16,6),layout='constrained')
for ax,metric,title in zip(axes,['joint','risk'],['Preservation + jerk reduction ≥10% (%)','Violations among applied corrections (%)']):
    values=np.array([[np.nan if lookup[c,m][metric]['mean'] is None else 100*lookup[c,m][metric]['mean'] for c in conditions] for m in methods]);cmap=plt.get_cmap('YlGnBu' if metric=='joint' else 'YlOrRd').copy();cmap.set_bad('#dddddd');ax.imshow(values,cmap=cmap,vmin=0,vmax=40 if metric=='joint' else 30,aspect='auto')
    for i in range(len(methods)):
        for j in range(len(conditions)):
            v=values[i,j];ax.text(j,i,'—' if np.isnan(v) else f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if v>(25 if metric=='joint' else 18) else 'black')
    ax.set_xticks(np.arange(len(conditions)),labels,rotation=40,ha='right',fontsize=8);ax.set_yticks(np.arange(len(methods)),names,fontsize=9);ax.set_title(title,fontsize=11)
fig.suptitle('PushT development: frozen settings under physical and sensing shifts\n105 states / 35 families; every condition uses its own paired nominal reference',fontsize=12)
fig.supxlabel('Full-command methods have broader correction freedom. Proprioceptive filters/geometry do not use RGB. — means no applied correction.',fontsize=9)
for ext in ['png','pdf']:fig.savefig(base/f'figures/geometric_shifts_development.{ext}',dpi=180)
