"""Cube development preservation/jerk and independently evaluated readout diagnostics."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zj_wam.io import resolve

base=resolve('artifacts/followup_v1');cube=base/'cube';out=base/'figures';out.mkdir(exist_ok=True);d=json.loads((cube/'proposal_ablation/analysis.json').read_text());methods=['controller_displacement','proprio_contact','lewm_global','oracle'];labels=['Controller','Contact proxy','LeWM global','Oracle*'];fig,axes=plt.subplots(1,2,figsize=(10,4.5))
for col,key in enumerate(['joint','risk']):
 ax=axes[col]
 for j,(bank,color,label) in enumerate([('common','#286f9e','Common proposals'),('non_wm_proposals','#d58a2e','Without LeWM directions')]):
  rows={r['method']:r for r in d['summary'] if r['split']=='development' and r['bank']==bank};values=np.array([rows[m][key]['mean'] for m in methods])*100;ci=np.array([rows[m][key]['ci95'] for m in methods])*100;x=np.arange(4)+(j-.5)*.22
  ax.errorbar(x,values,yerr=np.array([values-ci[:,0],ci[:,1]-values]),fmt='o',capsize=3,color=color,label=label)
 ax.set_xticks(np.arange(4),labels,rotation=12);ax.set_ylabel('Joint preservation + jerk success U (%)' if col==0 else 'Violations among modified states (%)');ax.grid(axis='y',alpha=.2);ax.set_ylim((-2,50) if col==0 else (-1,20))
 if col==1:ax.axhline(10,ls='--',color='#ab3f3f',lw=1)
axes[0].legend(fontsize=8);fig.suptitle('Cube: useful candidates exist; verifier advantage remains uncertain');fig.text(.5,.015,'96 development states / 24 families; 95% family intervals. *Oracle uses future physics. Proposal counts differ between banks.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.05,1,.94])
for ext in ['png','pdf']:fig.savefig(out/f'cube_development_tradeoff.{ext}',dpi=170)
plt.close(fig)
d=json.loads((cube/'probes_v2/analysis.json').read_text());controls=['real_endpoint','predicted','zero','wrong_action'];labels=['Real endpoints','Predicted effects','Zero effect','Wrong action'];fig,axes=plt.subplots(1,2,figsize=(10,4.4))
for col,(key,scale,title) in enumerate([('position_effect_error_m',1000,'Position-effect error (mm)'),('rotation_effect_error_rad',180/np.pi,'Rotation-effect error (degrees)')]):
 ax=axes[col]
 for j,(name,color,label) in enumerate([('frame__ridge_symmetry','#286f9e','Single frame'),('history__ridge_symmetry','#d58a2e','Three-frame history')]):
  rows={r['control']:r for r in d['effects'] if r['split']=='calibration' and r['readout']==name};values=np.array([rows[m][key]['mean'] for m in controls])*scale;ci=np.array([rows[m][key]['ci95'] for m in controls])*scale;x=np.arange(4)+(j-.5)*.22
  ax.errorbar(x,values,yerr=np.array([values-ci[:,0],ci[:,1]-values]),fmt='o',capsize=3,color=color,label=label)
 ax.set_xticks(np.arange(4),labels,rotation=12);ax.set_ylabel(title);ax.grid(axis='y',alpha=.2);ax.set_ylim(bottom=0)
axes[0].legend(fontsize=8);fig.suptitle('Cube effect diagnostics: symmetry-aware linear probes');fig.text(.5,.015,'Calibration families; probe choices fixed on development. 95% family intervals. Full physical rotation effects remain the target.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.05,1,.94])
for ext in ['png','pdf']:fig.savefig(out/f'cube_probe_effects.{ext}',dpi=170)
print('Created Cube tradeoff and probe figures.')
