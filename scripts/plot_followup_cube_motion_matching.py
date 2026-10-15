"""Visualize unequal realized robot responses at equal action norm."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from zj_wam.io import resolve

p=resolve('artifacts/followup_v1/cube/motion_matching');f=pd.read_csv(p/'candidate_pairs.csv');f=f[(f.split=='development')&(f.direction=='lewm_low')]
passed=(f.amplitude_mismatch<=.2)&(f.frequency_distance<=.2)
fig,axes=plt.subplots(1,2,figsize=(10.8,4.6))
colors={'approach':'#4477aa','grasp':'#ee7733','transport':'#228833','release':'#aa3377'}
for ax,key,factor,label in [(axes[0],'translation_rms',1000.,'EEF translation RMS deviation (mm)'),(axes[1],'rotation_rms',180/np.pi,'EEF rotation RMS deviation (deg)')]:
    x=f[key+'_right']*factor;y=f[key+'_left']*factor;assert (x>0).all() and (y>0).all()
    low=min(x.min(),y.min())*.7;high=max(x.max(),y.max())*1.4;grid=np.geomspace(low,high,200)
    ax.fill_between(grid,.8*grid,1.25*grid,color='#88bb88',alpha=.22,label='20% size caliper')
    ax.plot(grid,grid,color='0.35',linewidth=.8)
    for phase,color in colors.items():
        keep=f.phase==phase;ax.scatter(x[keep],y[keep],s=10,alpha=.35,c=color,label=phase,rasterized=True)
    ax.scatter(x[passed],y[passed],s=70,facecolors='none',edgecolors='black',linewidths=1.2,label='All size + frequency criteria')
    ax.set(xscale='log',yscale='log',xlim=(low,high),ylim=(low,high),xlabel='Random direction: '+label,ylabel='LeWM low: '+label)
    ax.grid(alpha=.15)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.04),ncol=3,frameon=False,fontsize=8)
fig.suptitle('Equal action norm does not match realized EEF motion',fontsize=12)
fig.text(.5,.005,'Development: 1,320 pairs in 68 states / 23 families; only 4 pairs in 2 families pass all 20% criteria.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.17,1,.96])
for suffix in ['png','pdf']:fig.savefig(p/('realized_motion_comparison.'+suffix),dpi=180,bbox_inches='tight')
print('WROTE',p/'realized_motion_comparison.png')
