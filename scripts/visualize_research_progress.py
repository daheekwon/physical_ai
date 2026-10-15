"""Self-contained report figures; conceptual diagrams contain no synthetic measurements."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets/research_progress'
OUT.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont('/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf')
plt.rcParams.update({'font.family':'UnDotum','font.size':11,'axes.unicode_minus':False,'savefig.facecolor':'white'})
blue='#286c96';orange='#bd773f';gray='#53616b'
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,text,color=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor='#f3f7fa',edgecolor=color,lw=1.1))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color='#23343f')
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=gray,lw=1.2))
# Use a recorded selected correction, not an invented smoothing example.
example=ROOT/'artifacts/followup_v1/sequence_oracle/210005.npz'
meta_path=example.with_suffix('.json');meta=json.loads(meta_path.read_text())
with np.load(example) as z:
 coef=z['coefficients'][meta['best_index']]
 block=int(np.argmax(np.linalg.norm(coef,axis=1)));offset=int(z['offsets'][block])
 delta=(z['basis']@coef[block]).reshape(5,2)*z['scale']*100
 axis=int(np.argmax(np.linalg.norm(delta,axis=0)))
 original=z['nominal'][offset:offset+5,axis]*100
 correction=delta[:,axis];modified=original+correction
u=np.linspace(0,1,5);envelope=u**2*(1-u)**2
basis=np.column_stack([envelope,envelope*(2*u-1)])
c=np.linalg.lstsq(basis,correction,rcond=None)[0]
np.testing.assert_allclose(basis@c,correction,atol=1e-10,rtol=0)
v=np.linspace(0,1,200);curve=v**2*(1-v)**2*(c[0]+c[1]*(2*v-1))
fig,axes=plt.subplots(1,3,figsize=(10,2.6),sharey=True)
fig.subplots_adjust(left=.07,right=.98,bottom=.22,top=.84,wspace=.36)
x=np.arange(1,6)
for ax,values,title,color in zip(axes,[original,correction,modified],['원래 명령','다항식 수정량','수정 명령'],[blue,orange,'#347969']):
 ax.axhline(0,color='#c4cbd0',lw=.7)
 ax.vlines(x,0,values,color=color,lw=1.4);ax.scatter(x,values,color=color,s=28,zorder=3)
 ax.set_xticks(x);ax.set(xlim=(.7,5.3),ylim=(-20,30),xlabel='명령 시점',title=title)
 ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Y축 상대 목표 (px)')
axes[1].plot(1+4*v,curve,color=orange,lw=1.2)
axes[2].scatter(x,original,facecolors='none',edgecolors=blue,s=35,label='원래 명령')
axes[2].legend(frameon=False,fontsize=8,loc='upper right')
fig.text(.354,.52,'+',fontsize=22,ha='center');fig.text(.684,.52,'=',fontsize=22,ha='center')
save(fig,'01_polynomial_addition')
fig,ax=plt.subplots(figsize=(10,2.75));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
rows=[(.79,'상태 읽기','실제 이미지','Encode → probe','실제 위치·회전'),(.48,'실제 변화 구분','실제로 실행한\n원래 / 수정 결과 이미지','각각 물체 상태 읽기\n→ 차이 계산','실제 물체 변화'),(.17,'미래 효과 예측','현재 이력\n+ 원래 / 수정 행동','미래 예측 → 상태 읽기\n→ 차이 계산','실제 물체 변화')]
for y,label,inp,process,truth in rows:
 ax.text(.01,y,label,va='center',fontsize=11,weight='bold')
 box(ax,.19,y-.10,.22,.20,inp);box(ax,.46,y-.10,.29,.20,process)
 arrow(ax,(.415,y),(.45,y));arrow(ax,(.755,y),(.80,y))
 ax.text(.895,y,truth+'\n와 오차 비교',ha='center',va='center',fontsize=10)
ax.text(.01,.985,'세 단계의 입력과 비교 대상 · 개념도',fontsize=10,color=gray,va='top')
save(fig,'02_probe_diagnostics')
path=ROOT/'artifacts/followup_v1/oracle_factorial/analysis.json';data=json.loads(path.read_text())
fig,axes=plt.subplots(1,2,figsize=(9.8,2.9),layout='constrained')
matrices={}
for ax,env,label in zip(axes,['pusht','ogb-cube'],['PushT 개발 105상태','Cube 개발 96상태']):
 rows={r['cell']:r for r in data['summary'] if r['environment']==env and r['split']=='development' and r['method']=='lewm_global'}
 m=np.array([[rows[c]['joint']['mean']*100 for c in 'AB'],[rows[c]['joint']['mean']*100 for c in 'CD']]);matrices[env]=m.tolist()
 ax.imshow(m,cmap='Blues',vmin=0,vmax=35,aspect='auto')
 for (i,j),v in np.ndenumerate(m):ax.text(j,i,f'{v:.2f}%',ha='center',va='center',color='white' if v>23 else '#152d3a',fontsize=16,weight='bold')
 ax.set_xticks([0,1],['예측 jerk 순위','실제 jerk 순위']);ax.set_yticks([0,1],['LeWM 보존 판정','실제 보존 판정']);ax.set_title(label,fontsize=12)
 ax.tick_params(length=0);ax.spines[:].set_visible(False)
fig.suptitle('보존과 jerk 감소를 함께 달성한 비율 U',fontsize=13)
save(fig,'03_selection_factorial')
seqpath=ROOT/'artifacts/followup_v1/sequence_oracle/analysis.json';seq=json.loads(seqpath.read_text());states=seq['states']
preserve=np.array([r['best']['jerk_gain']*100 for r in states]);free=np.array([r['best_ignoring_object_constraint']['jerk_gain']*100 for r in states]);assert sum(free>=10)==9 and sum(preserve>=10)==0
fig,ax=plt.subplots(figsize=(10,2.9),layout='constrained');x=np.arange(len(states));w=.36
ax.bar(x-w/2,preserve,w,color=blue,label='물체 보존 조건 적용')
ax.bar(x+w/2,free,w,color=orange,label='물체 보존 조건 없음')
ax.axhline(10,color='#a73d48',ls='--',lw=1.2,label='목표: 전체 jerk 10% 감소')
ax.set_xticks(x,[str(i+1) for i in x]);ax.set(xlabel='기존 개발 family (1–12)',ylabel='전체 30초 jerk 감소 (%)',ylim=(0,24))
ax.legend(fontsize=9,ncol=2,loc='upper left',frameon=False);ax.spines[['top','right']].set_visible(False)
save(fig,'04_sequence_oracle')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'figure_evidence.json').write_text(json.dumps({'conceptual_figures':['02_probe_diagnostics'],'measured_figures':['01_polynomial_addition','03_selection_factorial','04_sequence_oracle'],'inputs':{str(p.relative_to(ROOT)):h(p) for p in [path,seqpath,example,meta_path]},'polynomial_example':{'seed':210005,'selected_index':meta['best_index'],'block_index':block,'action_offset':offset,'axis':axis,'original_px':original.tolist(),'correction_px':correction.tolist(),'modified_px':modified.tolist()},'matrix_values_percent':matrices,'oracle_values_percent':{'preserving':preserve.tolist(),'ignoring_object':free.tolist()},'script_sha256':h(Path(__file__))},ensure_ascii=False,indent=2)+'\n')
print('Generated 4 figures; measured values read from audited analyses.')
