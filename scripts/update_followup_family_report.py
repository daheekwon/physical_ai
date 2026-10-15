import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,write_json
out=resolve('artifacts/followup_v1');audit=json.loads((out/'family_ablation/raw_audit.json').read_text());analysis=json.loads((out/'family_ablation/analysis.json').read_text())
def rate(h,name):return next(r['joint']['mean'] for r in analysis['oracle'] if r['split']=='development' and r['horizon']==h and r['stratum']=='all' and r['method']==name)
lines=['<!-- BEGIN FAMILY UPDATE -->','## F1 수정 자유도·실제 경계와 F9 방향 생성 비교','',
'개발 H=5·10·20 및 calibration H=5에서 각각 105개 상태, 총 420개 상태의 18,480개 후보를 실행했다. 상태마다 각 family 또는 subspace에서 같은 4개 radius의 controller-aware optimizer를 실행했다. 아래 oracle은 각 family의 4개 후보 중 실제 물체 궤적을 보존하면서 jerk를 가장 많이 줄이는 후보를 사후 선택한 진단이다. 배포 가능한 선택기의 성능이나 전역 최적해가 아니다.','',
'| Correction 공간 | H=5 oracle U | H=10 oracle U | H=20 oracle U |','| --- | --- | --- | --- |']
names={'quintic_full':'기존 quintic 4차원','lewm_low2':'Quintic 안 LeWM low 2차원','controller_low2':'Quintic 안 controller low 2차원','cubic_spline':'Cubic spline 자유도 확장','cubic_spline_terminal':'Cubic spline + 실제 종료 위치·속도 고정','interior_values':'내부 제어 시점 correction 전체','interior_values_terminal':'내부 correction 전체 + 실제 종료 위치·속도 고정'}
for name,label in names.items():lines.append('| '+label+' | '+' | '.join(f'{100*rate(h,name):.1f}%' for h in [5,10,20])+' |')
lines+=['',
'**긴 구간의 실패는 수정 공간과 경계 조건의 영향도 크다.** H=20 접촉 상태만 보아도 기존 quintic oracle U는 3.2%, 실제 종료 위치·속도를 고정한 cubic spline은 24.2%, 내부 correction 전체는 69.5%였다. 마지막 방법의 자유도는 32이며 기존 4차원과 직접 같은 표현이라고 볼 수 없다. 고차 polynomial 보간이 가능한 제어 시점 값을 최적화한 것으로, 제어 시점 사이의 연속 polynomial 실행까지 검증한 것은 아니다. 더 제한된 cubic spline에서도 개선이 있어, 자유도 확장 효과를 이 차이와 함께 해석한다.','',
'기존 quintic 4차원에 실제 pusher 종료 위치·속도 네 개 제약을 정확히 추가하면 남는 correction 차원이 0이었다. 다항식 자체의 endpoint 값·미분 조건과 실제 제어기 출력의 종료 조건은 다르다. 실제 경계를 유지하려면 수정 공간도 함께 설계해야 한다.','',
'H=5에서는 cubic spline과 내부 시점 correction이 같은 6차원 표본 공간을 이뤄 성능도 동일했다. 이를 서로 독립적인 두 성공으로 세지 않는다. H=10·20에서 각 공간의 차원이 달라지는 점은 artifact에 기록했다.','',
'### 같은 차원의 방향 생성 비교','',
'H=5의 2차원 LeWM low oracle U는 12.4%, controller low는 13.3%, 무작위 2차원 공간 세 반복은 각각 13.3%·12.4%·13.3%였다. 같은 차원에서 LeWM만의 이점은 아직 드러나지 않았다. 반면 전체 4차원 quintic은 32.4%여서, 저민감도 방향만 유지하면 jerk 감소에 쓸 방향을 버릴 수 있다. H=20에서는 LeWM low가 일부 비교보다 높은 점추정치를 보이므로 H=5 결과를 모든 horizon의 동일성으로 확장하지 않는다.','',
'이미지 접촉 verifier의 기존 규칙을 고정하고 H=5 후보 생성만 바꾸면 실제 U는 quintic 전체 25.7%, LeWM low와 controller low 각각 12.4%, 확장 6차원 공간 37.1%였다. 확장 공간의 선택 후 위반율은 1.2%였다. 이는 development 진단이며 새 후보 분포가 이전 calibration과 다르다. 최종 배포 규칙은 선택 분포를 포함해 잠근 뒤 test해야 한다.','',
'원시 물리 trace에서 보존 오차·jerk·action norm·basis 잔차를 독립 재계산했다. 최대 물체 보존 점수 차이는 7.2e−15 미만, jerk 개선율 차이는 1.1e−15 수준이다. 실제 종료 제약의 최대 pusher 위치 잔차는 0.0000543px, 속도 잔차는 0.000136px/s였다.','',
'근거: `family_ablation/analysis.json`, `family_ablation/fixed_verifier_analysis.json`, `family_ablation/raw_audit.json`.','<!-- END FAMILY UPDATE -->','']
p=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=p.read_text();begin='<!-- BEGIN FAMILY UPDATE -->';end='<!-- END FAMILY UPDATE -->'
if begin in s:s=s[:s.index(begin)]+s[s.index(end)+len(end):].lstrip('\n')
s=s.rstrip()+'\n\n'+'\n'.join(lines)
s=s.replace('| F1 | H=5·10·20 개발 실험 완료 | 표현/경계 ablation·최종 비교 |','| F1 | H=5·10·20 개발 및 family·실제 경계 ablation 완료 | 강한 baseline·최종 비교 |')
s=s.replace('F1 correction family·실제 종료 경계 및 F9 방향 생성 비교는 별도 실행 중이다.','F1 correction family·실제 종료 경계 및 F9 고정 verifier의 방향 생성 비교는 완료했다.')
s=s.replace('| F9 | 같은 후보의 14개 선택 기준 개발 비교 완료 | 방향 생성 비교·강한 baseline·최종 test |','| F9 | 14개 선택 기준 및 방향 생성 개발 비교 완료 | 강한 baseline·계산 예산 비교·최종 test |')
p.write_text(s)
pp=out/'protocol/progress.json';progress=json.loads(pp.read_text());progress['steps']['F1']['completed']='Development H5/10/20, H5 calibration, family and actual-boundary ablations with raw audit';progress['steps']['F1']['remaining']=['strong baseline and final held-out comparison'];progress['steps']['F1']['evidence']=['f1/summary.json','family_ablation/analysis.json','family_ablation/raw_audit.json']
f9=progress['steps']['F9'];f9['remaining']=[x for x in f9['remaining'] if x!='matched-dimension generator comparisons'];f9['evidence'].append('family_ablation/fixed_verifier_analysis.json');f9['strong_baselines']='Implementation and smoke verified; full development/calibration plus convergence-budget run in progress';progress['updated_unix']=time.time();write_json(pp,progress)
print('Family evidence added; full F0-F11 goal still incomplete.')
