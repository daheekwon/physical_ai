"""Create the proposal-specific addendum from audited diagnostic artifacts."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve

out=resolve('artifacts/phase1_additional')
load=lambda name:json.loads((out/name).read_text())
audit=load('audit.json');m=load('manifest.json');cfg=m['config']
controls=pd.DataFrame(load('controls_statistics.json')['descriptive'])
paired=load('frequency_primary_ci.json');oracle=pd.DataFrame(load('oracle_statistics.json')['summary'])
checks=pd.read_csv(out/'horizon_checks.csv');alignment=pd.read_csv(out/'objective_gradient_alignment.csv')
nominal=pd.read_csv(out/'nominal_episode_quality.csv');old=pd.read_csv(out/'original_path_and_nominal_quality.csv')
selected=pd.read_csv(out/'oracle_selected.csv');raw=pd.read_csv(out/'oracle_candidates.csv')
associations=load('controls_statistics.json')['representation_associations']
controller=load('controller_statistics.json');extension_audit=load('extensions_audit.json')
sampling=pd.DataFrame(load('sampling_rate_statistics.json')['summary'])


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def percentage(x):return f'{100*x:.1f}%'
def interval(x):return f'[{100*x[0]:.1f}%, {100*x[1]:.1f}%]'
sampling_rows=[[int(r.horizon),r.method,percentage(r.joint_path_10hz),percentage(r.joint_path_100hz),percentage(r.joint_both_rates)] for r in sampling[sampling.method.isin(['model_selected','oracle_path','oracle_path_100hz'])].itertuples()]
association_rows=[[r['response'],f"{r['spearman']:.3f}"] for r in associations if r['phase']=='contact' and r['relative_radius']==.1]
primary=controls[(controls.phase=='contact')&(controls.relative_radius==.1)]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
comparison=primary[['method','position_error','pusher']].copy()
comparison=pd.concat([comparison,pd.DataFrame([{'method':'controller','position_error':controller['contact_position_mean'],'pusher':controller['contact_pusher_path_mean']}])],ignore_index=True)
comparison=comparison.set_index('method').loc[['random','coordinate','frequency','low','controller']].reset_index()
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
for ax,key,title in zip(axes,['position_error','pusher'],['Mean object displacement (px)','Mean pusher path change (px)']):
    ax.bar(comparison.method,comparison[key],color=['gray','gray','tab:orange','tab:blue','tab:green']);ax.set_title(title);ax.tick_params(axis='x',rotation=25)
fig.suptitle('Contact states, equal action norm; controller-only follow-up is exploratory')
fig.tight_layout();fig.savefig(out/'plots/04_controller_only_comparison.png',dpi=150);plt.close(fig)
freqrows=[[r.method,f'{r.position_error:.4f}',f'{r.pusher:.4f}',f'{r.centroid:.3f}',percentage(r.final)] for r in primary.itertuples()]
rows=[]
for h in cfg['oracle_horizons']:
    for method in ['unconstrained_command','model_selected','oracle_final','oracle_path']:
        r=oracle[(oracle.phase=='all')&(oracle.horizon==h)&(oracle.method==method)].iloc[0]
        rows.append([h,method,percentage(r.preserved_path),percentage(r.joint_path),interval(r.joint_path_ci),percentage(r.jerk_reduction),percentage(r.physics_jerk_reduction),percentage(r.fallback)])
costrows=[]
for h,g in checks.groupby('horizon'):
    fd=np.array([json.loads((out/'oracle'/f'{r.state_id}_h{h:02d}.json').read_text())['finite_difference_relative_errors'] for r in g.itertuples()])
    costrows.append([h,f'{g.nominal_future_latent_prediction_error.mean():.3f}',f'{g.jacobian_seconds.median()*1000:.1f}',f'{g.duration_seconds.median():.2f}',f'{fd.max()*100:.2f}%'])
pathrows=[]
for radius in [.1,.4,.8]:
    for method in ['low','low_mix','random','high']:
        g=old[(old.phase=='contact')&(old.relative_radius==radius)&(old.method==method)]
        pathrows.append([radius,method,percentage(np.average(g.final_rate,weights=g.states)),percentage(np.average(g.path_rate,weights=g.states))])
ci=paired['position_error']['cluster_ci']
text=f'''# 영문 제안서 Phase 0과 Phase 1 추가 검증

작성일: 2026년 10월 9일

[영문 제안서](ZJ_WAM_Research_Proposal_Updated_EN.tex)의 Phase 0, 1A, 1B를 [기존 실행 결과](PHASE0_1_RESULTS.md)와 대조하여 **주파수·제어기 대조군, 다항식 simulator oracle, 긴 horizon, 경계 포함 jerk, 물체 궤적 보존과 기준 episode 품질**을 추가 검증했다. 기존 26,400개 실행과 calibration은 그대로 보존했으며 이번 결과는 별도 디렉터리에 있다.

**핵심 결론:** 주파수를 맞춘 뒤에도 LeWM low의 차이는 남았지만, PD 제어기 식만으로 찾은 저반응 방향의 평균 물체 위치 변화가 더 작았다. 따라서 이번 결과로 LeWM의 물체 반응 정보가 generic controller baseline보다 우월하거나 필수적이라고 주장할 수 없다. 한편 유한한 다항식 후보 안에 물체 궤적을 보존하며 실제 jerk를 줄이는 수정은 일부 상태에서 존재했다. 그 가능성을 모델 규칙이 안정적으로 회수하는지는 아직 불확실하다.

## 추가 요구사항과 실행 범위

{table(['제안서 항목','기존 근거','추가 실행'],[
['Phase 0 실행·복원·미분','공식 평가 5개 시작 상태와 5-action 미분 검증','계측이 동역학을 바꾸지 않는 회귀 검증, 10·20-action 미분 및 실제 미래 latent 오차'],
['Phase 1A 대조군','Low/high/random 비교','Coordinate 10개, low와 DCT 주파수 에너지를 맞춘 random 3개'],
['Phase 1A 제어기 영향','행동 norm만 동일','실제 pusher 변위와 시간 주파수, 모델 변화와 물체·pusher 변화의 상관'],
['Phase 1B B8 oracle','유한 크기와 low mixture 실행','같은 quintic correction 후보에서 물체 보존 및 실제 jerk 감소의 최선 후보 탐색'],
['Phase 1B horizon과 경계','0.5초 최종 결과','5·10·20 action, 이전 3개 및 이후 3개 제어 구간 포함'],
['Trajectory tolerance','중간 오차 저장','최종 결과 보존과 경로 전체 보존을 분리 집계'],
['Nominal quality','짧은 chunk 성공 지표','110개 기준 episode를 최종 기록 상태까지 재생']])}

새 대조군은 **{audit['controls_states']}개 상태, {audit['controls_executions']:,}개 실행**이다. Oracle은 development 4개, calibration 12개, test 24개 상태이며 각 split에서 서로 다른 family를 선택했다. 세 horizon에서 **{audit['oracle_executions']:,}개 후보 실행**을 수행했다. 마지막 controller-only 대조군 {controller['candidate_count']:,}개를 추가하여 합계 **{audit['controls_executions']+audit['oracle_executions']+controller['candidate_count']:,}개 새 후보 실행**이며 nominal 반복과 episode 품질 재생은 이 수에 포함하지 않는다.

이번 test family는 최초 연구에서 이미 결과를 본 family다. 새 후보를 실행하기 전에 프로토콜을 정하고 calibration과 test를 분리했지만, 완전히 새로운 외부 검증군은 아니다. 본 결과는 제안서가 요구한 추가 진단에 해당하며, 확증 실험에서는 새로운 평가군이 필요하다.

## Phase 1A 주파수를 맞추면 이점이 얼마나 남는가

5-action 방향을 시간축 DCT로 변환한 후, 주파수별 2차원 에너지는 그대로 유지하고 공간 방향을 무작위 회전시켰다. 이 방법은 low와 **모든 DCT 주파수의 에너지 및 정규화 action norm**을 일치시킨다. 좌표축 대조군은 10차원 행동 공간의 각 축이다. 모든 방법과 양·음 부호에 같은 유효 radius를 적용하고 clipping하지 않았다. DCT 일치는 모델의 정규화 행동 좌표계 기준이며, 축별 역정규화 scale은 실제 실행 때 반영한다.

접촉 상태 69개, 상대 radius 0.10의 결과는 다음과 같다.

{table(['방향','물체 최종 위치 변화 px','Pusher 평균 경로 변화 px','DCT 주파수 중심','최종 결과 보존율'],freqrows)}

Low−frequency-matched 위치 변화 차이는 **{paired['position_error']['low_minus_frequency_mean']:.4f}px**, family bootstrap 95% CI **[{ci[0]:.4f}, {ci[1]:.4f}]px**다. Low−frequency-matched 결합 물체 오차 차이도 음수이며 [원시 통계](../artifacts/phase1_additional/frequency_primary_ci.json)에 별도로 기록했다.

기존 low 대 random의 큰 격차는 주파수를 맞추면 상당히 줄어든다. 따라서 원래 결과를 전부 물체에 특화된 모델 지식의 효과로 해석할 수 없다. 주파수를 맞춘 뒤에도 차이는 남지만, pusher의 실제 이동 변화도 low에서 더 작다. 이번 대조군은 pusher 궤적의 크기와 방향까지 완전히 일치시킨 실험은 아니므로, 남은 효과의 원인을 하나로 단정하지 않는다.

![주파수 및 제어기 대조군](../artifacts/phase1_additional/plots/01_frequency_controller_controls.png)

### 물체 정보 없는 controller-only 대조군

주파수 대조군의 결과를 확인한 뒤, 추가적인 post hoc 진단으로 공식 PD 식만 사용한 대조군을 실행했다. 각 정규화 action 좌표가 5개 제어 시점의 pusher 위치에 미치는 선형 response matrix를 계산하고, pusher 경로 변화가 가장 작은 두 방향에서 고정된 네 방향을 선택했다. 물체 위치, LeWM 예측, 물체 결과는 방향 선택에 사용하지 않았다. 비교 norm은 위 대조군과 동일하다.

접촉 상태의 평균 물체 위치 변화는 **{controller['contact_position_mean']:.4f}px**, pusher 평균 경로 변화는 **{controller['contact_pusher_path_mean']:.4f}px**다. Low−controller-only 결합 물체 오차 차이는 **{controller['low_minus_controller_object_error']:.6f}**, family bootstrap 95% CI는 **{controller['cluster_ci']}**다. 음수는 LeWM low가 더 적은 물체 변화를 만들었음을 뜻한다. 이는 simulator를 사용해 물체에 유리한 방향을 사후 선택한 oracle이 아니라, 알려진 controller의 입력·출력 식만 사용하는 대조군이다.

실제 적용 action으로 계산한 해석적 pusher 변위와 simulator 변위는 전수 검사에서 최대 **{extension_audit['analytic_pusher_response_max_error']:.2e}px** 차이였다. LeWM low 방향 에너지 중 controller의 두 저반응 방향 공간으로 투영되는 비율은 평균 **{controller['mean_world_low_projection_into_controller_low_two']:.1%}**다. 이 수치와 물체 변화 비교를 함께 사용하여 generic controller attenuation의 설명력을 평가한다. 이 후속 대조군은 앞선 결과를 본 뒤 추가했으므로 독립적인 확증 검정으로 해석하지 않는다. 그럼에도 동일 action norm에서 generic controller 방향이 평균 물체 변화를 더 작게 만들었으므로, 현재 실험은 LeWM 고유의 추가 가치를 입증하지 못했다. 제어기와 실제 실행의 작은 수치 차이는 float32 action과 Vec2d 목표 계산의 반올림을 포함하며, 감사 허용 오차는 0.001px로 두고 실제 오차를 위에 그대로 보고했다.

![Controller-only와 LeWM 대조군](../artifacts/phase1_additional/plots/04_controller_only_comparison.png)

### Global latent가 반영하는 반응

{table(['실행 반응','유한 latent 변화와 Spearman 상관'],association_rows)}

동일한 상대 radius 0.10의 접촉 대조군에서 물체와 pusher 반응을 따로 비교했다. 이 상관은 global latent가 어떤 실행 변화와 연결되는지 보여주는 진단이며, latent에서 물체와 로봇 정보를 분리한 결과는 아니다.

## Phase 1B 다항식 oracle 정의

모든 horizon에서 같은 4차원 correction family를 사용했다. 각 축의 수정은 `u²(1−u)²[c0+c1(2u−1)]`이며, 시작·끝 수정값과 연속 다항식의 1차 미분이 0이다. 정규화 행동 공간에서 orthonormal basis를 구성했다. 이는 **수정량**의 다항식 표현이며, 전체 전문가 행동이 다항식이라는 뜻은 아니다. 실제 pusher의 경계 속도나 jerk를 0으로 강제하는 조건도 아니다.

상태·horizon마다 모델의 polynomial 공간 low 방향 2개, isotropic random 방향 16개, command roughness를 줄이는 방향 1개를 양·음 부호와 상대 radius 0.05·0.10·0.20·0.40에서 시험했다. 총 **152개 후보**에 nominal fallback을 추가한 유한 탐색이다. 모든 후보의 실제 norm, 동작 경계와 다항식 공간 소속을 감사했다.

- `oracle_final`: 최종 물체 위치·회전·coverage 조건을 만족하는 후보 중 실제 제어 시점 jerk가 가장 작은 후보.
- `oracle_path`: 위 조건과 물체 경로의 위치·회전 허용 오차까지 만족하는 후보 중 실제 jerk가 가장 작은 후보.
- `unconstrained_command`: 같은 후보 풀에서 물체·모델 제약 없이 raw command roughness가 가장 작은 후보. 물리적으로 가능한 최소 jerk의 상한으로 해석하지 않는다.
- `model_selected`: 물체 실행 결과와 실제 jerk를 보지 않고, 고정한 radius·유한 latent 변화 제약 아래 raw command roughness가 가장 작은 후보.

Oracle은 후보 풀 안에서 찾은 최선의 실현 가능한 결과다. 연속 coefficient 공간의 전역 최적해나 oracle 성능의 확정 상한이 아니다. 모든 방법이 같은 후보 풀을 사용하므로 spectral 방향 자체의 추가 가치를 검증하는 Phase 2 B3–B7 전체 비교를 대신하지 않는다.

## 측정 단위와 경계

수정 구간 이전 3개 제어 위치와 이후 동일한 전문가 행동 3개를 포함하여 jerk energy를 계산했다. 실행 시간을 늘리지 않았으며 제어 간격은 0.1초, 물리 적분 간격은 0.01초다. 위치는 PushT의 pixel 좌표이므로 jerk energy 단위는 **px²/s⁵**이고, SI 단위의 힘이나 에너지로 해석하지 않는다. 10Hz와 100Hz 결과를 별도로 보고하며 서로 다른 sampling rate의 energy 크기를 직접 비교하지 않는다.

Jerk의 의미 있는 감소는 development 실행 전 **10% 이상**으로 정했다. 기준 jerk가 거의 0인 상태는 개선으로 세지 않는다. 모든 평가 상태와 fallback을 분모에 포함한다. 경로 보존은 최종 위치 2px·회전 2도·coverage 감소 0.02 조건에 더해, 제어 시점의 물체 위치·회전 변화가 수정 구간 및 3개 suffix 동안 각각 2px·2도 이내인 것이다. 물리 시점의 물체 위치 최대 오차도 별도로 저장했다.

현재 action은 현재 pusher 위치에 대한 상대 목표다. Raw action의 3차 차분은 **roughness surrogate**이며 실제 위치 jerk와 동일한 양이 아니다. 이 surrogate를 이용한 모델 선택과 실제 jerk를 이용한 oracle 사이의 차이에는 결과 예측, calibration, 목적함수 대응, 후보 탐색의 한계가 함께 들어 있다. 이 차이를 모두 월드 모델 오류로 돌리지 않는다.

## Oracle과 모델 선택 결과

아래는 horizon마다 동일한 test 상태 24개를 모두 집계한 결과다. 각 horizon의 실제 nominal 접촉 여부는 다시 측정했으므로 긴 horizon에서 접촉 표본 수가 달라질 수 있다.

{table(['H','선택 방법','경로 보존','보존과 jerk 10% 감소','동시 달성 95% CI','10Hz jerk 평균 감소','100Hz jerk 평균 감소','Fallback'],rows)}

새 calibration은 각 horizon별로 12개 family만 사용했다. 선택한 active correction의 경로 위반율이 전체와 접촉에서 각각 10% 이하이고 최소 절반의 상태에서 active correction을 선택하는 규칙 중 surrogate 감소가 가장 큰 것을 택했다. 이 작은 표본에서의 경험적 조건은 확률적 안전 보증이 아니다.

5-action 규칙은 상대 radius 0.10, latent 변화 예산 {load('oracle_calibration.json')['rules']['5']['tau']:.6f}를 선택했다. 10·20-action에서는 정한 규칙 탐색과 활성 상태 조건을 만족하는 규칙을 찾지 못해 **수정 없이 nominal로 fallback**했다. 이 결과는 긴 horizon에 물리적 개선 여지가 없다는 뜻이 아니라, 이번 규칙과 표본으로 유용한 허용 범위를 확립하지 못했다는 뜻이다. Test 결과를 이용해 규칙을 다시 맞추지 않았다.

![Oracle과 모델 선택의 차이](../artifacts/phase1_additional/plots/02_oracle_gap.png)

### Sampling rate에 대한 추가 분석

주 평가의 10Hz jerk endpoint는 유지하고, 같은 후보 풀에서 100Hz 물리 시점 jerk를 직접 최소화하는 `oracle_path_100hz`를 별도 계산했다. 아래 비율은 모두 물체 경로 보존과 해당 sampling rate의 jerk 10% 감소를 함께 만족하는 상태 비율이다.

{table(['H','선택 방법','10Hz 동시 달성','100Hz 동시 달성','두 rate 모두 동시 달성'],sampling_rows)}

예를 들어 5-action의 10Hz oracle이 10Hz에서 보인 15/24 동시 달성은 100Hz에서 8/24로 줄어든다. 100Hz를 직접 최적화한 oracle은 100Hz에서 12/24를 달성하지만 두 rate 모두에서는 9/24다. 따라서 하나의 sampling rate에서 측정한 jerk 감소를 모든 시간 해상도의 동작 개선으로 확대하지 않는다. 100Hz oracle 역시 유한 후보 풀의 진단 결과이며 기존 calibration을 변경하지 않았다.

## Phase 0 긴 horizon의 모델 오차와 미분 비용

{table(['행동 수','실제 미래 대비 latent L2 오차 평균','Jacobian 시간 중앙값 ms','152개 후보 전체 시간 중앙값 s','방향 유한차분 최대 상대 오차'],costrows)}

실제 미래 이미지를 공식 encoder로 처리한 표현과 nominal rollout의 예측 표현을 비교했다. 이 latent 오차는 물체 위치 오차가 아니며, 출력 표현 전체의 예측 오차다. 모델 weight는 모두 고정했고, 각 horizon에서 같은 물리 초기 상태를 반복 실행한 trace가 정확히 일치했다. 미분 검증은 polynomial low, random, command 방향의 central difference 간격 0.05를 사용했다.

실행 시간은 CPU 4-thread 환경에서 측정했다. 후보 탐색 시간에는 prefix 재생과 simulator 검증이 포함되므로 deployment inference latency와 구분한다. Jacobian 시간과 모델 후보 forward 누적 시간은 [horizon 검사 CSV](../artifacts/phase1_additional/horizon_checks.csv)에 따로 저장했다.

![Horizon별 오차와 미분 비용](../artifacts/phase1_additional/plots/03_horizon_diagnostics.png)

## 중간 물체 궤적과 nominal 성공

기존 test 실행의 접촉 상태를 최종 결과 기준과 경로 기준으로 다시 집계했다. 이 표의 경로는 기존 5-action 구간이며, 위 oracle 표의 추가 suffix를 포함하지 않는다.

{table(['상대 radius','방향','최종 보존율','경로 보존율'],pathrows)}

기준 episode 110개를 데이터의 마지막 기록 상태까지 재생한 결과, 환경의 공식 state-based success 기준에서 **{int(nominal.nominal_episode_success.sum())}/{len(nominal)}개가 성공**했다. 목표는 각 기록 episode의 마지막 상태이므로 이것은 원본 재현과 nominal 품질 확인이며 새로운 goal-reaching benchmark는 아니다. 이번 선택에는 실패한 nominal episode가 없어 성공·실패 episode 간 효과 차이는 평가할 수 없다. 짧은 chunk 끝의 목표 도달 여부는 이 episode 성공과 별도로 집계했다.

[추가 경로 및 nominal 집계](../artifacts/phase1_additional/original_path_and_nominal_quality.csv), [전체 episode 재생](../artifacts/phase1_additional/nominal_episode_quality.csv), [명령 및 실제 jerk의 국소 gradient 대응](../artifacts/phase1_additional/objective_gradient_alignment.csv)에 상세 값을 기록했다. Gradient 대응은 작은 양·음 후보의 실제 jerk 차이를 polynomial 좌표에 최소제곱 적합한 진단이며, 그 잔차도 함께 보고한다.

## 연구 판단과 다음 단계

Phase 0은 5·10·20-action에서 계측·복원·미분 경로가 동작함을 뒷받침한다. 공식 평가 자체는 기존의 5-start 실행 확인이며 논문 전체 benchmark를 재현했다는 주장은 유지하지 않는다.

Phase 1A의 low 대 random 차이는 재현되었다. 그러나 주파수 대조군이 격차를 크게 줄이고, controller-only 대조군은 평균 물체 변화를 더 작게 만들었다. 따라서 **저반응 방향의 존재는 뒷받침되지만 LeWM 고유의 물체 반응 정보가 제공하는 추가 가치는 입증되지 않았다**로 판단한다. 후속 비교에는 controller-only 저반응 방향과 실제 pusher 변위를 맞춘 대조군을 포함해야 한다.

Phase 1B는 새 oracle에서 실제 물체 보존과 jerk 감소의 가능성을 직접 평가했다. 그 가능성의 크기와 모델 선택의 달성률은 위 표가 근거다. 일부 조건에서 oracle이 개선을 찾아도, 모델 기반 자동 허용 범위와 목적함수가 그 개선을 안정적으로 회수한다는 주장은 아직 별도의 검증이 필요하다. 후속 작업은 실제 jerk에 맞는 relative-action 목적함수, horizon별 예측·허용 범위, 그리고 새로운 평가군을 우선한다.

## 재현과 감사

[추가 실행 방법](PHASE0_1_ADDITIONAL_RUNBOOK.md)에 전체 명령과 프로토콜을 기록했다. [전수 감사](../artifacts/phase1_additional/audit.json)는 핵심 추가 {audit['all_raw_trajectories_checked']:,}개 후보의 norm·bounds·궤적 지표, 다항식 공간, 경계 포함 jerk, oracle 선택, calibration hash를 검증한다. [확장 감사](../artifacts/phase1_additional/extensions_audit.json)는 controller-only {controller['candidate_count']:,}개 실행과 100Hz oracle의 선택을 검증한다. 원본 README·영문 제안서·기존 실험 코드 및 calibration은 변경하지 않았다.
'''
resolve('docs/PHASE0_1_ADDITIONAL_RESULTS.md').write_text(text)
print('Wrote docs/PHASE0_1_ADDITIONAL_RESULTS.md')
