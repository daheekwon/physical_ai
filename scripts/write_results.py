"""Generate the Korean results report from verified experiment artifacts."""
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import ROOT,write_json

out=ROOT/'artifacts/phase1'
load=lambda name: json.loads((out/name).read_text())
audit=load('audit.json');stats=load('statistics.json');allowance=load('heldout_allowance.json')
cal=load('calibration.json');supp=load('supplement_statistics.json');completion=load('completion.json')
assert audit['complete'] and stats['evaluation_split']=='test'
assert all(v['completed']==v['expected'] for v in completion.values())
api=load('allowance_api_check.json');assert all(r['matches_heldout_decisions'] for r in api)

def desc(group,radius=.1,phase='contact'):
    return next(r for r in stats['descriptive'] if r['group']==group and r['phase']==phase and r['relative_radius']==radius)

def allowed(group,phase='contact'):
    return next(r for r in allowance['evaluation'] if r['group']==group and r['phase']==phase)

def fmt_ci(ci):return f'[{ci[0]:.4f}, {ci[1]:.4f}]'

pair=next(r for r in stats['paired'] if r['phase']=='contact' and r['relative_radius']==.1 and r['comparison']=='low-random')
cor=supp['primary_contact_correlation']['predicted_change']
prob=next(r for r in supp['pairwise_probabilities'] if r['phase']=='contact' and r['relative_radius']==.1 and r['comparison']=='low_vs_random_draw')
status_a='PASS' if pair['mean_difference_ci'][1]<0 and cor['cluster_ci'][0]>0 else 'UNCERTAIN'
# The prototype's largest allowance is supported only if its held-out lower
# confidence limit meets the prespecified 90% operational preservation target.
limit=1-load('manifest.json')['config']['calibration_max_violation_rate']
status_b='PASS within the tested grid' if all(allowed(k)['deployed_preservation_ci'][0]>=limit for k in ['low','low_basis','low_mix']) else 'UNCERTAIN'
lines=[
'# ZJ-WAM Phase 0과 Phase 1 실험 결과',
'\n작성일: 2026년 10월 9일',
'\n**Frozen LeWM이 제안한 저민감도 방향의 실제 물체 교란을 PushT에서 검증했다.** 방향 발견과 유한 수정 범위 탐색, calibration 규칙의 새 평가군 검증까지 실행했다. Jerk 감소나 polynomial smoothing은 실행하지 않았다.',
'\n## 단계별 결론',
'\n| 단계 | 판단 | 근거 |','| --- | --- | --- |',
'| Phase 0 | PASS | 공식 평가 실행, checkpoint 무결성, frozen weights, 입력 미분, 동일 상태 재생 검증 완료 |',
f'| Phase 1A | {status_a} | 접촉 상태의 paired low−random 효과와 예측·실행 상관을 별도 평가군에서 확인 |',
f'| Phase 1B | {status_b} | 유한 수정의 보존율은 측정했으며, 자동 선택한 최대 크기의 보존율과 불확실성은 아래에 제시 |',
'\nPASS는 이 실험 범위의 경험적 근거를 뜻한다. 연속적인 safe region 전체나 새로운 물리 조건의 보증을 뜻하지 않는다. Phase 1B의 판단은 calibration에서 정한 허용 실패율 10%에 대해 실제 최대 수정 선택의 신뢰구간이 충분한지 확인한 운영상 판단이며, 논문 수준의 보편적인 안전 기준이 아니다.',
'\n## 최종 실행 세팅',
'\n- Model: 공식 pretrained LeWM, 18,034,478개 파라미터, 모든 가중치 고정.',
'- Task와 data: 공식 PushT expert trajectories. 18,685개 episode는 초기 배치 기준 185개 family로 묶인다.',
'- Execution: 공식 PushT simulator, 10Hz 제어와 0.01초 physics step.',
'- Model context: t−10, t−5, t의 3개 이미지와 이미 실행한 행동 10개. 새 행동 5개를 묶은 10차원 입력의 민감도를 계산한다.',
'- 예측 horizon: 0.5초. 비교 표현은 192차원 projected CLS latent이며 물체와 pusher 정보가 함께 포함된다.',
'- 상태 분할: development 40개, calibration 60개, test 120개. Family는 분할 사이에 겹치지 않는다.',
f'- Calibration 적합에는 원본 궤적 재현 기준을 만족한 {cal["calibration_source_states_used"]}개 상태를 사용했다. 제외한 2개 상태의 원시 실행 결과는 보존했다.',
f'- 최종 프로토콜에서 총 {audit["states"]}개 상태, {audit["candidate_rows"]:,}개 counterfactual 행동을 실제 실행했다. 원래 초기화 방식의 별도 실행은 최종 통계에 합산하지 않았다.',
'- 방향: low, high, random 3개, 추가 low basis 2개, 저민감도 부분공간의 혼합 방향 3개. 각 방향에 양·음 부호를 적용했다.',
'- 상대 수정 크기: 0.02, 0.05, 0.10, 0.20, 0.40, 0.80. 정규화 공간의 실제 L2 radius는 상대 크기에 √10을 곱한다.',
'\n물체 보존 기준은 위치 변화 ≤2px, 회전 변화 ≤2도, target polygon coverage 감소 ≤0.02를 동시에 만족하는 것이다. 원래 물체 결과 대비 변화이며 물체의 정지를 요구하지 않는다.',
'\n## Phase 0 재현과 구현 점검',
'\n원본 LeWM 평가 스크립트는 현재 dependency에 없는 이전 solver 경로를 참조하여 실패했다. 공식 stable-worldmodel의 최신 평가기를 수정 없이 실행했고, 5개 시작 상태 중 4개가 목표에 도달했다. 이는 실행 확인이며 논문의 50개 시작 상태 benchmark 재현은 아니다. 전체 설정과 로그는 [공식 평가 결과](../artifacts/phase0/official_results.txt)에 있다.',
'\n데이터와 checkpoint의 SHA-256은 Hugging Face 제공 값과 일치했다. 공식 predictor와 cached-context adapter의 예측이 일치하며, 기준 행동 반복 실행과 영 perturbation 검사도 통과했다. 유한차분 간격 sweep으로 float32 취소 오차를 점검하고 0.05를 고정했다.',
f'\n전수 확인한 low 방향의 finite-difference 상대 오차 중앙값은 {audit["finite_difference_median_relative_error_by_low_high_random"][0]*100:.2f}%, 최댓값은 {audit["finite_difference_max_relative_error_by_low_high_random"][0]*100:.2f}%다. High 방향 최댓값은 {audit["finite_difference_max_relative_error_by_low_high_random"][1]*100:.2f}%다. 매우 작은 low derivative에서는 상대 오차가 커질 수 있으므로 개별 결과도 저장했다.',
'\n## 초기화 문제의 수정과 데이터 품질',
'\n초기 전수 감사에서 일부 재생 궤적이 원본 물체 위치와 크게 달랐다. 공식 `_set_state`가 내부적으로 physics step을 수행해, 목표 상태를 먼저 설정할 때의 충돌 보정이 초기 위치를 바꾸는 것이 주요 원인이었다. Adapter는 공식 geometry를 새로 만든 뒤 초기 위치·회전·pusher 속도를 시간 진행 없이 설정하고, 모든 후보에서 동일 전문가 prefix를 재생하도록 수정했다.',
'\n초기 물체 속도·각속도는 데이터에 없어 0으로 가정한다. 이 정보 부족 때문에 calibration의 한 family에 속한 두 상태가 원본 재현 기준을 넘었으며, calibration 적합에서 제외했다. 제외는 후보 행동의 성공 여부가 아니라 원본 상태와 재생 상태의 차이로 결정했다.',
'\n수정 전 결과와 당시 소스는 `artifacts/phase1_initial_replay/`에 보존했다. 이전 test family 60개는 최종 평가에서 모두 제외하고, 새로운 family 60개를 사용했다. 최종 test의 원본 재현 오차는 [재생 품질 CSV](../artifacts/phase1/replay_quality.csv)와 [초기화 사전 점검](../artifacts/phase0/replay_preflight_v2.csv)에 있다.',
'\n## Phase 1A 동일 크기 방향 비교',
f'\n최종 test의 접촉 상태는 {audit["phase_counts"]["test"].get("contact",0)}개다. 아래는 주요 비교 크기인 상대 radius 0.10의 결과다. 정규화 공간 L2 radius는 약 0.3162이다.',
'\n| 방향 | 평균 위치 변화 px | 평균 회전 변화 도 | 물체 보존율 |',
'| --- | --- | --- | --- |']
for group in ['low','random','high','low_mix']:
    r=desc(group);lines.append(f'| {group} | {r["position_error_mean"]:.4f} | {math.degrees(r["rotation_error_mean"]):.4f} | {100*r["preservation_rate"]:.2f}% |')
lines += [
f'\n결합 물체 오차는 `위치 오차 / 120 + 회전 오차 / π`다. Low−random의 평균 paired 차이는 **{pair["mean_difference"]:.5f}**, family 단위 bootstrap 95% 신뢰구간은 **{fmt_ci(pair["mean_difference_ci"])}**다. 음수는 low가 덜 교란함을 뜻한다. 위치와 회전은 결합값과 별도로 보고한다.',
f'\nLow가 개별 random 방향보다 작은 실제 물체 변화를 만든 비율은 **{100*prob["probability"]:.2f}%**이며, 95% cluster 신뢰구간은 {fmt_ci(prob["probability_ci"])}다. 이는 random 3개의 평균과 비교한 승률과 구분한 값이다.',
f'\n접촉 상태에서 유한 predicted change와 실제 물체 변화의 Spearman 상관은 **{cor["spearman"]:.3f}**, 95% cluster 신뢰구간은 **{fmt_ci(cor["cluster_ci"])}**다. 수정 크기를 섞지 않고 상대 radius 0.10 안에서 계산했다.',
'\n![방향별 실제 물체 변화](../artifacts/phase1/plots/01_direction_comparison.png)',
'\n비접촉·근접·접촉 결과는 분리해서 보고한다. 비접촉에서 물체 변화가 모두 0인 경우는 방향 발견의 성공으로 세지 않고 tie로 기록했다.',
'\n![접촉 단계별 방향 비교](../artifacts/phase1/plots/03_contact_comparison.png)',
'\n## Phase 1B 유한 수정과 방향 조합',
'\n| 상대 radius | Low 보존율 | Random 보존율 | High 보존율 | Low mixture 보존율 |',
'| --- | --- | --- | --- | --- |']
for radius in [.02,.05,.1,.2,.4,.8]:
    vals=[100*desc(group,radius)['preservation_rate'] for group in ['low','random','high','low_mix']]
    lines.append(f'| {radius:.2f} | '+ ' | '.join(f'{v:.2f}%' for v in vals)+' |')
lines += [
'\n이 결과는 작은 미분값만 확인한 것이 아니라 유한 크기의 실제 실행까지 비교한 것이다. 개별 저민감도 방향과 그 혼합 방향을 함께 시험했다. 실패 후 더 큰 크기에서 다시 통과한 경우도 `finite_ranges.csv`에 남겼으며, 단조로운 safe interval을 가정하지 않았다.',
'\n![수정 크기별 보존율](../artifacts/phase1/plots/05_finite_preservation.png)',
'\n## 고정한 모델 허용 범위 규칙의 새 평가군 결과',
f'\nCalibration에서 고정한 pooled 규칙은 latent 변화 **≤{cal["rules"]["pooled"]["predicted_change"]:.6f}**, 정규화 수정 radius **≤{cal["rules"]["pooled"]["radius"]:.6f}**다. 각 방향에서 작은 크기부터 연속해서 통과하는 시험 grid 중 가장 큰 크기를 선택한다. 최종 선택에는 simulator contact label이나 실제 물체 결과를 입력하지 않는다.',
'\n| 방향 종류 | 접촉 상태에서 선택한 최대 수정의 보존율 | 95% cluster CI | 평균 선택 L2 radius |',
'| --- | --- | --- | --- |']
for group in ['low','low_basis','low_mix']:
    r=allowed(group);ci=r['deployed_preservation_ci']
    lines.append(f'| {group} | {100*r["deployed_preservation_rate"]:.2f}% | [{100*ci[0]:.2f}%, {100*ci[1]:.2f}%] | {r["deployed_radius_mean"]:.4f} |')
lines += [
'\n작은 수정까지 모두 합친 보존율은 위의 “실제로 선택하는 최대 수정”의 보존율보다 높을 수 있다. 두 지표는 `heldout_allowance.json`에서 분리했다. 최대 수정 선택에서 발생한 실패는 [실패 사례 CSV](../artifacts/phase1/deployed_failure_cases.csv)에 모두 남겼다. Test 결과를 본 뒤 calibration 규칙을 다시 맞추지 않았다.',
'\n## 허용 오차와 적용 범위',
'\n위치·회전·coverage 허용 오차를 각각 원래 기준의 0.5배, 1배, 2배로 적용한 보존율도 계산했다. 이는 설명용 민감도 분석이며 고정된 모델 선택 규칙을 바꾸지 않는다.',
'\n![허용 오차 민감도](../artifacts/phase1/plots/06_tolerance_sensitivity.png)',
f'\n행동 clipping은 사용하지 않았다. 최종 {audit["candidate_rows"]:,}개 후보 중 {audit["bounds_reduced_rows"]}개는 경계 때문에 모든 방향에 공통으로 적용하는 radius를 줄였으며, 실제 적용값을 저장했다. 분석에서 제외된 후보는 {audit["excluded_rows"]}개다.',
'\n이 결과는 frozen 모델의 물체 전용 표현을 검증한 것이 아니다. 전체 latent의 민감도가 물체 실행 결과와 연결되는지 확인했다. Test는 이번 calibration과 초기 배치군이 분리되지만, 원래 pretrained 모델의 학습 데이터와 분리됐다는 뜻은 아니다. 결과 보존은 선택한 0.5초 horizon과 tolerance에 한정된다.',
'\n## 검증과 산출물',
'\n- 7개 핵심·회귀 테스트 통과. 초기화 시 시간 진행 방지와 접촉 중 숨겨진 속도를 포함한 replay 검증을 포함한다.',
f'- {audit["states"]}개 상태의 동일 초기 상태 재실행과 영 perturbation 검증 통과.',
f'- {audit["candidate_rows"]:,}개 후보의 동일 norm, 행동 bounds, 저장된 궤적과 위치·회전 지표 일치 확인.',
'- Family 분할의 중복 없음, 이전 test와 새 test의 중복 없음, 고정 calibration hash 및 checkpoint 불변 확인.',
'- 공개 allowance API를 실제 held-out 관측 3개에 적용해 저장된 모델 선택과 일치함을 확인.',
'\n| 파일 | 내용 |','| --- | --- |',
'| [실행 방법](PHASE0_1_RUNBOOK.md) | 설정, 명령, source와 데이터 버전, 재현 방법 |',
'| [Phase 0 검사](../artifacts/phase0/checks.json) | 모델·정규화·미분·상태 재현 점검 |',
'| [실험 manifest](../artifacts/phase1/manifest.json) | 고정 설정, 실제 상태 목록과 분할 |',
'| [전체 후보 CSV](../artifacts/phase1/all_candidates.csv) | 모든 후보의 예측 및 실제 물체 반응 |',
'| [주요 통계](../artifacts/phase1/statistics.json) | 크기·접촉별 비교와 신뢰구간 |',
'| [추가 확률과 상관](../artifacts/phase1/supplement_statistics.json) | 개별 random 비교와 correlation CI |',
'| [허용 범위 규칙](../artifacts/phase1/calibration.json) | calibration만으로 고정한 선택 규칙 |',
'| [최종 허용 범위 평가](../artifacts/phase1/heldout_allowance.json) | 새 평가군에서 최대 수정 선택의 보존율 |',
'| [전수 감사](../artifacts/phase1/audit.json) | 실제 원시 데이터와 불변성 검증 |',
'\n## 다음 연구 단계에 대한 판단',
'\n저민감도 방향과 실제 물체 결과 사이의 관계, 그리고 유한한 작은 수정에서의 보존을 확인했다면 제한된 범위에서 Phase 2를 검토할 근거가 된다. 다만 큰 수정에 대한 자동 허용 범위를 그대로 안전하다고 취급해서는 안 된다. 새 규칙을 개선할 경우 별도의 calibration과 새 평가군이 필요하다.',
'\n다음 핵심 질문은 이 허용 수정 범위 안에 **실제로 jerk를 줄이는 polynomial correction이 존재하는가**다. 현재 실험으로 그 답까지 얻었다고 주장하지 않는다.',
]
(ROOT/'docs/PHASE0_1_RESULTS.md').write_text('\n'.join(lines)+'\n')
write_json(out/'phase_decisions.json',{'phase0':'PASS','phase1A':status_a,'phase1B':status_b,'scope':'executed direction discovery, finite corrections and fixed-rule heldout evaluation; no jerk or polynomial experiment'})
print(ROOT/'docs/PHASE0_1_RESULTS.md')
