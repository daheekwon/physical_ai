"""Record completed diagnostics without marking remaining F1/F5/F7-F11 work complete."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,write_json,sha256

out=resolve('artifacts/followup_v1');read=lambda p:json.loads((out/p).read_text())
f1=read('f1/summary.json');f2=read('interventions/raw_audit.json');matched=read('matched_amplitudes/summary.json');capture=read('lewm_audit/readouts/calibration_capture.json')
effect=read('lewm_audit/effects/development_summary.json');resolution=read('lewm_audit/effects/resolution.json');activation=read('activation/completion.json')
progress=read('protocol/progress.json')
progress['steps']['F1'].update({'completed':'H=5,10,20 development; controller-aware/raw objectives and same-family finite bank oracle',
    'remaining':['correction-family and actual-boundary ablations','strong baseline and final held-out comparison in F9']})
progress['steps']['F2']={'status':'development_calibration_complete','evidence':['interventions/raw_audit.json','scene_controls/audit.json','matched_amplitudes/summary.json','protocol/model_numerical_audit.json'],
                       'candidate_count':sum(f2['candidates'].values()),'scene_executions':378,'matched_amplitude_executions':1470,'remaining':['final test under locked selection rules']}
progress['steps']['F3']={'status':'development_calibration_complete','evidence':['lewm_audit/readouts/selection.json','lewm_audit/readouts/calibration_capture.json','lewm_audit/readouts/label_budget.csv','lewm_audit/readouts/matched_capacity.json'],
                       'remaining':['final locked test evaluation']}
progress['steps']['F4']={'status':'h5_development_calibration_complete','evidence':['lewm_audit/effects/resolution.json','lewm_audit/effects/development_summary.json'],
                       'remaining':['multiple-horizon endpoints and zero/repeated-frame diagnostic','final locked test evaluation']}
progress['steps']['F5']={'status':'h5_development_calibration_complete','evidence':'lewm_audit/effects/development_summary.json',
                       'remaining':['H=10/20 propagation','teacher-forced vs free rollout','short-horizon and relinearization contact transition study','final locked test evaluation']}
progress['steps']['F6']={'status':'bounded_diagnostic_complete','development_families':12,'calibration_replication_families':8,'intervention_rows':4800,
                       'evidence':['activation/selection.json','activation/analysis.json'],
                       'remaining':['intact-model representation choice usefulness is tested in F8/F9, not established by patching']}
progress['verification']={'tests':'16 passed at diagnostic implementation checkpoint','raw_audit':f2,'native_model':'protocol/model_numerical_audit.json'}
write_json(out/'protocol/progress.json',progress)
p=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=p.read_text();marker='\n## 후속 진단 결과: F2–F6\n'
if marker in text:text=text.split(marker)[0]
text=text.replace('| F1 | H=5 개발 실험 완료 | 긴 horizon·표현/경계 ablation·최종 비교 |','| F1 | H=5·10·20 개발 실험 완료 | 표현/경계 ablation·최종 비교 |')
text=text.replace('| F2 | 공통 물리 intervention bank 실행 중 | 전체 bank, 접촉/near-miss 및 이동/회전의 명시적 매칭, 실제 pusher 크기 매칭 |','| F2 | 개발·calibration bank와 접촉·장면·실행 크기 대조 완료 | 규칙 고정 후 최종 test |')
text=text.replace('| F3 | 수집·probe 학습 코드 준비 | 실제 수집·학습·capacity 대조·label-budget 곡선·family 불확실성 |','| F3 | 학습·별도 family 평가·capacity·label-budget 진단 완료 | 최종 test |')
text=text.replace('| F4 | 미실행 | 실제 endpoint 변화의 측정 해상도 |','| F4 | H=5 실제 변화 해상도 진단 완료 | 긴 horizon·반복 계측 대조·최종 test |')
text=text.replace('| F5 | 미실행 | 물리 효과 propagation, 잘못된 action·zero-effect 대조, 중간 궤적 사각지대 |','| F5 | H=5 예측·wrong-action·zero-effect·persistence 대조 완료 | 긴 horizon·teacher forcing·재선형화·최종 test |')
text=text.replace('| F6 | 미실행 | Activation patch·ablation·대조군·held-out 재현 |','| F6 | 제한된 patch·ablation·대조군·별도 family 재현 완료 | 실제 metric/smoothing 이점은 F8·F9에서 검증 |')
text+=marker+'''
아래는 development와 calibration을 이용한 진단 결과다. 최종 test 15개 family의 counterfactual 실행은 여전히 잠겨 있다. 방법 선택의 최종 성능과 구분한다.

### F1 horizon 확장

같은 4차원 quintic correction family와 실제 jerk 목적함수로 세 horizon을 실행했다. 모든 수치는 100Hz 물체 궤적 보존과 실제 jerk 10% 감소의 동시 달성이다.

| Horizon | 전체 oracle | 접촉 상태 oracle |
| --- | --- | --- |
'''
for h in [5,10,20]:
 a=next(r for r in f1['oracle'] if r['horizon']==h and r['phase']=='all');b=next(r for r in f1['oracle'] if r['horizon']==h and r['phase']=='contact')
 text+=f"| {h} actions | {a['joint']}/{a['states']} ({a['joint_rate']:.1%}) | {b['joint']}/{b['states']} ({b['joint_rate']:.1%}) |\n"
text+='''
Horizon이 길어질수록 이 제한된 family에서 유용한 수정을 찾는 비율이 낮아졌다. 긴 horizon에 물리적 자유도가 전혀 없다는 결론은 아니다. 상태별 접촉 여부도 horizon에 따라 다시 판정하므로 접촉 분모가 달라진다.

### F2: pusher 크기를 맞춘 뒤의 LeWM 방향

개발·calibration 각각 105개 상태, 각각 23,520개 후보의 원시 trace를 감사했다. 위치·회전·suffix coverage·jerk·action norm을 저장된 trace에서 다시 계산해 기록값과 일치함을 확인했다. GPU native encode와 scalar/batch prediction의 차이는 검사한 표본에서 0이었고, 36개 방향의 유한차분 상대 오차 최댓값은 2.17%였다.

최종 상태만 통과하지만 전체 100Hz 구간 및 suffix 보존을 위반한 후보는 두 split 합계 **5,559/47,040**개였다. 이 수치는 중간 상태·suffix를 추가한 영향까지 포함하며, 전부 10Hz와 100Hz sampling 차이 때문이라는 뜻은 아니다.

이산 radius에서 실제 pusher 크기·주파수를 동시에 맞추면 controller와의 공통 지지 범위가 부족했다. 따라서 물체 결과를 보지 않고 controller 식으로 진폭을 정해 pusher RMS를 맞추는 추가 1,470개 실행을 수행했다. Action norm은 같지 않으므로 별도로 보고했다.

- RMS만 맞춘 접촉 비교에서는 LeWM low의 정규화 물체 오차가 controller보다 평균 0.098 높았다. Family CI는 [0.066, 0.127]이었다. 시간 패턴과 최대 변위 차이가 남으므로 이것만으로 순수한 방향 정보의 우열을 확정하지 않는다.
- RMS·최대 변위·주파수까지 맞춘 frequency 대조군과의 접촉 비교에서 low−대조군 오차 차이는 −0.0079, CI [−0.0350, 0.0168]이었다. 28개 family의 156쌍에서 뚜렷한 이점이 확인되지 않았다.
- 같은 엄격한 조건의 controller 비교는 3개 family의 24쌍뿐이었다. 효과를 일반화하기에 공통 지지 범위가 부족하다.

별도 54개 장면·378개 실행에서는 중심/편심/near-miss의 pusher 변위를 약 2.8e−13px 오차로 맞췄다. 중심 pushing은 주로 이동을 만들고, 편심 pushing은 회전을 동반했으며, near-miss 126개 실행은 모두 비접촉·물체 무변화였다. 장면 위치·방향을 바꿀 때마다 10개 과거 action과 입력 frame을 실제로 재생성했다. 이 장면들은 기전 진단이며 새 expert test family로 세지 않는다.

근거: [F2 전수 감사](../artifacts/followup_v1/interventions/raw_audit.json), [실행 크기 매칭](../artifacts/followup_v1/matched_amplitudes/summary.json), [장면 대조군](../artifacts/followup_v1/scene_controls/audit.json).

### F3: 물체 상태를 읽을 수 있는가

100개 probe 학습 family의 2,000개 시점으로 readout을 학습했다. 35개 development family에서 regularization을 선택한 뒤, 별도 35개 calibration family의 700개 시점에서 고정한 readout을 평가했다. World-model 가중치는 고정했다.

| 표현·linear probe | 물체 위치 평균 오차 | 회전 평균 오차 |
| --- | --- | --- |
'''
for r in capture['results']:
 if r['readout']=='ridge':text+=f"| {r['representation']} | {r['position_mae']:.2f}px | {r['rotation_mae_deg']:.2f}도 |\n"
text+='''
평균값·label shuffle보다 물체 정보를 잘 읽지만, 2px·2도 수준의 절대 상태 판정에는 오차가 크다. 작은 MLP가 일관되게 해결하지도 못했다. 이는 검사한 표현·readout·감독 데이터량의 결과이며 정보가 전혀 없다는 뜻은 아니다. Label-budget 곡선, family CI, 공간/history 표현을 192차원으로 맞춘 capacity 대조도 저장했다. Encoder의 더 좋은 state readout을 LeWM이 그대로 미래 예측할 수 있다고 가정하지 않는다.

근거: [별도 family capture](../artifacts/followup_v1/lewm_audit/readouts/calibration_capture.json), [label budget](../artifacts/followup_v1/lewm_audit/readouts/label_budget.csv), [차원 대조](../artifacts/followup_v1/lewm_audit/readouts/matched_capacity.json).

### F4·F5: 작은 변화의 측정과 미래 효과 예측

Calibration의 실제 encoded endpoint 쌍에서 측정한 효과 오차의 90th percentile을 구하고, 그 두 배를 방향 해석에 필요한 효과 크기 기준으로 고정했다. Native projected CLS+linear probe의 기준은 **위치 5.74px, 회전 4.28도**였다. 이는 이 프로토콜의 보수적인 진단 기준이며, 그보다 작은 모든 변화를 원천적으로 구분할 수 없다는 정보론적 한계나 물리적 안전 기준이 아니다.

개발 bank의 접촉 후보 13,664개에서 같은 frozen readout을 적용한 결과다.

| 효과 예측 기준 | 위치 변화 오차 | 회전 변화 오차 |
| --- | --- | --- |
'''
for method in ['real_encoded','predicted','zero_effect','wrong_action']:
 r=next(x for x in effect['summaries'] if x['representation']=='projected_cls' and x['readout']=='ridge' and x['phase']=='contact' and x['method']==method)
 text+=f"| {method} | {r['position_error_mean']:.3f}px | {r['rotation_error_mean_deg']:.3f}도 |\n"
text+='''
LeWM 예측은 zero-effect보다 위치 오차가 0.763px 작았고, paired family CI는 [−0.920, −0.608]px였다. Wrong-action 대조보다도 작았다. **행동 변화에 따른 물체 반응 정보는 확인된다.**

진단 기준을 넘는 위치 효과는 접촉 후보의 8.14%, 회전 효과는 2.94%였다. 그 부분집합에서 예측 위치 방향이 실제 방향의 45도 이내인 비율은 90.3%, 회전 부호 일치는 93.0%였다. 이를 전체 후보의 정확도로 해석하지 않는다. Unresolved 후보도 물체 보존 통계에 전부 남겼다.

실제 encoded endpoint의 paired readout 오차가 예측 readout보다 컸다. 실제 endpoint readout은 측정 오차 대조군이며, 예측 오차의 수학적 하한은 아니다. 예측 표현의 평활화·수축 효과 등도 가능한 설명이므로 이 차이를 모델이 현실보다 정확하다는 주장으로 해석하지 않는다.

이 결과는 LeWM의 행동 효과 예측 정보를 뒷받침하지만, 2px·2도 궤적 보존을 승인하는 규칙의 신뢰성을 입증하지는 않는다. F8·F9에서 실제 승인 후 위반율과 jerk 감소를 확인해야 한다.

근거: [고정한 진단 해상도](../artifacts/followup_v1/lewm_audit/effects/resolution.json), [효과·대조군 분석](../artifacts/followup_v1/lewm_audit/effects/development_summary.json).

### F6: 내부 activation의 영향

실제 효과를 읽을 수 있는 성공·실패 사례를 골라 development 12개 family에서 성분을 선택하고, 별도 calibration 8개 family에서 같은 선택을 재현했다. 6개 predictor layer의 마지막 시간 token에서 16개 성분을 대상으로 양방향 patch, ablation, context-mean replacement, sham, 같은 개수·L2 크기의 random 대조군, 두 intervention 강도를 실행했다. 총 **4,800개 내부 intervention**이며 실제 simulator 실행 수와 구분한다.

Sham은 출력을 바꾸지 않았고, target/random의 intervention 크기 매칭도 검증했다. 선택 성분이 모델 출력에 영향을 주지만, activation 변화량이 큰 성분이 같은 크기의 random intervention보다 특별히 더 중요한 물체 정보를 담는다는 근거는 얻지 못했다. Calibration에서 선택 성분의 patch/ablation은 평균적으로 random보다 출력 변화와 전체 예측 손상이 작았다. 따라서 이 선택 기준을 곧바로 새 물체 전용 metric으로 채택하지 않는다.

이 결과는 모델 내부 계산의 영향에 관한 것이며 물리적 안전의 증거가 아니다. 배포 비교는 patch하지 않은 원래 predictor를 사용한다. 이후 우선순위는 내부 성분 탐색 확대보다 F7의 간단한 물체 모델과 F8·F9의 승인·smoothing 비교다.

근거: [사례·성분 고정](../artifacts/followup_v1/activation/selection.json), [matched intervention 분석](../artifacts/followup_v1/activation/analysis.json).

## 현재 연구 판단과 남은 작업

1. 물체 궤적을 유지하면서 실제 jerk를 낮출 수정은 일부 상태에 존재한다.
2. LeWM은 물체 반응 정보를 예측하지만, 현재 low 방향과 readout만으로 정밀한 보존 판정을 신뢰할 근거는 충분하지 않다.
3. Pusher 크기·주파수 통제 후 LeWM 방향의 추가 가치도 아직 확립되지 않았다.
4. 다음 핵심 검증은 **같은 jerk 감소 후보를 controller-only, LeWM, 간단한 접촉·국소 모델로 각각 선택했을 때의 위반율과 유용한 적용 비율**이다.

F1의 표현/경계 ablation, F4·F5의 긴 horizon과 실제 관측 재주입·재선형화, F7–F11, 최종 test와 종합 비교는 남아 있다. 전체 목표를 완료로 표시하지 않았다.
'''
p.write_text(text)
snap=out/'protocol/source_snapshots';snap.mkdir(exist_ok=True)
for path in list(resolve('scripts').glob('*followup*.py'))+list(resolve('configs').glob('followup*.json'))+[resolve('zj_wam/model_audit.py'),resolve('zj_wam/readouts.py')]:
    (snap/(sha256(path)+'_'+path.name)).write_bytes(path.read_bytes())
write_json(out/'protocol/runtime_checkpoint.json',{'updated_unix':time.time(),'goal_status':'active','live_sessions':[],
    'completed_this_turn':['F2 development+calibration, matched amplitudes, physical scene controls and raw audit','F3 capture probes, label budgets, capacity controls','F4/F5 H5 effects, frozen calibration resolution','F6 4800 bounded interventions and replication','F1 H10 and H20 development'],
    'next_actions':['Implement/run F7 controller/contact/local/online response models, then F8 calibration and F9 common-candidate comparison','Complete F5 long-horizon and teacher-forced/relinearization studies without dropping original requirements','Complete F1 family/boundary and strong baseline ablations','Keep final test interventions locked until the complete selection pipeline is frozen','Continue F10 and F11 gates/assets/hardware verification and final requirement-by-requirement audit'],
    'important':'Use docs/ZJ_WAM_FOLLOWUP_RESULTS.md and protocol/progress.json. Old write_followup_progress.py is an earlier checkpoint writer; use this updater for current diagnostics. No claim of full completion.'})
print('Updated results and progress through F6; F7-F11 remain pending')
