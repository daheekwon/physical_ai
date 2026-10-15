"""Evidence-backed progress report; pending experiment requirements stay visible."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,write_json

out=resolve('artifacts/followup_v1')
manifest=json.loads((out/'protocol/manifest.json').read_text())
audit=json.loads((out/'protocol/audit_summary.json').read_text())
f1=json.loads((out/'f1/summary.json').read_text())
f2=[json.loads(p.read_text()) for p in (out/'interventions').glob('*_h[0-9][0-9].json')]
all_result=next(x for x in f1['oracle'] if x['horizon']==5 and x['phase']=='all')
contact=next(x for x in f1['oracle'] if x['horizon']==5 and x['phase']=='contact')
progress=json.loads((out/'protocol/progress.json').read_text())
progress['steps']['F0']={'status':'complete','evidence':'protocol/audit_summary.json','states':audit['audited_states']}
progress['steps']['F1']={'status':'in_progress','completed':'H=5 development, controller-aware and raw objectives, finite-bank oracle',
                       'remaining':['H=10 and H=20','correction-family and boundary ablations','calibration and final held-out comparison'],
                       'evidence':'f1/summary.json'}
progress['steps']['F2']={'status':'in_progress','completed_state_horizons':len(f2),'executed_candidates':sum(x['candidate_count'] for x in f2),
                       'remaining':['complete bank','explicit contact/near-miss matching and translation/rotation suites',
                                    'matched actual pusher magnitude analysis','finite-difference GPU audit','physical scene-change/history controls']}
progress['verification']={'test_suite':'15 tests passed at F0/F1 implementation checkpoint; later additions require further checks'}
write_json(out/'protocol/progress.json',progress)
text=f'''# 후속 실험 결과 — 진행 중

대상: [후속 실험 계획서](ZJ_WAM_FOLLOWUP_EXPERIMENT_PLAN.md). 전체 F0–F11 수행 목표는 유지한다. 아래 결과는 현재 완료된 실행에 한정하며, 아직 전체 연구 결론이 아니다. 최종 test intervention은 아직 실행하지 않았다.

## 현재 확인한 결과

**F0: 실행과 측정 기반은 검증되었다.** {audit['audited_states']}개 상태를 감사했고 {audit['eligible_states']}개가 사전 정의한 source replay 품질 기준을 통과했다. 동일 root의 반복 실행에서 계측한 상태의 최대 차이는 {audit['repeat_max']}이다. 제어기 식으로 계산한 pusher 위치와 실제 실행의 최대 차이는 {audit['controller_max']:.8f}px다. 위치·회전·물체 속도·접촉을 100Hz로 기록하며, 수정 뒤 원래 행동 3개를 실행한 suffix를 포함한다.

기존 artifact에서 확인한 초기 상태 family 185개 중 170개가 기존 실험에 노출되었다. 후속 split은 probe 학습 100개, development 35개, calibration 35개, 기존 후속 실험에 노출되지 않은 test 15개다. Test는 45개 상태지만 독립 family는 15개이므로 낮은 위반 확률을 정밀하게 추정하기에는 제한이 있다. 이는 downstream 선택 규칙에 대한 분리이며, pretrained WM의 학습 데이터 비노출을 뜻하지 않는다. 공식 action normalization은 기존 전체 데이터 통계를 유지한다.

**F1 H=5 development: 물체를 보존하면서 실제 jerk를 낮출 후보가 일부 상태에 존재한다.** 35개 family의 105개 상태에서 {f1['candidate_count']:,}개 유효 후보를 실행했다. 계획한 raw 후보 중 action bound를 넘는 후보는 실행하지 않았다. 동일 quintic correction 공간에서 raw action roughness와 100Hz 실행 jerk 목적함수를 비교하고, 유효 후보 전체에서 물체 보존 조건을 만족하는 최선의 후보를 사후 선택하는 oracle을 계산했다.

| Oracle 집계 | 물체 보존 + 실제 jerk 10% 감소 | Family bootstrap 95% CI |
| --- | --- | --- |
| 전체 | {all_result['joint']}/{all_result['states']} ({all_result['joint_rate']:.1%}) | {all_result['joint_family_ci'][0]:.1%}–{all_result['joint_family_ci'][1]:.1%} |
| 접촉 상태 | {contact['joint']}/{contact['states']} ({contact['joint_rate']:.1%}) | {contact['joint_family_ci'][0]:.1%}–{contact['joint_family_ci'][1]:.1%} |

Oracle은 이미 실행한 후보 중 최선이며 연속 공간의 최적해나 배포 가능한 방법이 아니다. 새 지표는 100Hz 물체 위치·회전과 suffix 끝 coverage를 포함하므로 이전의 10Hz 경로 보존 수치와 직접 같은 프로토콜로 비교하지 않는다.

## 목적함수 비교

아래는 개발 상태 전체에서 각 radius의 optimizer 결과다. 두 optimizer 모두 아직 물체 승인 규칙을 사용하지 않는다. 같은 radius가 같은 물체 위반율을 보장하지 않으므로 이 표를 최종 방법 우열로 해석하지 않는다.

| 목적함수 | Relative radius | 물체 궤적 보존 | 보존·jerk 감소 동시 달성 | 실제 jerk 평균 감소 |
| --- | --- | --- | --- | --- |
'''
for r in f1['optimizers']:
    if r['horizon']==5 and r['phase']=='all':
        name='실제 제어기 jerk' if r['method']=='optimizer_executed' else 'Raw action roughness'
        text+=f"| {name} | {r['radius']:.2f} | {r['preservation_rate']:.1%} | {r['joint_rate']:.1%} | {r['mean_jerk_gain']:.1%} |\n"
text+='''
현재 결과는 실제 실행을 반영한 목적함수를 사용할 이유를 보여준다. 다만 radius가 커질수록 물체 보존율이 내려가므로, 이후 LeWM 또는 접촉·국소 모델의 후보 승인 규칙이 필요하다. LeWM의 추가 가치에 관한 새 결론은 아직 없다.

## 단계별 완료 범위

| 단계 | 현재 상태 | 남은 검증 |
| --- | --- | --- |
| F0 | 완료 | 후속 코드 변경 시 해당 측정 불변식 재검증 |
| F1 | H=5 개발 실험 완료 | 긴 horizon·표현/경계 ablation·최종 비교 |
| F2 | 공통 물리 intervention bank 실행 중 | 전체 bank, 접촉/near-miss 및 이동/회전의 명시적 매칭, 실제 pusher 크기 매칭 |
| F3 | 수집·probe 학습 코드 준비 | 실제 수집·학습·capacity 대조·label-budget 곡선·family 불확실성 |
| F4 | 미실행 | 실제 endpoint 변화의 측정 해상도 |
| F5 | 미실행 | 물리 효과 propagation, 잘못된 action·zero-effect 대조, 중간 궤적 사각지대 |
| F6 | 미실행 | Activation patch·ablation·대조군·held-out 재현 |
| F7 | 미실행 | 접촉·국소·온라인 보정 모델 |
| F8 | 미실행 | 승인 risk/coverage 및 유한 수정 calibration |
| F9 | 미실행 | 같은 후보·optimizer에서의 결정적 비교와 강한 baseline |
| F10 | 미실행 | 반복 실행·조건 변화·적응·장기 drift |
| F11 | 미실행 | 호환 자산 검증 후 모델·task 확장, 실물은 하드웨어/계측 접근 확인 필요 |

## 검증과 재현

Dense 계측이 원래 접촉 dynamics를 바꾸지 않는지, 제어기 예측이 실제 접촉 실행과 일치하는지, optimizer가 실제 jerk를 줄이는지, 10Hz 사이의 회전 위반과 suffix coverage 위반을 검출하는지를 검사했다. 기존 테스트를 포함한 15개 테스트가 통과했다. 이후 추가된 probe·분석 코드까지 이 테스트 수로 검증했다고 주장하지 않는다.

- 설정: [followup_v1.json](../configs/followup_v1.json)
- Split·모델·source provenance: [manifest](../artifacts/followup_v1/protocol/manifest.json)
- 기존 데이터 노출 이력: [exposure](../artifacts/followup_v1/protocol/exposure.json)
- F0 전수 감사: [audit_summary](../artifacts/followup_v1/protocol/audit_summary.json)
- F1 수치·신뢰구간: [summary](../artifacts/followup_v1/f1/summary.json)
- 지속 진행 상태: [progress](../artifacts/followup_v1/protocol/progress.json)

```bash
source scripts/env.sh
.venv/bin/python scripts/run_followup.py audit --split all
.venv/bin/python scripts/run_followup.py f1 --split development --horizon 5
.venv/bin/python scripts/analyze_followup_f1.py
.venv/bin/python scripts/run_followup_interventions.py --split development --horizon 5
```

각 상태의 완료 JSON을 마지막에 쓰므로 terminal 종료가 확인된 뒤 재실행하면 완료 상태를 건너뛴다. 실행 중인 동일 작업을 중복 시작하지 않는다. Test intervention과 test probe 수집은 final selection lock이 없으면 차단한다. 기존 README와 이전 실험 결과는 보존한다.
'''
resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md').write_text(text)
print(json.dumps({'report':'docs/ZJ_WAM_FOLLOWUP_RESULTS.md','f2_states':len(f2),'f2_candidates':sum(x['candidate_count'] for x in f2)}))
