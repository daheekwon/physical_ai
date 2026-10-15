"""Append current F7-F9 and long-horizon evidence without resetting earlier results."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve,write_json
out=resolve('artifacts/followup_v1');sel=json.loads((out/'selection/evaluation_summary.json').read_text());response=json.loads((out/'response_scores/diagnostics.json').read_text());long=json.loads((out/'lewm_audit/long_horizon/analysis.json').read_text())
labels={'controller_pusher':'제어기만 사용','lewm_global':'LeWM global latent','lewm_whitened':'LeWM 분산 정규화','lewm_object':'LeWM + 감독 물체 readout','lewm_history_object':'LeWM + 감독 history readout','vision_contact':'이미지 + 접촉 모델','vision_local':'이미지 + 국소 모델','vision_adaptive':'이미지 + 온라인 보정','state_contact':'GT pose + 접촉 모델','state_local':'GT pose + 국소 모델','state_adaptive':'GT pose + 온라인 보정','vision_contact_endpoint':'이미지 접촉 모델 endpoint만'}
def get(method,stratum='all',origin='selector'):
 return next(r for r in sel['selection'] if r['split']=='development' and r['stratum']==stratum and r['rule_source']==origin and r['method']==method)
def pct(v):return '정의되지 않음' if v is None else f'{100*v:.1f}%'
lines=['<!-- BEGIN SELECTION UPDATE -->','## F7 반응 모델과 F8–F9 공통 후보 비교','',
'**개발·calibration 단계 결과이며 최종 test 결과가 아니다.** 각각 35개 family·105개 상태에서 intervention bank와 다항식 smoothing bank를 평가했다. 14개 점수의 threshold·radius는 calibration family로 정했고, 아래는 development family에서의 진단 결과다. Development는 앞선 표현·방법 개발에 사용되었으므로 새로운 확증 test로 해석하지 않는다.','',
'F7은 현재까지 얻은 관측으로만 접촉·국소·온라인 보정 모델을 계산한다. 상태 접근 모델에는 실제 물체 pose와 지난 0.3초 trace를 제공했다. 이미지 기반 모델에는 LeWM과 같은 3개 RGB 관측, 알려진 PushT 모양·색상과 로봇 proprioception을 제공했다. 두 모델의 사전 지식과 관측 이력이 같지는 않으므로 정보 접근을 구분한다. 이미지 pose 추정은 development 평균 1.08px·0.74도, calibration 평균 0.97px·0.67도 오차였고, 일부 상태는 2px·2도를 넘었다.','',
'접촉 상태의 동일 H=5 endpoint에서 물체 변화 예측 오차는 LeWM 1.049px·0.595도, 이미지 접촉 모델 0.785px·0.598도, 이미지 온라인 보정 0.677px·0.463도였다. 온라인 보정은 과거 관측으로 갱신한 translation/rotation 반응 계수 두 개를 사용하며, 물리적 질량·마찰을 식별했다고 주장하지 않는다. 국소 회귀는 excitation 부족 시 접촉 모델과 불확실성으로 보수적으로 제한했다.','',
'### 같은 smoothing 후보에서의 선택 결과','',
'모든 방법이 같은 controller-aware 후보 bank에서 예상 실행 jerk가 가장 낮은 승인 후보를 선택한다. 기준 nominal jerk와 후보 jerk 모두 알려진 제어기 예측으로 계산하고, 미래 물체·접촉·실행 jerk는 선택에 공급하지 않는다. 물체 위반은 100Hz 궤적과 suffix의 원래 E_obj 정의로 판정한다.','',
'Calibration 목표는 전체 및 접촉 상태의 선택 후 경험적 위반율 ≤10%, active correction 비율 ≥25%다. 조건을 만족하는 규칙 중 예상 jerk 10% 개선 상태 비율을 최대화한다. 이는 확률적 안전 보증이 아니다.','',
'| 방법 | 보존 + jerk 10% 감소 U | U의 family 95% CI | 수정 적용 비율 | 적용 후 물체 위반율 | 접촉 상태 적용 후 위반율 |','| --- | --- | --- | --- | --- | --- |']
for method,label in labels.items():
 r=get(method);c=get(method,'contact');ci=r['joint']['ci95']
 lines.append(f"| {label} | {pct(r['joint']['mean'])} | {pct(ci[0])}–{pct(ci[1])} | {pct(r['active_coverage']['mean'])} | {pct(r['conditional_violation']['mean'])} | {pct(c['conditional_violation']['mean'])} |")
lines+=['','같은 유한 후보 bank의 물리 결과 oracle은 U=34.3%다. Action norm만 사용하는 규칙은 사전 선언한 calibration 조건을 충족하지 못해 모두 nominal로 fallback했고 U=0이었다. 이를 보존 성능 개선으로 해석하지 않는다.','',
'LeWM global의 controller 대비 U 차이는 +4.8%p이며 paired family 95% CI는 −1.0~+12.4%p다. 감독 readout 변형은 +8.6%p [1.9, 17.1], 이미지 접촉 모델은 +18.1%p [11.4, 25.7]였다. 이는 development 진단이며 다중 방법 탐색을 거친 최종 확증 효과 크기가 아니다. 감독 readout에는 별도의 2,000개 학습 frame label이 추가된다.','',
'**현재 근거는 LeWM이 쓸모없다는 결론보다, 물체를 직접 읽는 기준과 간단한 접촉 모델을 비교해야 한다는 결론을 지지한다.** 이미지 접촉 모델을 chunk endpoint만 검사하도록 제한하면 U가 26.7%에서 13.3%로 내려갔다. Dense·suffix 반응을 평가하는 차이가 중요한 후보 설명이며, 이를 모두 더 좋은 물리 표현의 효과로 돌리지 않는다. 이 둘은 각기 보정한 규칙의 비교다.','',
'### 후보 전체의 위험과 선택 후 위험은 달랐다','',
'이미지 접촉 모델의 F8 bank 규칙은 development 후보 전체에서 승인 coverage 73.2%, 승인 후 위반율 3.9%였다. 그 규칙을 그대로 smoothing bank의 최선 후보 선택에 사용하면 위반율은 16.2%였다. 별도의 F9 선택 분포에서 보정한 뒤 같은 development 평가의 위반율은 1.9%가 됐다. 후보 분포와 optimizer 선택이 모두 달라지는 비교이므로 상승분을 선택 연산 하나의 인과 효과로 해석하지 않는다. 제어기 점수에서도 4.3%→15.2% 차이가 관측되었다.','',
'따라서 후보 승인 보정만으로 배포 선택의 위험을 판단할 수 없다. 이 결과를 본 뒤 test threshold를 재선택하지 않으며, 최종 test는 모든 주 비교 방법을 고정한 후 사용한다.','',
'## F4–F5 긴 horizon과 과거 관측 갱신','',
'H=5·10·20 각각 development 105개 상태의 기존 F1 후보를 평가했다. 동일한 frozen native readout으로 실제 endpoint, free rollout, 5/10행동마다 이미 실행한 관측을 공급하는 context refresh, 초기 마지막 frame 반복, norm을 맞춘 wrong action, zero-effect를 비교했다. Context refresh는 원래 행동 선택 시점보다 많은 과거 실행 정보를 얻는 진단이며, open-loop correction 선택이나 재선형화 smoothing을 완료한 것으로 보지 않는다.','',
'아래는 접촉 상태에서 마지막 endpoint의 물체 변화 예측 오차다. 각 horizon의 후보 공간·접촉 상태 집합도 달라지므로 horizon 차이 전체를 예측 누적만의 영향으로 해석하지 않는다. 같은 H 안의 free/refresh 차이가 직접 비교다.','',
'| Horizon | Free rollout 위치 / 회전 | 5행동마다 관측 갱신 | 실제 endpoint encode | Zero-effect |','| --- | --- | --- | --- | --- |']
for h in [5,10,20]:
 def entry(m):
  r=next(x for x in long['summary'] if x['horizon']==h and x['endpoint_step']==h and x['representation']=='projected_cls' and x['stratum']=='contact' and x['method']==m)
  return f"{r['position_effect_error']['mean']:.3f}px / {r['rotation_effect_error_deg']['mean']:.3f}도"
 lines.append(f"| {h} | {entry('free')} | {entry('refresh_5')} | {entry('real_encoded')} | {entry('zero_effect')} |")
lines+=['','H=20에서는 과거 관측 갱신 후 오차가 작아졌지만 여전히 2px·2도 수준을 넘는다. H=10에서는 갱신이 개선하지 않았다. 실제 endpoint readout도 오차가 커서 긴 구간의 문제를 predictor 누적 오류 하나로 설명할 수 없다. Readout 오차는 예측 오차의 수학적 하한이 아니다. Wrong-action보다 낮은 오차는 행동 효과 정보가 있음을 지지하지만, 그것만으로 궤적 보존 승인에 충분하다고 볼 수 없다.','',
'### 검증과 남은 범위','',
'- 현재 테스트 23개 통과. 미래 결과 label 변경에 대한 선택 불변성과, context refresh가 현재/미래 target을 소비하지 않는 검증을 포함한다.','- F7 점수 계산 420개 bank-state artifact 완료. F8 후보 승인과 F9 선택 규칙 각각 14개를 별도로 보정했다.','- F1 correction family·실제 종료 경계 및 F9 방향 생성 비교는 별도 실행 중이다.','- 강한 baseline, 직접 비선형 WM 제약, soft sensitivity penalty, 접촉 전환의 짧은 correction·재선형화, 최종 test, F10·F11은 아직 남아 있다.','',
'근거: `response_scores/diagnostics.json`, `response_scores/jerk_input_audit.json`, `selection/bank_rules.json`, `selection/selector_rules.json`, `selection/evaluation_summary.json`, `lewm_audit/long_horizon/analysis.json`.','<!-- END SELECTION UPDATE -->','']
p=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=p.read_text();begin='<!-- BEGIN SELECTION UPDATE -->';end='<!-- END SELECTION UPDATE -->'
if begin in text:text=text[:text.index(begin)]+text[text.index(end)+len(end):].lstrip('\n')
text=text.rstrip()+'\n\n'+'\n'.join(lines)
text=text.replace('| F7 | 미실행 | 접촉·국소·온라인 보정 모델 |','| F7 | 개발·calibration 점수와 모델 오차 평가 완료 | 최종 test와 조건 변화 |')
text=text.replace('| F8 | 미실행 | 승인 risk/coverage 및 유한 수정 calibration |','| F8 | 14개 후보 승인 규칙 보정·개발 평가 완료 | 최종 test, 추가 생성기 |')
text=text.replace('| F9 | 미실행 | 같은 후보·optimizer에서의 결정적 비교와 강한 baseline |','| F9 | 같은 후보의 14개 선택 기준 개발 비교 완료 | 방향 생성 비교·강한 baseline·최종 test |')
text=text.replace('| F4 | H=5 실제 변화 해상도 진단 완료 | 긴 horizon·반복 계측 대조·최종 test |','| F4 | H=5 해상도 및 H=5·10·20 endpoint 진단 완료 | 반복 계측 대조·최종 test |')
text=text.replace('| F5 | H=5 예측·wrong-action·zero-effect·persistence 대조 완료 | 긴 horizon·teacher forcing·재선형화·최종 test |','| F5 | H=5·10·20 예측·대조군·과거 관측 갱신 진단 완료 | 접촉 전환 correction·재선형화·최종 test |')
p.write_text(text)
progress_path=out/'protocol/progress.json';progress=json.loads(progress_path.read_text());steps=progress['steps']
steps['F1']['completed']+='; H5 calibration; family/boundary ablation running'
steps['F4']['status']='multi_horizon_development_diagnostic_complete';steps['F4']['remaining']=['repeat-measurement controls and final locked test'];steps['F4']['evidence'].append('lewm_audit/long_horizon/analysis.json')
steps['F5']['status']='multi_horizon_context_refresh_development_complete';steps['F5']['evidence']=['lewm_audit/effects/development_summary.json','lewm_audit/long_horizon/analysis.json'];steps['F5']['remaining']=['short-horizon and actual correction relinearization at contact transitions','final locked test']
steps['F7']={'status':'development_calibration_complete','evidence':['response_scores/diagnostics.json','response_scores/jerk_input_audit.json'],'remaining':['final locked test','shift adaptation in F10']}
steps['F8']={'status':'calibrated_development_evaluated','evidence':['selection/bank_rules.json','selection/evaluation_summary.json'],'remaining':['final locked test','additional generator comparisons']}
steps['F9']={'status':'common_bank_development_comparison_complete','evidence':['selection/selector_rules.json','selection/evaluation_summary.json'],'remaining':['matched-dimension generator comparisons','classical/LiPo/GROOVE baselines','direct nonlinear WM constraint and soft sensitivity penalty','matched compute/convergence budgets','final locked test']}
progress['verification']['tests']='23 passed, including causal context refresh, selection label invariance and physical boundary constraints';progress['updated_unix']=time.time();write_json(progress_path,progress)
print('Updated results and progress; full goal remains incomplete.')
