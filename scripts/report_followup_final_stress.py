"""Report audited expert-test shifts/repetition without changing locked analysis."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json

ROOT=resolve('artifacts/followup_v1/final_expert')
OUT=ROOT/'test'
read=lambda p:json.loads(p.read_text())

def main():
    evidence=read(OUT/'report_evidence.json')
    assert sha256(resolve('artifacts/followup_v1/protocol/final_selection_lock.json'))==evidence['lock_sha256']
    markers=list(ROOT.glob('*.done.json'));assert len(markers)==17
    for path in markers:
        m=read(path);assert m['exit']=='completed; raw audit is a separate required step' and m['lock_sha256']==evidence['lock_sha256'],path
    for name,h in evidence['analysis_sha256'].items():assert sha256(OUT/name)==h,name
    rep=read(OUT/'repeated.json');shift=read(OUT/'shifts.json');runtime=read(OUT/'runtime.json')
    assert not any(x['partial'] for x in [rep,shift,runtime])
    chosen=['repeat5_lewm','repeat5_observed_pose','repeat5_online_gains','open20_piecewise_vision']
    titles={'repeat5_lewm':'LeWM, 5개 행동마다 반복 수정','repeat5_observed_pose':'관측 pose 기반 접촉 모델, 반복 수정','repeat5_online_gains':'온라인 접촉 보정, 반복 수정','open20_piecewise_vision':'Piecewise 다항식, 20개 행동을 한 번에 수정'}
    rows=[]
    for name in chosen:
        r=next(x for x in rep['summary'] if x['method']==name)
        rows.append(f"| {titles[name]} | {100*r['joint']['mean']:.1f}% | {100*r['risk']['mean']:.1f}% | {100*r['long_risk']['mean']:.1f}% |")
    text='''<!-- BEGIN FINAL EXPERT STRESS -->
## 최종 평가에서 드러난 반복 사용과 조건 변화의 한계

독립 test의 15개 초기 상태 family·45개 상태를 사용한 추가 결과다. 최종 분석 파일 7개의 해시와 전체 17개 실행 단계의 정상 종료·동일 protocol 잠금을 확인했다. 아래는 이미 잠근 보조 실험이며 조건별 threshold를 다시 맞추지 않았다.

**짧은 correction의 승인 규칙을 반복 사용해도 전체 물체 궤적이 보존되는 것은 아니다.** 원래 expert 행동 20개(2초)를 기준으로 수정하고, 추가 원래 행동 3개(0.3초)까지 동일 시각의 물체 pose를 비교했다. 반복 방법은 수정된 자기 관측을 사용하지만 원래 행동을 생성하는 정책은 재계획하지 않는다. 아래 위반율의 분모는 실제 수정을 적용한 상태다.

| 방법 | 보존·jerk 동시 달성률 U | 2초 수정＋0.3초 suffix 위반율 | 최대 1초 suffix까지 확장한 위반율 |
| --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

긴 suffix는 남아 있는 원래 행동 수에 따라 3–10개를 사용하며, 짧은 구간에서 확인한 위반도 유지하는 누적 판정이다. Piecewise 비교는 16개 coefficient를 사용하므로 단일 quintic과 자유도가 같지 않다. 모든 비교는 별도 family 불확실성을 포함한 원본 표와 함께 해석해야 한다. LeWM의 짧은 H5 최종 선택 위반율 약 4.9%를 이 반복 구간의 위험 보증으로 확장할 수 없다.

**관측 잡음에서는 두 물체 모델 계열의 실패 방식이 달랐다.** RGB 채널에 표준편차 15의 잡음을 넣은 조건에서 LeWM의 U는 17.8%였지만 선택 후 위반율이 18.2%로 경험적 목표 10%를 넘었다. 고정·온라인 접촉 모델은 모두 수정을 거절해 적용 비율과 U가 0이었다. 이 경우 조건부 위반율은 미정의이며, 성공적인 smoothing으로 세지 않는다. 마찰 0.2와 같은 잡음을 함께 적용한 조건에서도 LeWM 위반율은 20.5%, 접촉 모델 적용 비율은 0이었다.

마찰 0.8에서는 고정·온라인 접촉 모델의 U가 각각 22.2%·24.4%였지만 위반율은 14.0%·11.6%였다. Nominal에서 경쟁력 있는 방법도 변화된 조건에서 기존 calibration 목표를 유지하지 못할 수 있다. 잡음은 모델의 과거 이미지 입력에 적용하고 물체 평가용 물리 상태는 깨끗하게 유지했다. 각 조건의 수정 효과는 같은 변화된 환경에서 원래 행동을 실행한 결과와 비교했다.

**추론 비용 역시 별도 제약이다.** 같은 후보의 full 설정에서 p95 refinement 시간은 controller 약 43.0ms, LeWM 약 90.8ms, 관측 접촉 모델 약 88.4ms였다. 이 공통 구현에서는 비-WM이라는 이유만으로 비용이 작지 않았다. 정책 생성 비용과 실제 대기 중의 물리 변화는 이 측정에 포함되지 않으므로, 아래 지연 실행 실험에서 따로 검증한다.

이 결과는 국소 허용량만 반복 적용하는 방식의 한계와 관측 품질에 대한 의존성을 보여준다. 누적 오차를 고려하는 규칙, 장기 잔류 효과, 실제 지연을 포함한 실행의 필요성이 남는다. 최종 test에서 발견한 실패를 보고 규칙을 변경한다면 새 개발·calibration·독립 평가로 분리해야 한다.

근거: [반복 실행 최종 분석](../artifacts/followup_v1/final_expert/test/repeated.json), [조건 변화 최종 분석](../artifacts/followup_v1/final_expert/test/shifts.json), [비용 최종 분석](../artifacts/followup_v1/final_expert/test/runtime.json), [보고 근거 무결성](../artifacts/followup_v1/final_expert/test/report_evidence.json).
<!-- END FINAL EXPERT STRESS -->
'''
    path=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');doc=path.read_text();begin='<!-- BEGIN FINAL EXPERT STRESS -->';end='<!-- END FINAL EXPERT STRESS -->'
    if begin in doc:
        start=doc.index(begin);stop=doc.index(end,start)+len(end);doc=doc[:start]+text.strip()+doc[stop:]
    else:
        anchor='<!-- END FINAL EXPERT TEST -->';assert anchor in doc;doc=doc.replace(anchor,anchor+'\n\n'+text,1)
    path.write_text(doc)
    write_json(OUT/'stress_report_evidence.json',{'source_sha256':sha256(Path(__file__)),'final_evidence_sha256':sha256(OUT/'report_evidence.json'),'final_lock_sha256':evidence['lock_sha256'],'completed_stages':17,'inputs':evidence['analysis_sha256'],'scope':'Descriptive secondary final-test stress results; no retuning or new success claim.'})
    print('Final expert stress report updated; all 17 stage records and 7 analysis hashes verified')

if __name__=='__main__':main()
