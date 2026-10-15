"""Write audited locked expert-test findings; never change selection settings."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
from scripts.run_followup_final_expert import verify_lock


def main():
    lock=verify_lock();base=resolve('artifacts/followup_v1');out=base/'final_expert/test'
    load=lambda name:json.loads((out/name).read_text())
    banks=load('bank_audit.json');sel=load('selection.json');runtime=load('runtime.json');shifts=load('shifts.json');repeated=load('repeated.json')
    for a in [banks,sel,runtime,shifts,repeated]:assert not a['partial']
    assert all(v['states']==45 for v in banks['banks'].values());assert all(v==45 for v in runtime['states'].values());assert all(v==45 for v in shifts['states'].values());assert all(v==45 for v in repeated['states'].values())
    primary=next(r for r in sel['paired'] if r['primary']);d=primary['U_difference'];lo,hi=d['ci95'];p=100*d['mean'];cut=lock['config']['minimum_practical_U_difference']
    bymethod={r['method']:r for r in sel['summary'] if r['pipeline']=='common_bank' and r['stratum']=='all'};contact={r['method']:r for r in sel['summary'] if r['pipeline']=='common_bank' and r['stratum']=='contact'};m=bymethod['lewm_global'];risk=m['risk']['mean'];cr=contact['lewm_global']['risk']['mean'];qualified=risk is not None and cr is not None and max(risk,cr)<=.1 and m['active']['mean']>=.25
    if lo>=cut and qualified:judgment='선언한 실질적 개선 폭과 관측 위험·적용 비율 조건을 충족했다.'
    elif lo>0 and qualified:judgment='양의 차이를 뒷받침하지만 선언한 실질적 개선 폭까지 입증하지는 못했다.'
    else:judgment='사전 선언한 조건에서 LeWM의 추가 가치를 입증하지 못했다. 동등성이나 모든 환경에서의 무용성을 뜻하지 않는다.'
    def pct(x):return '미정의' if x is None else f'{100*x:.1f}%'
    table=[]
    for method in ['controller_pusher','lewm_global','vision_contact','vision_adaptive','lewm_object','lewm_history_object','state_contact','state_adaptive']:
        r=bymethod[method];table.append(f"| {method} | {pct(r['joint']['mean'])} | {pct(r['risk']['mean'])} | {pct(r['active']['mean'])} |")
    strong=[]
    for r in sel['summary']:
        if r['pipeline']=='strong_locked' and r['stratum']=='all':strong.append(f"| {r['method']} | {pct(r['joint']['mean'])} | {pct(r['risk']['mean'])} |")
    block=f'''<!-- BEGIN FINAL EXPERT TEST -->
## 잠근 PushT expert 최종 평가

독립 downstream test 15개 초기 상태 family·45개 상태에서 사전 선언한 실험을 완료하고 원시 결과를 감사했다. Encoder·predictor·probe·승인 threshold·radius와 calibration 비용 profile은 test에서 다시 학습하거나 선택하지 않았다. 이 분리는 WM 사전학습 데이터의 비노출을 보장하지 않으며, 독립 family 15개는 낮은 위반 확률을 정밀하게 추정하기에 제한적이다.

**주 비교: 같은 후보에서 LeWM을 추가한 효과.** LeWM global−controller의 동시 달성률 U 차이는 {p:+.2f}%p, family bootstrap 95% 구간 [{lo*100:+.2f},{hi*100:+.2f}]%p다. {judgment} 선언한 실질적 개선 폭은 U 10%p이고, 물체 위치 2px·회전 2도·coverage 감소 0.02, 실제 jerk 10% 감소 기준을 유지했다. 관측 위험 조건은 전체 및 접촉의 선택 후 위반율 10% 이하, 최소 수정 적용 비율 25%다. 이는 경험적 조건이며 확률적 안전 보증이 아니다.

| 같은 후보의 선택기 | U | 선택 후 위반율 | 수정 적용 비율 |
| --- | --- | --- | --- |
{chr(10).join(table)}

`lewm_object`·`lewm_history_object`는 물리 label로 학습한 readout을 추가한 감독 변형이다. `state_*`는 simulator 물체 상태를 직접 제공받는 별도 정보 조건이며, 이미지 입력 방법과 정보 접근이 같지 않다. 보조 비교는 탐색적인 구간으로 보고하고 다중 비교 전체의 확증적 우위를 주장하지 않는다.

| 고정된 강한 smoothing 설정 | U | 선택 후 위반율 |
| --- | --- | --- |
{chr(10).join(strong)}

Full-command 필터·기하 방법은 수정 경계 자유도가 더 넓다. 이 표의 차이를 물체 모델 정보 자체의 인과 효과로 해석하지 않는다. 같은 차원의 low/random/controller 방향과 고정 verifier 비교는 별도의 `family_*` 결과에 유지했다. Oracle은 실제 실행한 후보 중 보존 조건을 충족하는 최선이며 배포 가능한 선택기나 전역 최적해가 아니다.

최종 원시 감사는 H5 후보 bank 4종과 H10·H20 진단의 모든 후보에 대해 동일 시간의 물체 pose, 독립적으로 재계산한 goal coverage, prefix와 suffix를 포함한 실제 jerk, action norm·한계를 확인했다. 선택용 CSV는 같은 원시 후보와 연결되는지도 검사했다. 실제 실행 비용, 8개 물리·관측 조건 변화, H20 반복 수정은 각각 별도 결과를 제공하며 누적 기준을 국소 기준으로 바꾸지 않았다.

근거: [사전 잠금](../artifacts/followup_v1/protocol/final_selection_lock.json), [원시 후보 감사](../artifacts/followup_v1/final_expert/test/bank_audit.json), [선택·paired 비교](../artifacts/followup_v1/final_expert/test/selection.json), [capture](../artifacts/followup_v1/final_expert/test/capture.json), [효과 예측 진단](../artifacts/followup_v1/final_expert/test/effect_diagnostics.json), [고정 비용 budget](../artifacts/followup_v1/final_expert/test/runtime.json), [조건 변화](../artifacts/followup_v1/final_expert/test/shifts.json), [반복 실행](../artifacts/followup_v1/final_expert/test/repeated.json).

이 절은 PushT expert의 최종 평가다. 정책·Cube 최종 평가, 추가 모델·task·실물 검증까지 포함한 전체 F0–F11 완료를 뜻하지 않는다.
<!-- END FINAL EXPERT TEST -->
'''
    p=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=p.read_text();start='<!-- BEGIN FINAL EXPERT TEST -->';end='<!-- END FINAL EXPERT TEST -->'
    if start in s:s=s[:s.index(start)]+block+s[s.index(end)+len(end):]
    else:
        header=s.index('\n\n',s.index('\n\n')+2);s=s[:header]+'\n\n'+block+s[header:]
    s=s.replace('최종 test intervention은 아직 실행하지 않았다.','PushT expert 최종 test 결과를 아래에 별도로 보고한다. 정책·Cube와 추가 모델·task 평가의 완료 여부는 구분한다.')
    p.write_text(s);write_json(out/'report_evidence.json',{'lock_sha256':sha256(base/'protocol/final_selection_lock.json'),'analysis_sha256':{name:sha256(out/name) for name in ['bank_audit.json','selection.json','runtime.json','shifts.json','repeated.json','capture.json','effect_diagnostics.json']},'report_sha256':sha256(p),'scope':'Locked expert test only, not full goal completion'});print('FINAL EXPERT REPORT WRITTEN',flush=True)
if __name__=='__main__':main()
