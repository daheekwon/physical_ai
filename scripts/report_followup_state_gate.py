"""Report the audited state-gate diagnostic without treating it as finaltest."""
import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'state_gate';read=lambda p:json.loads(p.read_text())

def main():
    audit=read(OUT/'raw_audit.json');analysis=read(OUT/'analysis.json');protocol=read(OUT/'protocol.json');rules=read(OUT/'rules.json');assert audit['states']==210 and audit['candidate_rows']==11336 and audit['selected_rows']==840;assert audit['source_sha256']==sha256(resolve('scripts/audit_followup_state_gate.py'));assert audit['analysis_sha256']==sha256(OUT/'analysis.json') and audit['protocol_sha256']==sha256(OUT/'protocol.json')
    for p,h in {**protocol['inputs'],**audit['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    for p,h in read(OUT/'execution_complete.json')['files'].items():assert sha256(OUT/p)==h,p
    def row(method,role):return next(r for r in analysis['summary'] if r['method']==method and r['role']==role and r['phase']=='all')
    def pct(v):return '—' if v is None else f'{100*v:.2f}%'
    lines=['<!-- BEGIN STATE GATE DIAGNOSTIC -->','## 상태별 허용량과 후보별 미래 예측의 분리: 첫 개발 비교','',
           'PushT의 기존 controller/random 다항식 후보에서 **210개 상태·11,336개 후보**를 재사용했다. 수정 크기는 모든 방법에서 같은 상대 radius 0.2 이하로 제한했고, 후보별 예상 실행 jerk 순위도 고정했다. 24개 development family(72개 상태)로 gate를 학습하고, 별도 35개 family(105개 상태)에서 승인 threshold를 보정했다. 나머지 11개 family(33개 상태)는 gate 학습·threshold 보정에서 제외한 진단에 사용했다. 이 11개 family는 과거 연구에서 이미 관찰된 자료이므로 새로운 최종 test가 아니다.','',
           '현재 상태 gate는 LeWM의 현재 실제 영상 latent만, nominal 미래 gate는 원래 행동의 예측 endpoint latent만 받아 상태별 제어기 변위 허용량을 조절한다. 같은 고정 192→8차원 무감독 투영과 같은 ridge 규제를 사용했다. 학습 목표는 상태별 후보의 실제 물체 오차/제어기 변위 비율의 95분위수이며 물리 label을 사용한 감독 변형이다. 실행 전 gate에는 실제 미래 결과를 제공하지 않는다. 두 gate 모두 LeWM 표현을 쓰므로 완전한 비-WM 방법으로 부르지 않는다.','',
           '후보별 미래 방식은 각 수정 행동의 frozen LeWM 예측과 nominal 예측 사이의 global latent 거리를 사용한다. 네 방법은 같은 calibration 위험 목표 10%, 최소 수정 적용 비율 25%, 접촉 후보의 선택 후 위험 조건을 사용한다. 적합한 threshold가 없으면 해당 규칙을 비활성화한다.','',
           '| 방법 | Calibration에서 규칙 채택 | 진단 U | 진단 적용 비율 | 선택 후 위반율 | 평균 실제 jerk 감소 |','| --- | --- | --- | --- | --- | --- |']
    for method in protocol['config']['methods']:
        r=row(method,'diagnostic');lines.append(f"| {method} | {'가능' if rules[method]['enabled'] else '조건 충족 규칙 없음'} | {pct(r['joint']['mean'])} | {pct(r['active']['mean'])} | {pct(r['selected_risk']['mean'])} | {pct(r['jerk_gain']['mean'])} |")
    contrast=next(r for r in analysis['paired'] if r['role']=='diagnostic' and r['baseline']=='controller_only' and r['metric']=='joint');lo,hi=contrast['ci95']
    lines+=['',f"후보별 미래 방식의 controller-only 대비 U 차이는 **{100*contrast['mean']:+.2f}%p**, family bootstrap 95% 구간 [{100*lo:+.2f},{100*hi:+.2f}]%p다. 이 작은 기존 개발 표본의 차이를 독립 최종 평가의 우위로 해석하지 않는다.",'',
            '현재 상태 gate와 nominal 미래 gate가 모두 비활성화됐다면 이는 여기서 고정한 8차원·linear ridge·상태별 단일 배율·calibration 격자로 유용한 규칙을 찾지 못했다는 결과다. 후보별 효과 예측이 원리적으로 필수라거나 더 강한 상태 gate가 불가능하다는 뜻은 아니다. 감독 정보와 표현 입력도 다르므로 모델 내부의 인과 기여를 완전히 식별하는 비교는 아니다.','',
            '독립 감사는 모든 후보의 100Hz 물체 오차·polygon coverage·실제 jerk, action norm, causal PD 실행 예측과 비용, 현재/nominal latent 입력, train-only scaler와 별도 sklearn ridge 재적합, 모든 calibration threshold·840개 선택 및 family 구간을 재계산했다. 물체 보존은 같은 시각 위치 2px·회전 2도와 coverage 감소 0.02, jerk 개선 기준은 10%다.','',
            '이번 실행은 저장된 표현을 재사용한 선택 진단이다. 실제 encoder/predictor 호출을 포함한 계산 비용, 다른 gate 용량, 새 family의 최종 평가가 남아 있다. 기존 최종 평가의 선택 규칙은 이 결과를 보고 수정하지 않았다.','',
            '근거: [설정](../configs/followup_state_gate.json), [분할·입력 잠금](../artifacts/followup_v1/state_gate/protocol.json), [분석](../artifacts/followup_v1/state_gate/analysis.json), [독립 감사](../artifacts/followup_v1/state_gate/raw_audit.json), [선택 결과](../artifacts/followup_v1/state_gate/selected.csv).','<!-- END STATE GATE DIAGNOSTIC -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=doc.read_text();begin='<!-- BEGIN STATE GATE DIAGNOSTIC -->';end='<!-- END STATE GATE DIAGNOSTIC -->'
    if begin in text:i=text.index(begin);j=text.index(end)+len(end);text=text[:i]+section+text[j:]
    else:i=text.index('<!-- BEGIN OUTPUT REPLACEMENT LEWM -->');text=text[:i]+section+'\n\n'+text[i:]
    doc.write_text(text);assert section in doc.read_text();write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{p:sha256(OUT/p) for p in ['protocol.json','analysis.json','raw_audit.json','rules.json','selected.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time(),'scope':'Exploratory gate-fit/calibration/diagnostic comparison; new finaltest and full computation costs pending.'});print('REPORTED state gate210states',flush=True)

if __name__=='__main__':main()
