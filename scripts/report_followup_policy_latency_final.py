"""Report only a fully audited independent latency test, with both references."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,write_json,sha256

BASE=resolve('artifacts/followup_v1/policy_latency');DEST=BASE/'test/final_analysis';read=lambda p:json.loads(p.read_text())

def report():
    from scripts.run_followup_policy_latency_final import verify,LOCK
    lock=verify();result=read(DEST/'analysis.json');audit=read(DEST/'raw_audit.json');assert not result['validation_only'] and not audit['validation_only'];assert audit['episodes']==216
    assert result['lock_sha256']==audit['lock_sha256']==sha256(LOCK)
    assert audit['source_sha256']==sha256(resolve('scripts/analyze_followup_policy_latency_final.py'))
    assert read(BASE/'final_execution_complete.json')['episodes']==216
    primary=next(x for x in result['paired'] if x['method']=='lewm_global' and x['profile']=='full' and x['reference']=='same_profile');delta=primary['joint'];lo,hi=delta['ci95']
    pct=lambda x:'미정의' if x is None else f'{100*x:.1f}%'
    rows=[]
    for method in lock['config']['methods']:
        same=next(r for r in result['summary'] if r['method']==method and r['profile']=='full' and r['reference']=='same_profile');original=next(r for r in result['summary'] if r['method']==method and r['profile']=='full' and r['reference']=='zero_nominal')
        rows.append(f"| {method} | {pct(same['joint']['mean'])} | {pct(same['active_risk']['mean'])} | {pct(original['preserved']['mean'])} | {pct(same['any_success']['mean'])} | {pct(same['hold_fraction']['mean'])} |")
    verdict='사전 선언한 경험적 조건과 실질적 개선 기준을 충족했다.' if result['primary_decision']['supported_practical_advantage'] else '사전 선언한 조건에서 LeWM의 실질적인 추가 우위를 입증하지 못했다.'
    block=f'''<!-- BEGIN POLICY LATENCY FINAL TEST -->
## 물리 대기 시간을 포함한 정책의 독립 최종 평가

새로운 초기 상태 seed 12개에서 6방법×3지연 조건, **216개 episode**를 완료하고 원시 물리 trace를 감사했다. 먼저 지연 development 72개 전체를 별도 최종 감사기로 재현한 뒤, timing 분포·정책·선택 규칙·분석을 잠갔다. Test에서는 어떤 모델이나 threshold도 다시 학습·보정하지 않았다.

주 조건은 관측 획득·정책·refinement 지연을 모두 포함한다. 그 조건의 수정 없는 정책을 기준으로 한 LeWM−제어기 **U 차이는 {100*delta['mean']:+.2f}%p**, paired family 95% 구간 [{100*lo:+.2f},{100*hi:+.2f}]%p다. {verdict} 미리 정한 실질적 개선 폭은 U 10%p, 경험적 위반율 목표는 10%, 최소 수정 적용 비율은 25%다.

| 방법 | 같은 지연 nominal 대비 U | 수정 episode 중 위반율 | 원래 무지연 nominal 궤적 보존율 | Task 성공률 | 대기 시간 비율 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

**두 기준 궤적은 의미가 다르다.** 같은 지연 조건의 nominal과 비교한 결과는 지연된 정책 실행에 smoothing을 추가한 효과다. 이것이 지연 없이 원래 의도한 시간 정렬 물체 궤적까지 보존했다는 뜻은 아니다. 따라서 무지연 nominal 대비 보존율도 함께 제시했다. Task 성공률도 궤적 보존의 대체 지표로 사용하지 않는다.

각 episode는 대기를 포함해 300개 제어 시점(30초)으로 고정했다. 요청 시점의 자기 관측으로 명령을 계산하고, 준비될 때까지 마지막 절대 목표 위치를 유지한 뒤 저장한 상대 명령을 적용했다. 실행 시간이 부족한 마지막 명령은 일부만 또는 전혀 전달되지 않을 수 있으며 실제 전달된 정규화 correction으로만 적용 여부를 판정했다. 늦게 도착한 명령을 사후에 다시 최적화하지 않았다.

지연 없음·정책 지연만·전체 지연의 세 조건을 모두 원본 분석에 제공한다. 지연은 전체 calibration에서 측정한 분포를 물리 시뮬레이션에 재생한 것이며 이 실행의 wall time을 그대로 주입한 것이 아니다. 공유 GPU에서 측정한 시간과 직렬 요청·대기·8행동 실행 방식의 결과이며 최적 비동기 제어나 실물 지연을 검증한 결과로 확장하지 않는다.

물체 pose는 100Hz, coverage는 제어 시점에서 검사하고 독립 polygon 계산으로 확인했다. 대기·전달 시각, causal history, 명령 변환, 같은 초기 상태, 국소 반사실 비교와 전체 jerk를 재검사했다. Episode 지표의 정확 이항 구간과 paired family 구간을 함께 제공한다. 12개 family에서 실패가 0개여도 95% 상한은 약 26.5%이므로 낮은 모집단 위험을 인증하지 않는다.

근거: [독립 지연 test 잠금](../artifacts/followup_v1/policy_latency/final_protocol.json), [최종 분석](../artifacts/followup_v1/policy_latency/test/final_analysis/analysis.json), [원시 감사](../artifacts/followup_v1/policy_latency/test/final_analysis/raw_audit.json), [개별 episode 평가](../artifacts/followup_v1/policy_latency/test/final_analysis/evaluation.csv).
<!-- END POLICY LATENCY FINAL TEST -->
'''
    p=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');doc=p.read_text();begin='<!-- BEGIN POLICY LATENCY FINAL TEST -->';end='<!-- END POLICY LATENCY FINAL TEST -->'
    if begin in doc:doc=doc[:doc.index(begin)]+block+doc[doc.index(end)+len(end):]
    else:
        index=doc.index('\n\n',doc.index('\n\n')+2);doc=doc[:index]+'\n\n'+block+doc[index:]
    p.write_text(doc);write_json(DEST/'report_evidence.json',{'inputs':{name:sha256(DEST/name) for name in ['analysis.json','raw_audit.json','evaluation.csv']},'lock_sha256':sha256(LOCK),'source_sha256':sha256(Path(__file__)),'scope':'Independent serial-latency simulation only, not fullF0-F11 completion'})
    print('REPORTED latency final216episodes',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-pid',type=int);args=p.parse_args()
    if args.dependency_pid:
        start=time.monotonic();print('WAITING for full216 latencytest audit, no partial report',flush=True)
        while not (DEST/'analysis.json').exists() or not (DEST/'raw_audit.json').exists():
            try:os.kill(args.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError('Latency final parent ended before audit; inspect logs')
            if time.monotonic()-start>21600:raise RuntimeError('Observation window elapsed; revalidate parent')
            time.sleep(10)
    report()
if __name__=='__main__':main()
