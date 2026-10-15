"""Report audited long-term effects of existing physical boundary constraints."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json,sha256

OUT=resolve('artifacts/followup_v1/terminal_residual');read=lambda p:json.loads(p.read_text())

def main():
    m=read(OUT/'manifest.json');a=read(OUT/'analysis.json');audit=read(OUT/'raw_audit.json');assert not a['smoke'] and audit['states']==210 and audit['candidate_rows']==5040
    assert audit['manifest_sha256']==sha256(OUT/'manifest.json')
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    f=pd.read_csv(OUT/'evaluation.csv');assert len(f)==5040
    names={'quintic_full':'Quintic','quintic_full_terminal':'Quintic＋종료 제약','interior_values':'Interior','interior_values_terminal':'Interior＋종료 제약','cubic_spline':'Spline','cubic_spline_terminal':'Spline＋종료 제약'}
    rows=[]
    for name in m['config']['families']:
        r=next(x for x in a['summary'] if x['split']=='development' and x['method']==name)
        rows.append(f"| {names[name]} | {r['dimension']} | {100*r['short_oracle_U']['mean']:.1f}% | {100*r['carryover_oracle_U']['mean']:.1f}% | {100*r['full_oracle_U']['mean']:.1f}% |")
    terminal=f[f.method.isin(['interior_values_terminal','cubic_spline_terminal']) & f.active];shortsafe=terminal[terminal.short_safe];late=shortsafe[~shortsafe.long_safe];pq=float(terminal.terminal_pusher_position.max());pv=float(terminal.terminal_pusher_velocity.max());after=float(terminal.post_correction_pusher_max.max())
    block=f'''<!-- BEGIN TERMINAL RESIDUAL -->
## Pusher 종료 상태 제약의 보존 효과와 수정 자유도

첫 correction의 장기 이탈을 확인한 뒤, 기존 F1의 종료 제약 후보를 더 길게 실행했다. Development·calibration 각각 35개 family·105개 상태, 합계 **210개 상태·5,040개 후보**다. 원래 H=5 correction과 3행동 suffix의 같은 행동을 재생하고, 이후 원래 상대 명령을 이어 총 23행동(2.3초)까지 관찰했다. 새 물체 결과로 correction을 다시 만들거나 후보를 고르지 않았다.

기존 첫 0.8초의 물리 trace와 후보 행동을 재현하고, 물체 pose·독립 polygon coverage·전체 jerk·실제 pusher 종료 상태를 감사했다. 장기 보존은 이전 0.8초의 판정을 유지하면서 긴 구간의 위치·회전 최대 오차와 coverage 조건을 추가한다. 짧은 구간에서 검출한 위반을 긴 구간 평가에서 지우지 않는다.

**종료 제약은 실제 pusher 위치·속도를 원래 값으로 돌려놓는다.** 활성화된 Interior/Spline 종료 제약 후보의 0.5초 시점 최대 위치 차이는 {pq:.3g}px, 속도 차이는 {pv:.3g}px/s였고, 이후 pusher 경로의 최대 차이는 {after:.3g}px였다. 두 확장 family에서 짧은 보존을 통과한 활성 후보 {len(shortsafe):,}개 중 장기 보존에 새로 실패한 후보는 **{len(late):,}개**였다. 즉 관측한 후보들은 모두 2.3초까지 보존을 유지했다. 이는 이 구간과 후보 집합에서 확인한 결과이며, 더 긴 시간이나 새로운 상태의 보존 보장은 아니다. 후보는 같은 상태에서 반복 생성되므로 이 개수를 독립 표본 수로 해석하지 않는다. Pusher의 종료 상태를 맞추는 조건이 물체의 종료 상태까지 동일하게 만든다는 뜻도 아니다.

아래는 **development 105개 상태에서 유한 네 radius 중 해당 후보가 하나라도 존재한 비율**이다. 미래 실행을 보고 고르는 진단 oracle이며 배포 가능한 승인 규칙의 U가 아니다.

| Correction family | 자유도 | 0.8초 보존＋그 구간 jerk 10% 감소 | 앞의 개선＋2.3초 물체 보존 | 2.3초 보존＋전체 jerk 10% 감소 |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

**보존의 지속성은 좋아졌지만 유용한 수정 기회는 줄었다.** Development에서 Spline/Interior의 짧은 개선 가능 비율은 종료 제약으로 46.7%에서 19.0%로 줄었다. 2.3초까지 물체 보존을 요구하면 무제약은 29.5%, 종료 제약은 19.0%이며, 전체 구간 jerk 10% 감소까지 요구한 비율은 둘 다 5.7%였다. 따라서 종료 제약의 종합 우위는 확인하지 못했다. Calibration에서도 짧은 개선 가능 비율은 41.0%에서 5.7%로 감소했고, 전체 구간 개선 비율은 3.8%에서 1.9%로 감소했다.

마지막 열은 전체 구간의 jerk가 실제로 충분히 줄었는지를 묻는다. 짧은 구간에서 10%를 줄여도 긴 전체 구간의 감소율은 그보다 작을 수 있으므로 두 기준을 분리했다. Calibration 결과와 family 단위 paired 구간도 원본 분석에 함께 제공한다.

종료 제약은 수정 자유도를 줄인다. Quintic은 4개에서 0개가 되어 nominal만 남고, 두 확장 family는 6개에서 2개가 남았다. 따라서 종료 제약의 낮은 교란만 보고 유용한 smoothing이라고 판단하지 않는다. 실제 수정 비율, 짧은 개선 가능성, 그 개선을 유지하는 장기 보존, 전체 jerk 감소를 함께 확인해야 한다. 이 H5의 Interior와 Spline 비교를 서로 독립적인 물리 task의 재현으로 세지 않는다.

이번 실험은 **expert의 고정 후속 명령과 기존 controller 조건**에서의 진단이다. 물체 종료 속도까지 고정하거나 정책이 자기 관측으로 재계획하는 실험은 아니며, 새로운 규칙을 독립 test에 적용한 결과도 아니다. 다음 반복 refinement에서는 pusher 상태의 잔류 효과와 물체 상태의 잔류 효과를 구분하고, 종료 제약을 더 큰 다항식 공간과 결합할 때 얻는 실제 개선을 별도로 검증해야 한다.

근거: [사전 진단 조건](../configs/followup_terminal_residual.json), [전체 분석](../artifacts/followup_v1/terminal_residual/analysis.json), [원시 물리 감사](../artifacts/followup_v1/terminal_residual/raw_audit.json), [후보별 결과](../artifacts/followup_v1/terminal_residual/evaluation.csv).
<!-- END TERMINAL RESIDUAL -->
'''
    p=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');doc=p.read_text();begin='<!-- BEGIN TERMINAL RESIDUAL -->';end='<!-- END TERMINAL RESIDUAL -->'
    if begin in doc:doc=doc[:doc.index(begin)]+block+doc[doc.index(end)+len(end):]
    else:
        anchor='<!-- END POLICY FIRST CORRECTION -->';assert anchor in doc;doc=doc.replace(anchor,anchor+'\n\n'+block,1)
    p.write_text(doc);write_json(OUT/'report_evidence.json',{'inputs':{name:sha256(OUT/name) for name in ['manifest.json','analysis.json','raw_audit.json','evaluation.csv']},'source_sha256':sha256(Path(__file__)),'active_terminal_candidates':len(terminal),'short_safe_terminal_candidates':len(shortsafe),'late_failing_terminal_candidates':len(late),'late_failing_terminal_states':int(late.state_id.nunique()),'scope':'Development/calibration expert terminal-state residual diagnosis; no new final-test performance claim.'})
    print('REPORTED terminal residual',len(late),'latefails/',len(shortsafe),'shortsafe terminal candidates',flush=True)
if __name__=='__main__':main()
