"""Report audited fixed-future policy terminal-state diagnostic."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from zj_wam.io import resolve,write_json,sha256
BASE=resolve('artifacts/followup_v1/policy_terminal');read=lambda p:json.loads(p.read_text())


def main():
    m=read(BASE/'manifest.json');a=read(BASE/'analysis.json');raw=read(BASE/'raw_audit.json')
    assert not a['smoke'] and raw['candidate_branches']==384 and raw['families']==12
    assert raw['manifest_sha256']==sha256(BASE/'manifest.json')
    for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    f=pd.read_csv(BASE/'evaluation.csv');assert len(f)==384
    names={'quintic':'Quintic','quintic_terminal':'Quintic＋종료 제약','spline':'Spline','spline_terminal':'Spline＋종료 제약'}
    rows=[]
    for h in m['config']['horizons']:
        for method in m['config']['families']:
            r=next(r for r in a['summary'] if r['horizon']==h and r['method']==method)
            values=[f"{100*r[k]['mean']:.1f}%" for k in ['short_oracle_U','carryover_oracle_U','full_oracle_U']]
            rows.append(f"| {h} | {names[method]} | {r['dimension']} | "+' | '.join(values)+' |')
    terminal=f[f.method.str.endswith('_terminal') & f.active];safe=terminal[terminal.local_safe];late=safe[~safe.full_safe]
    conditional=[]
    for h in m['config']['horizons']:
        g=terminal[terminal.horizon==h];s=g[g.local_safe];v=s[~s.full_safe]
        conditional.append(f"H={h}: 활성 종료 제약 후보 {len(g)}개 중 국소 보존 {len(s)}개, 그중 장기 위반 {len(v)}개({v.seed.nunique()}개 초기 상태)")
    block=f'''<!-- BEGIN POLICY TERMINAL DIAGNOSIS -->
## 정책 chunk의 종료 상태 제약: 고정 후속 명령에서의 장기 검증

Expert 진단에서 관측한 종료 제약의 보존 효과를 frozen 정책이 생성한 동작으로 확장했다. **개발용 12개 초기 상태·두 수정 길이·384개 후보**를 실행하고 독립 감사했다. 매번 처음 수정할 수 있는 t=16(1.6초)에서 시작한다. H=5는 알려진 8행동 중 앞 5개만 수정하고 나머지 3개를 유지하며, H=8은 8개 전체에 correction을 적용한다. 두 조건 모두 현재 알려진 8행동의 실제 pusher jerk를 최소화한다. 미래 정책 명령이나 물체 결과는 optimizer에 제공하지 않는다.

**이번 후보 집합에서는 보존과 유용한 jerk 감소를 함께 달성하지 못했다.** 종료 제약이 없는 후보는 H=5에서 25.0%, H=8에서 family에 따라 41.7–50.0%의 상태에서 국소 보존과 jerk 10% 감소를 달성했지만, 그 개선과 30초 물체 보존을 함께 유지한 상태는 없었다. 종료 제약 후보는 두 H 모두 국소 jerk 10% 감소를 달성하지 못했다. H=5 종료 제약의 국소 보존 후보 48개는 모두 장기 보존을 유지했지만, H=8에서는 국소 보존 후보 46개 중 10개(3개 초기 상태)가 장기 보존에 실패했다. 따라서 pusher 종료 위치·속도의 일치만으로 물체의 장기 보존을 보장할 수 없다. 종료 제약 후보의 국소 jerk 감소는 H=5에서 최대 약 0.97%, H=8에서 최대 약 3.07%였다. H=8의 국소 통과 후 장기 위반은 episode 3.51–24.20초에 처음 발생했다. Correction은 2.4초에 끝났으므로 가장 늦은 위반은 그로부터 약 21.8초 뒤였다.

수정 후에는 nominal 정책 실행에서 기록한 **상대 명령**을 끝까지 재생했다. 새로운 관측을 입력하여 정책을 다시 호출하지 않았다. 같은 상대 명령이라도 pusher의 종료 상태가 다르면 이후 실제 경로는 달라질 수 있다. 종료 제약은 이 위치·속도 차이를 줄이는 조건이며 물체의 위치·속도를 같게 만드는 조건은 아니다.

물체 보존은 **같은 초기 상태의 nominal 실행과 동일 시각의 물체 위치·회전 궤적**을 비교한다. 100Hz 위치 2px·회전 2도 및 각 제어 시점 coverage 감소 0.02를 검사했다. 국소 판정은 두 H 모두 같은 8행동(0.8초), 전체 판정은 원래 300행동(30초)이며 국소 위반을 전체 판정에서 지우지 않는다. Pusher는 jerk를 줄이기 위해 다른 경로를 갈 수 있다.

| 수정 H | Family | 자유도 | 국소 보존＋국소 jerk 10% 감소 | 앞의 개선＋30초 물체 보존 | 30초 보존＋전체 jerk 10% 감소 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

각 비율은 **네 radius 중 조건을 만족하는 후보가 하나라도 있는 초기 상태의 비율**이다. 사후 oracle 진단이며 실제 승인 규칙의 성능이 아니다. 같은 상태의 후보들은 독립 표본이 아니므로 384개를 독립 성공·실패로 세지 않는다. 12개 초기 상태 단위의 구간과 종료 제약 유무의 paired 차이는 분석 파일에 제공한다.

국소 보존 이후의 지속성은 다음과 같다. {'; '.join(conditional)}. 종료 제약 후보 전체의 실제 종료 pusher 차이는 최대 위치 {terminal.terminal_q.max():.3g}px·속도 {terminal.terminal_v.max():.3g}px/s였다. 이 유한 후보의 관측 결과로 새로운 상태나 무한 시간의 보존을 보장하지 않는다.

짧은 구간에서 줄인 jerk가 30초 전체 에너지의 10%에 못 미칠 수 있으므로, **국소 개선을 장기 물체 보존과 함께 유지하는지**와 **전체 jerk를 충분히 줄이는지**를 분리했다. H를 바꾸면 수정 길이와 표현 자유도도 함께 바뀌므로 horizon 하나의 효과로 해석하지 않는다. 특히 Quintic은 종료 제약 후 자유도가 0인 대조군이다.

이는 **고정 후속 명령에서 한 번 수정하는 진단**이다. 자기 관측에 따라 정책을 재계획하거나 수정이 반복되는 조건, LeWM을 이용한 후보 승인, 별도 최종 test의 결과는 아니다. 그런 평가에서는 같은 종료 제약이 유용한 수정 기회를 얼마나 유지하는지도 다시 확인해야 한다.

근거: [사전 설정](../configs/followup_policy_terminal.json), [전체 분석](../artifacts/followup_v1/policy_terminal/analysis.json), [독립 물리 감사](../artifacts/followup_v1/policy_terminal/raw_audit.json), [후보별 결과](../artifacts/followup_v1/policy_terminal/evaluation.csv).
<!-- END POLICY TERMINAL DIAGNOSIS -->
'''
    path=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');doc=path.read_text();begin='<!-- BEGIN POLICY TERMINAL DIAGNOSIS -->';end='<!-- END POLICY TERMINAL DIAGNOSIS -->'
    if begin in doc:doc=doc[:doc.index(begin)]+block+doc[doc.index(end)+len(end):]
    else:
        anchor='<!-- END TERMINAL RESIDUAL -->';assert anchor in doc;doc=doc.replace(anchor,anchor+'\n\n'+block,1)
    path.write_text(doc)
    write_json(BASE/'report_evidence.json',{'inputs':{name:sha256(BASE/name) for name in ['manifest.json','analysis.json','raw_audit.json','evaluation.csv']},'source_sha256':sha256(Path(__file__)),'scope':'12development seeds384fixedfuture branches; no finaltest or deployableverifier claim.'})
    print('REPORTED policy terminal diagnosis',flush=True)
if __name__=='__main__':main()
