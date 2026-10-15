"""Publish development direction/jerk-overlap evidence after complete raw audit."""
import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'direction_utility';read=lambda p:json.loads(p.read_text())
def main():
    done=read(OUT/'pipeline_complete.json');audit=read(OUT/'raw_audit.json');a=read(OUT/'analysis.json');assert done['raw_audit_sha256']==sha256(OUT/'raw_audit.json') and audit['analysis_sha256']==sha256(OUT/'analysis.json');assert audit['states']==105 and audit['families']==35 and audit['candidates']==6300 and not audit['smoke'];assert audit['source_sha256']==sha256(resolve('scripts/analyze_followup_direction_utility.py'))
    for p,h in read(OUT/'pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p
    f=pd.read_csv(OUT/'state_metrics.csv');assert len(f)==525 and f.state_id.nunique()==105
    lookup={r['method']:r for r in a['summary']}
    for method,r in lookup.items():
        g=f[f.method==method]
        for metric in ['eta','oracle_useful','controller_active','controller_useful','controller_gain','unique_actions']:
            np.testing.assert_allclose(g[metric].mean(),r[metric]['mean'],atol=1e-12)
    names={'controller_low':'제어기 저반응','lewm_low':'LeWM 저반응','random':'무작위','lewm_plus_gradient':'LeWM＋jerk gradient','full':'전체 공간 (4차원 참고)'}
    lines=['<!-- BEGIN DIRECTION UTILITY -->','## 추가 A: 물체 저반응 방향과 jerk 감소 방향의 관계','',
    '계획서 4.7절의 추가 A를 **개발 family 35개·상태 105개·수정 후보 6,300개**에서 실행했다. 기존 H=5 quintic correction의 4차원 좌표 안에서 제어기 저반응, LeWM 저반응, 무작위, jerk gradient를 포함한 LeWM 기반 공간을 각각 2차원으로 구성했다. 전체 4차원 공간은 더 큰 자유도를 쓰는 참고 조건이다. LeWM은 기존 frozen Jacobian을 재사용했고 새 모델 학습은 없다.','',
    'η는 현재 예상 실행 jerk gradient 중 해당 subspace에 투영되는 성분의 제곱 비율이다. **η가 크면 국소적으로 jerk를 줄일 방향을 많이 포함하지만 물체 궤적 보존을 뜻하지 않는다.** Gradient를 포함한 방법의 η=1은 구성상 성질이며 학습된 모델의 발견으로 세지 않는다. 보존은 위치 2px·회전 2도·coverage 감소 0.02, 100Hz 수정 구간＋0.3초 suffix에서 판정했다.','',
    '| 수정 공간 | 차원 | η 평균 | 유한 후보 oracle U, % | 공통 제어기 verifier U, % | 선택 후 위반율, % | 수정 적용 비율, % |','| --- | --- | --- | --- | --- | --- | --- |']
    for method in ['controller_low','lewm_low','random','lewm_plus_gradient','full']:
        r=lookup[method];risk=r['controller_selected_risk']['mean'];risktext=f'{100*risk:.2f}' if risk is not None else '정의 안 됨';lines.append(f"| {names[method]} | {r['dimension']} | {r['eta']['mean']:.3f} | {100*r['oracle_useful']['mean']:.2f} | {100*r['controller_useful']['mean']:.2f} | {risktext} | {100*r['controller_active']['mean']:.2f} |")
    lines+=['','U는 물체 궤적 보존과 실제 jerk 10% 감소를 함께 달성한 상태의 비율이다. Oracle U는 그 방법에서 실행한 유한 후보 중 그런 수정이 하나라도 존재하는지이며, 전체 연속 공간의 가능성이나 배포 성능이 아니다. 제어기 verifier는 기존 calibration의 radius·pusher 변위 threshold를 모든 공간에 동일하게 적용하고 예상 jerk로 선택했다. **새 후보 분포에 다시 보정하지 않은 전이 진단**이므로 기존 위험 목표가 유지된다는 보장은 없다. LeWM 기반 공간의 후보 생성에는 LeWM 정보가 들어가므로 해당 열을 완전한 비-WM pipeline이라고 부르지 않는다.','',
    '| LeWM 저반응 대비 비교 | Oracle U 차이, %p [family 95% 구간] | 제어기 verifier U 차이, %p [family 95% 구간] |','| --- | --- | --- |']
    for method in ['controller_low','random','lewm_plus_gradient','full']:
        cells=[]
        for metric in ['oracle_useful','controller_useful']:
            r=next(x for x in a['paired_contrasts'] if x['method']==method and x['metric']==metric);lo,hi=r['ci95'];cells.append(f"{r['mean']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]")
        lines.append(f"| {names[method]} | {' | '.join(cells)} |")
    delta=next(r for r in a['paired_contrasts'] if r['method']=='lewm_plus_gradient' and r['metric']=='oracle_useful');chosen=next(r for r in a['paired_contrasts'] if r['method']=='lewm_plus_gradient' and r['metric']=='controller_useful')
    lines+=['',f"**Jerk gradient를 포함한 2차원 공간은 LeWM 저반응 2차원 공간 대비 oracle U를 {delta['mean']*100:.2f}%p 높였다.** 그러나 공통 verifier의 U 차이는 {chosen['mean']*100:.2f}%p이고 95% 구간 [{chosen['ci95'][0]*100:.2f}, {chosen['ci95'][1]*100:.2f}]%p는 0을 포함한다. 저반응 공간의 선택이 유용한 후보를 제한할 수 있지만, 후보를 넓히는 것만으로 실행 전 선택까지 해결되지는 않았다는 개발 근거다.",'',
    '이 비교는 gradient를 추가한 효과이며 LeWM 고유의 추가 가치를 입증하지 않는다. 같은 차원에서 gradient에 무작위 성분을 결합한 대조와 분리하지 않았으므로 남은 LeWM 성분의 기여는 미확인이다. 관측 위반이 0인 방법도 실제 위험이 0이라는 뜻은 아니며, bootstrap이 관측되지 않은 실패를 추정해 주지는 않는다.']
    lines+=['','각 방법·상태는 네 radius에서 gradient 투영 방향의 양·음 두 후보와 제한된 optimizer 한 후보, 총 12개를 요청했다. Signed 후보에는 모든 방법이 함께 만족하는 action-bound radius를 사용해 적용 norm을 맞췄다. Optimizer에는 같은 radius 한도를 주었지만 실제 norm은 달라질 수 있다. 중복·비활성 후보 및 solver 성공률을 원본에 별도 기록했다. 동일 요청 수를 동일한 고유 행동 수나 같은 계산 비용으로 해석하지 않는다.','',
    '독립 검증은 기존 nominal 물리 trace와의 정확한 일치, polynomial/subspace 소속, 공통 radius, 저고유값 공간의 Rayleigh trace, 두 유한차분 크기에서 jerk gradient, 직접 재계산한 PD 실행과 raw 물리 위치, 보존·jerk 및 선택 결과를 포함했다. 최초 smoke의 action 여유 계산에서 float32 nominal과 저장된 float64 bank 간 차이가 발견되어 최초 코드·로그를 보존하고 margin 계산을 명시적 float64로 고정한 뒤 재실행했다.','',
    '이 결과는 방향 선택의 개발 진단이다. η·oracle·실제 선택 결과를 함께 보아 hard low-subspace 제한을 유지할지, gradient를 포함하거나 sensitivity를 soft penalty로 바꿀지 결정한다. 이 결과로 새 방법을 선택한다면 새 calibration과 독립 test가 필요하며, 기존 최종 test 결과를 소급 교체하지 않는다.','',
    '근거: [추가 A 설정](../configs/followup_direction_utility.json), [고정 실행](../artifacts/followup_v1/direction_utility/pipeline_lock.json), [분석·paired 구간](../artifacts/followup_v1/direction_utility/analysis.json), [원시 감사](../artifacts/followup_v1/direction_utility/raw_audit.json), [상태별 결과](../artifacts/followup_v1/direction_utility/state_metrics.csv), [부호·radius 반응](../artifacts/followup_v1/direction_utility/response_curves.csv).','<!-- END DIRECTION UTILITY -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();start='<!-- BEGIN DIRECTION UTILITY -->';end='<!-- END DIRECTION UTILITY -->'
    if start in s:
        i=s.index(start);j=s.index(end,i)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN CUBE CAUSAL CHUNKS -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'analysis_sha256':sha256(OUT/'analysis.json'),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()});print('REPORTED directionutility',flush=True)
if __name__=='__main__':main()
