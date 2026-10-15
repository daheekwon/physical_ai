"""Report audited common-gradient controls without selecting a test-time method."""
import json,sys,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'gradient_controls';read=lambda p:json.loads(p.read_text())
def main():
    done=read(OUT/'pipeline_complete.json');audit=read(OUT/'raw_audit.json');a=read(OUT/'analysis.json')
    assert done['raw_audit_sha256']==sha256(OUT/'raw_audit.json') and audit['analysis_sha256']==sha256(OUT/'analysis.json')
    assert not audit['smoke'] and audit['states']==105 and audit['families']==35 and audit['candidates']==7560
    assert audit['source_sha256']==sha256(resolve('scripts/analyze_followup_gradient_controls.py'))
    for p,h in read(OUT/'pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p
    f=pd.read_csv(OUT/'state_metrics.csv');assert len(f)==630 and f.state_id.nunique()==105
    lookup={r['method']:r for r in a['summary']};methods=read(resolve('configs/followup_gradient_controls.json'))['methods']
    for name,r in lookup.items():
        g=f[f.method==name]
        for metric in ['eta','oracle_useful','controller_active','controller_useful','controller_gain','unique_actions']:
            np.testing.assert_allclose(g[metric].mean(),r[metric]['mean'],atol=1e-12)
        np.testing.assert_allclose(g[g.controller_active].controller_violation.mean(),r['controller_selected_risk']['mean'],atol=1e-12)
    signed=[];additional={x:[] for x in methods}
    for path in sorted(OUT.glob('development*_h05.csv')):
        z=pd.read_csv(path);z['useful']=(z.norm>1e-6)&(z.object_error<=1)&(z.jerk_gain>=.1)
        common=z[(z.method=='gradient_lewm')&(z.kind=='signed')].useful.any();signed.append(bool(common))
        for name in methods:additional[name].append(bool(z[z.method==name].useful.any() and not common))
    names={'gradient_random0':'Gradient＋무작위 0','gradient_random1':'Gradient＋무작위 1','gradient_random2':'Gradient＋무작위 2','gradient_controller':'Gradient＋제어기 저반응','gradient_lewm':'Gradient＋LeWM 저반응','full':'전체 4차원 참고'}
    lines=['<!-- BEGIN GRADIENT CONTROLS -->','## 추가 C: 공통 jerk gradient 이후 LeWM 방향의 추가 가치','',
    '**개발 family 35개·상태 105개·후보 7,560개**를 실행하고 원시 궤적에서 독립 검증했다. 모든 2차원 공간의 첫 축을 동일한 제어기 기반 jerk gradient로 고정하고, 나머지 축만 무작위 세 seed·제어기 저반응·LeWM 저반응으로 바꿨다. 저반응 축은 gradient의 직교 여공간에 제한한 민감도 행렬의 최소 고유벡터다. 이전 추가 A의 LeWM 축을 순서대로 직교화한 방식과 다르므로 별도 실험으로 기록한다.','',
    '물체 보존은 같은 시각의 위치 2px·회전 2도 및 끝점 coverage 감소 0.02 조건이며, 100Hz 수정 구간과 0.3초 suffix를 포함한다. U는 이 보존 조건과 실제 jerk 10% 감소를 함께 달성한 상태 비율이다. 모든 공간의 η=1은 공통 gradient를 포함시킨 구성상 결과이며 LeWM의 발견이 아니다.','',
    '| 공간 | Oracle U, % | 공통 verifier U, % | 선택 후 위반율, % | 적용 비율, % | 평균 실제 jerk 감소, % | 공통 signed 후보 외 추가 oracle 성공, %p |','| --- | --- | --- | --- | --- | --- | --- |']
    for name in methods:
        r=lookup[name];lines.append(f"| {names[name]} | {100*r['oracle_useful']['mean']:.2f} | {100*r['controller_useful']['mean']:.2f} | {100*r['controller_selected_risk']['mean']:.2f} | {100*r['controller_active']['mean']:.2f} | {100*r['controller_gain']['mean']:.3f} | {100*np.mean(additional[name]):.2f} |")
    lines+=['',f"모든 방법이 공유하는 양·음 gradient 후보만의 oracle U는 {100*np.mean(signed):.2f}%다. 각 방법은 네 radius에서 같은 signed 후보 8개와 자기 공간의 최적화 후보 4개를 요청한다. 따라서 signed 후보의 중복은 의도한 대조이며, 공간 간 차이는 최적화 후보에서 발생한다. 실제 고유 후보 수·활성 후보 수·solver 성공률도 상태별 원본에 보고한다. 전체 4차원 결과는 더 큰 자유도의 참고치다.",'',
    '| LeWM 공간 − 대조 공간 | Oracle U 차이, %p [family 95% 구간] | Verifier U 차이, %p [family 95% 구간] |','| --- | --- | --- |']
    for name in [x for x in methods if x!='gradient_lewm']:
        cells=[]
        for metric in ['oracle_useful','controller_useful']:
            r=next(r for r in a['paired_contrasts'] if r['method']==name and r['metric']==metric)
            cells.append(f"{-100*r['mean']:+.2f} [{-100*r['ci95'][1]:+.2f}, {-100*r['ci95'][0]:+.2f}]")
        lines.append(f"| {names[name]} | {' | '.join(cells)} |")
    assert not any(additional['gradient_lewm'])
    np.testing.assert_allclose(lookup['gradient_lewm']['controller_useful']['mean'],lookup['gradient_controller']['controller_useful']['mean'],atol=0)
    lines+=['',
    '**현재 개발 비교에서는 LeWM 보조 방향의 추가 이점을 확인하지 못했다.** LeWM의 oracle U 27.62%는 공통 signed gradient 후보만으로 얻는 값과 같았고, LeWM 공간의 최적화 후보는 추가 동시 성공 상태를 만들지 못했다. 제어기 공간의 oracle U는 28.57%, 무작위 공간은 28.57–31.43%였다. 실행 전 선택의 U는 LeWM·제어기 모두 8.57%, 무작위는 7.62–8.57%였다. LeWM과 제어기의 평균 실제 jerk 감소는 각각 4.016%, 4.021%로 차이가 매우 작았다. 평균 jerk는 fallback과 위반 사례까지 포함한 전체 상태 평균이며 보존된 상태만의 성능으로 해석하지 않는다.', '',
    '세 무작위 seed는 같은 상태에서 반복한 대조이며 독립 family 수를 늘리지 않는다. 표의 구간은 상태를 family 단위로 묶은 탐색적 paired 구간이고 다중 비교 보정된 확증 결과가 아니다. Oracle은 실제 미래 결과를 아는 유한 후보 진단이다. 실제 verifier에는 기존 controller radius·pusher 변위 규칙을 그대로 전이했으며 새 후보 분포에 다시 보정하지 않았다. 따라서 이 결과를 새로운 위험 보증이나 독립 최종 test로 해석하지 않는다.','',
    '해석은 후보의 잠재 이점과 실행 전 선택의 이점을 분리한다. LeWM의 oracle U가 더 높아도 verifier U의 차이가 불확실하면 배포 가능한 추가 가치가 입증된 것은 아니다. 무작위·제어기 대조보다 개선되지 않으면 gradient를 포함한 사실만으로 앞선 추가 A의 개선을 설명할 수 있는지 먼저 검토한다. 이 진단을 반영해 새 방법을 선택하려면 새로운 calibration과 최종 평가가 필요하다.','',
    '독립 감사는 기존 nominal trace 재현, 원시 pose 기반 polygon coverage, 물체 궤적·실행 jerk, 별도 PD 적분, 유한차분 gradient, 제한된 최소 고유값, 무작위 seed 재현, 공통 gradient와 signed 행동, 다항식 및 subspace 소속, radius·선택 규칙을 포함한다. 기존 LeWM Jacobian을 재사용했으므로 이 실험의 생성 소요 시간을 온라인 LeWM 전체 추론 지연으로 보고하지 않는다.','',
    '근거: [고정 설정](../configs/followup_gradient_controls.json), [실행 잠금](../artifacts/followup_v1/gradient_controls/pipeline_lock.json), [분석과 paired 구간](../artifacts/followup_v1/gradient_controls/analysis.json), [독립 원시 감사](../artifacts/followup_v1/gradient_controls/raw_audit.json), [상태별 결과](../artifacts/followup_v1/gradient_controls/state_metrics.csv).','<!-- END GRADIENT CONTROLS -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();start=lines[0];end=lines[-1]
    if start in s:
        i=s.index(start);j=s.index(end,i)+len(end);s=s[:i]+section+s[j:]
    else:
        i=s.index('<!-- BEGIN DIRECTION UTILITY -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'analysis_sha256':sha256(OUT/'analysis.json'),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'common_signed_oracle_U':float(np.mean(signed)),'extra_oracle_U':{k:float(np.mean(v)) for k,v in additional.items()},'unix':time.time()});print('REPORTED gradient controls',flush=True)
if __name__=='__main__':main()
