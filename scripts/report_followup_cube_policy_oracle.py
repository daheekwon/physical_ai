"""Report finite oracle gap without treating future labels as a deployable method."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_policy_oracle';read=lambda p:json.loads(p.read_text())

def main():
    a=read(OUT/'analysis.json');audit=read(OUT/'raw_audit.json');m=read(OUT/'manifest.json');assert not a['smoke'] and a['states']==48 and audit['analysis_sha256']==sha256(OUT/'analysis.json') and audit['manifest_sha256']==sha256(OUT/'manifest.json');f=pd.read_csv(OUT/'selected.csv');sf=pd.read_csv(OUT/'state_potential.csv');assert len(sf)==48 and sf.seed.nunique()==12
    for p,h in audit['inputs'].items():assert sha256(resolve(p))==h,p
    for r in a['summary']:
        v=f[f.method==r['method']];v=v if r['phase']=='all' else v[v.phase==r['phase']]
        for metric in ['active','joint','jerk_gain','predicted_gain','selected_risk']:
            g=v[v.active] if metric=='selected_risk' else v;col='violation' if metric=='selected_risk' else metric
            if not len(g):assert r[metric]['mean'] is None;continue
            mean,ci=cluster(g[col].to_numpy(float),g.initial_pose_family.to_numpy());np.testing.assert_allclose([mean,*ci],[r[metric]['mean'],*r[metric]['ci95']],rtol=0,atol=1e-10)
    for metric,record in a['candidate_potential'].items():
        mean,ci=cluster(sf[metric].to_numpy(float),sf.initial_pose_family.to_numpy());np.testing.assert_allclose([mean,*ci],[record['mean'],*record['ci95']],rtol=0,atol=1e-10)
    names={'controller_displacement':'제어기 verifier','proprio_contact':'접촉 proxy','lewm_global':'LeWM verifier','unrestricted_expected':'물체 제약 없이 예상 jerk 최소','physical_filter_expected':'실제 보존 필터＋예상 jerk 순위','finite_physical_oracle':'실제 보존 필터＋실제 jerk 순위','unrestricted_physical_oracle':'물체 제약 없이 실제 jerk 최소'};lines=['<!-- BEGIN CUBE POLICY FINITE ORACLE -->','## Cube 정책 후보 공간: 승인 전 가능성과 유한 물리 oracle','',f"기존 nominal compiler의 **12개 초기 상태·48개 상태·{a['candidates']:,}개 후보**를 실제로 분기 실행했다. 접근·파지·운반·해제는 후보 결과를 보기 전에 nominal의 현재 접촉·높이로 정했다. 각 root는 이전 nominal 실행을 그대로 재생했고, 수정 없는 H5＋3 분기는 저장된 물리 trace와 정확히 일치했다.",'','모든 실제 verifier는 같은 controller-aware 다항식 후보, 같은 예상 EEF 병진 jerk와 기존 expert calibration 규칙을 사용한다. 현재까지의 입력으로 후보와 선택을 확정한 뒤 물리 분기를 실행했다. 실제 보존 필터와 실제 jerk 순위는 사후 원인 진단이며 실행 전에 쓸 수 있는 방법이 아니다.','','| 선택 기준 | 국소 U, % | 수정 적용, % | 활성 수정의 물체 위반, % | 평균 실제 jerk 감소, % |','| --- | --- | --- | --- | --- |']
    for method in names:
        r=next(x for x in a['summary'] if x['phase']=='all' and x['method']==method);risk='—' if r['selected_risk']['mean'] is None else f"{100*r['selected_risk']['mean']:.2f}";lines.append(f"| {names[method]} | {100*r['joint']['mean']:.2f} | {100*r['active']['mean']:.2f} | {risk} | {100*r['jerk_gain']['mean']:.2f} |")
    n_pred=int(sf.any_expected_10percent.sum());n_actual=int(sf.oracle_U.sum());lines+=['',f'승인 전 후보 전체에서도 예상 jerk 10% 개선 후보가 존재한 상태는 **{n_pred}/48개**였다. 실제 궤적 보존과 실제 jerk 10% 감소를 함께 달성하는 후보가 존재한 상태는 **{n_actual}/48개**였다. 원시 표에는 단계별 결과와 family 신뢰구간을 함께 제공한다. 48개 상태는 12개 초기 상태에 묶여 있으므로 독립 family 48개로 세지 않는다.','','첫 물리 oracle은 실제 보존을 만족한 후보를 예상 jerk로 정렬한다. 다음 oracle은 같은 보존 후보를 실제 jerk로 정렬한다. 두 결과의 차이는 이 유한 bank에서 순위를 정확히 알 때 회복할 여지이며, 후보 생성·접촉·surrogate의 독립적인 인과 기여로 분해해 더할 수 없다. 물체 제약 없는 실제 최선은 보존을 포기했을 때의 참고값이다.','','물체 보존은 수정 H5와 같은 suffix 3개 전체의 500Hz 위치 2mm·full 회전 2도 기준이다. 유용한 수정은 실제 EEF 병진 jerk 10% 감소를 함께 요구한다. 첫·마지막 행동과 gripper correction은 선언한 다항식 제약을 유지하며 실제 적용 float32 action으로 검사했다. 각 radius와 optimizer 실패 기록을 보존했다.','','감사는 모든 후보의 제약·norm·past-only response·예상 비용, 실제 물체 pose와 jerk, 기존 nominal trace, 모든 선택을 재계산했다. LeWM은 모든 후보에서 native CPU 예측을 다시 확인했고 첫 family의 각 단계에서 첫·마지막 후보 8개를 물리적으로 재실행했다. 이 결과는 전역 최적해, 새로운 정책용 calibration, 독립 최종 test 또는 전체 episode 보존의 증거가 아니다.','','근거: [상태·입력 잠금](../artifacts/followup_v1/cube_policy_oracle/manifest.json), [전체 후보](../artifacts/followup_v1/cube_policy_oracle/candidates.csv), [상태별 가능성](../artifacts/followup_v1/cube_policy_oracle/state_potential.csv), [비교·family 구간](../artifacts/followup_v1/cube_policy_oracle/analysis.json), [독립 감사](../artifacts/followup_v1/cube_policy_oracle/raw_audit.json).','<!-- END CUBE POLICY FINITE ORACLE -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:marker='<!-- END CUBE OBJECTIVE DECISION -->';i=s.index(marker)+len(marker);s=s[:i]+'\n\n'+section+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['analysis.json','raw_audit.json','selected.csv','state_potential.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()});print('REPORTED POLICY ORACLE',a['states'],a['candidates'],flush=True)

if __name__=='__main__':main()
