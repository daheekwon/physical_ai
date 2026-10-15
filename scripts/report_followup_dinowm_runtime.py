"""Publish audited full-inference CPU cost and physical selection results."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_runtime';read=lambda p:json.loads(p.read_text())

def main():
    a=read(OUT/'analysis.json');audit=read(OUT/'raw_audit.json');assert not a['smoke'] and a['states']==402 and audit['analysis_sha256']==sha256(OUT/'analysis.json');f=pd.read_csv(OUT/'evaluation.csv');t=pd.read_csv(OUT/'timings.csv')
    for p,h in audit['inputs'].items():assert sha256(resolve(p))==h,p
    for r in a['summary']:
        v=f[(f.environment==r['environment'])&(f.split==r['split'])&(f.method==r['method'])];tt=t[(t.environment==r['environment'])&(t.split==r['split'])&(t.method==r['method'])];np.testing.assert_allclose([tt.total_seconds.median(),tt.total_seconds.quantile(.95)],[r['p50_seconds'],r['p95_seconds']],rtol=0,atol=1e-10)
        for metric in ['active','joint','jerk_gain','predicted_gain','parent_choice_match','risk','contact_risk']:
            g=v[v.active] if metric in ['risk','contact_risk'] else v
            if metric=='contact_risk':g=g[g.phase=='contact'] if r['environment']=='pusht' else g[g.phase!='approach']
            if not len(g):assert r[metric]['mean'] is None;continue
            col='violation' if metric in ['risk','contact_risk'] else metric;mean,ci=cluster(g[col].to_numpy(float),g.initial_pose_family.to_numpy());np.testing.assert_allclose([mean,*ci],[r[metric]['mean'],*r[metric]['ci95']],atol=1e-10,rtol=0)
    for r in a['paired']:
        v=f[(f.environment==r['environment'])&(f.split==r['split'])];x=v[v.method==r['method']].set_index('state_id');y=v[v.method=='controller'].set_index('state_id').loc[x.index];m,ci=cluster((x[r['metric']].astype(float)-y[r['metric']].astype(float)).to_numpy(),x.initial_pose_family.to_numpy());np.testing.assert_allclose([m,*ci],[r['mean'],*r['ci95']],atol=1e-10,rtol=0)
    lines=['<!-- BEGIN DINO CPU RUNTIME -->','## DINO-WM을 포함한 실제 CPU 추론 비용 비교','',f"PushT 210개·Cube 192개 상태에서 5방법을 각각 3회 새로 실행해 **{a['timings']:,}회 시간 측정과 {a['outcomes']:,}개 선택 결과**를 검증했다. Calibration과 development를 구분한 기존 상태 비교이며 새로운 최종 test는 아니다.",'','모든 방법은 CPU에서 PyTorch 2 threads·BLAS 2 threads를 사용했다. 다항식 후보 생성·최적화, 필요한 영상 전처리·encoder·predictor, 점수 계산과 선택을 매번 실행했다. 환경별 방법당 warm-up 1회를 제외하고 상태 안에서 방법·반복 순서를 무작위화했다. 모델 로딩·disk I/O·영상 획득·rendering·물리 평가는 시간에서 제외했다. 공유 host에서 측정한 구현별 시간이며 최적 하드웨어 성능의 한계로 해석하지 않는다.','','아래는 development의 실제 시간과 물리 결과다. U는 선언한 시간 정렬 물체 궤적 허용 오차와 실제 jerk 10% 감소의 동시 달성률이며, 위험은 선택한 활성 수정 중 실제 위반율이다.','','| 환경 | 방법 | U, % | 적용, % | 위반, % | 시간 p50 / p95, s |','| --- | --- | --- | --- | --- | --- |']
    for r in a['summary']:
        if r['split']!='development':continue
        risk='—' if r['risk']['mean'] is None else f"{100*r['risk']['mean']:.2f}"
        lines.append(f"| {r['environment']} | {r['method']} | {100*r['joint']['mean']:.2f} | {100*r['active']['mean']:.2f} | {risk} | {r['p50_seconds']:.3f} / {r['p95_seconds']:.3f} |")
    changed=int((~f.parent_choice_match).sum());lines+=['',f'기존 점수는 LeWM GPU 추론을 포함한다. 이번 CPU native 점수로 선택이 달라진 구성은 {changed}/{len(f)}개이며 그 실제 선택을 평가했다. 기존 선택을 강제하거나 score 차이를 반올림해 지우지 않았다. Verifier threshold는 기존 calibration 값을 그대로 사용한 이전 실험의 규칙 이전이며, CPU에서도 기존 경험적 위험이 유지된다고 가정하지 않는다.','','후보 action은 기존 물리 실행과 엄격한 허용 오차 1e−9 이내로 일치해야 재사용했다. 독립 native CPU single-candidate LeWM 추론, 감사된 DINO native token 거리, 제어기·접촉 점수와 비용, 모든 선택·호출 수·시간 항목, dense 물체 오차·jerk를 검사했다. 제어기·접촉 경로에는 world model 호출이 없다. DINO는 검증한 third-party retrain이며 원래 공개 checkpoint의 성능으로 부르지 않는다.','','DINO는 native single-candidate FP32 추론을 사용했다. 별도 4상태 검사에서 native batch4·8은 single 추론 대비 token 최대 차이 1.15e−5 이내였지만 CPU 시간 이점이 관측되지 않아 이 측정에서는 사용하지 않았다. 그 검사는 기술 검증이며 일반적인 batch 속도 결론이 아니다.','','100ms–30s 예산별 적격성은 calibration의 경험적 위험·최소 적용 비율·측정 p95로만 정했다. 부적격이면 nominal fallback으로 집계하고 development 초과 시간 비율을 별도 보고했다. 실제 물리 시계에 계산 대기를 삽입한 실험이 아니므로 hard deadline 또는 지연 하 궤적 보존을 보증하지 않는다. 별도 새 family 최종 평가·반복 실행·적응은 남아 있다.','','근거: [고정 프로토콜](../artifacts/followup_v1/dinowm_runtime/protocol.json), [원시 감사](../artifacts/followup_v1/dinowm_runtime/raw_audit.json), [전체 시간](../artifacts/followup_v1/dinowm_runtime/timings.csv), [분석·예산별 비교](../artifacts/followup_v1/dinowm_runtime/analysis.json), [native 배치 검사](../artifacts/followup_v1/dinowm_batching_smoke/complete.json).','<!-- END DINO CPU RUNTIME -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN DINO COMMON VERIFIER -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['analysis.json','raw_audit.json','evaluation.csv','timings.csv']},'source_sha256':sha256(Path(__file__)),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()});print('REPORTED CPU RUNTIME',flush=True)

if __name__=='__main__':main()
