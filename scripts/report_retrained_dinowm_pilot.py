"""Report fully audited actual-effect and same-candidate ranking pilot."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from zj_wam.io import resolve,write_json,sha256
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_retrained_effect_pilot';MATCH=BASE/'dinowm_retrained_matched_pilot';read=lambda p:json.loads(p.read_text())


def fmt(value):return '미정의' if value is None else f'{value:.2f}'


def main_report():
    a=read(OUT/'analysis.json');raw=read(OUT/'raw_audit.json');b=read(MATCH/'analysis.json');audit=read(MATCH/'raw_audit.json');assert raw['states']==21 and audit['state_method_rows']==84
    for folder in [OUT,MATCH]:
        m=read(folder/'manifest.json');assert read(folder/'raw_audit.json')['manifest_sha256']==sha256(folder/'manifest.json')
        for p,h in m['inputs'].items():assert sha256(resolve(p))==h,p
    for name,h in audit['parent_inputs'].items():assert sha256(OUT/name)==h,name
    states=pd.read_csv(OUT/'state_metrics.csv');matched=pd.read_csv(MATCH/'state_metrics.csv');rows=[];rankrows=[];contrasts=[]
    for r in a['summary']:
        env=r['environment'];g=states[states.environment==env]
        keys=['predicted_vs_actual_change_rho','actual_vs_endpoint_object_rho','predicted_vs_endpoint_object_rho','mean_effect_cosine']
        cells=[fmt(r[k]['mean'])+f' ({g[k].notna().sum()}/{len(g)})' for k in keys]
        rows.append(f"| {env} | {r['states']} / {r['families']} | {r['candidate_rows']} | "+' | '.join(cells)+' |')
    for r in b['summary']:
        g=matched[(matched.environment==r['environment'])&(matched.method==r['method'])]
        rankrows.append(f"| {r['environment']} | {r['method']} | {fmt(r['endpoint_rho']['mean'])} | {fmt(r['trajectory_rho']['mean'])} | {fmt(r['violation_auc']['mean'])} ({g.violation_auc.notna().sum()}/{len(g)}) |")
    for r in b['paired']:
        if r['baseline']!='controller':continue
        metric=r['trajectory_rho'];lo,hi=metric['ci95'] if metric['ci95'] is not None else (None,None)
        contrasts.append(f"{r['environment']}: {fmt(metric['mean'])}, family 95% 구간 [{fmt(lo)}, {fmt(hi)}]")
    block=f'''<!-- BEGIN DINOWM EFFECT PILOT -->
## DINO-WM 실제 intervention pilot: 표현과 예측을 나눠 평가

재학습 DINO-WM을 **PushT 개발용 3개 family·9개 상태, Cube 3개 family·12개 상태**에 적용했다. 모델 결과를 보기 전에 고른 기존 후보 전체의 예측·실제 영상 latent를 저장하고, 원시 물리 trace에서 기존 물체 오차를 재계산했다. Nominal을 포함해 900개 후보의 latent를 저장했고, 원래 CSV 형식에 따라 분석한 행은 {raw['candidate_rows']:,}개다(PushT의 nominal 9개는 분석 행에서 제외). PushT는 correction family bank, Cube는 기존 공통 bank를 사용했으므로 task 간 후보 구성은 다르다. 새로운 물리 실행이나 최종 test가 아니다.

**변화량의 크기 순서를 읽는 능력과 변화 벡터의 정확성은 달랐다.** 예측·실제 영상 변화량의 순위 상관은 PushT 약 0.87, Cube 약 0.96이었지만, 변화 벡터 cosine은 약 0.29와 0.18이었다. 8행동 물체 보존 오차의 순위 상관은 DINO-WM이 각각 약 0.90·0.72, 제어기가 약 0.94·0.84였다. 이 작은 표본에서는 전체 patch 변화량을 DINO-WM으로 바꾸는 것만으로 제어기보다 나은 보존 판단이 확인되지 않았다.

첫 표는 상태별 Spearman 상관의 평균과 intervention 벡터 cosine이다. 괄호는 해당 지표가 정의되는 상태 수/전체 상태 수다. 물체가 움직이지 않거나 후보 점수가 모두 같아 상관을 정의할 수 없는 상태는 0으로 바꾸지 않았다. 실제 영상과 예측은 각 후보의 **nominal 대비 변화**로 비교했다.

| Task | 상태 / family | 후보 행 | 예측 변화량 ↔ 실제 영상 변화량 | 실제 영상 변화량 ↔ 5행동 끝 물체 오차 | 예측 변화량 ↔ 5행동 끝 물체 오차 | 예측·실제 변화 벡터 cosine |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

물체 오차는 5행동 끝에서 nominal 대비 위치·회전 차이를 기존 허용 오차로 나눈 값의 최대치다. 아래의 전체 보존 오차는 8행동 동안의 같은 시각 위치·회전 최대 편차이며, PushT는 기존 coverage 조건도 포함한다. 영상·latent 비교 시점과 긴 물리 평가 구간을 섞어 하나의 예측 정확도로 부르지 않는다.

같은 후보에서 네 점수의 순서를 비교했다. DINO-WM과 LeWM은 예측 latent 변화량, 제어기는 기존 실행 반응식의 robot 변위 점수, action norm은 수정 크기다. 상관과 ROC-AUC는 후보 크기 순서를 얼마나 맞추는지 보는 **진단 지표**이며, calibration한 승인 규칙의 U/R이 아니다.

| Task | 점수 | 5행동 끝 물체 오차 상관 | 8행동 보존 오차 상관 | 8행동 위반 ROC-AUC (정의된 상태 수) |
| --- | --- | --- | --- | --- |
{chr(10).join(rankrows)}

DINO-WM−controller의 8행동 보존 오차 상관 차이는 {'; '.join(contrasts)}였다. Family 수가 task마다 3개뿐이고 동일 상태의 여러 후보가 강하게 관련되므로 이 수치로 모델 우위나 일반화된 보존 성능을 확정하지 않는다. 위반/비위반이 함께 있는 상태에서만 AUC가 정의되며, 적용 범위와 승인 후 실패율을 대신하지 않는다.

전체 patch latent에는 robot·배경도 들어 있다. 실제 영상 latent와 물체 오차의 상관이 높더라도 물체만 구분했다는 증거는 아니다. 실제 robot 변화와의 상관도 원본 분석에 제공한다. 후보 bank에는 LeWM 기반 방향이 포함되며, encoder·학습 조건·표현 차원·추론 precision도 두 모델에서 다르다. 이번 DINO-WM은 CPU FP32이며 원래 LeWM baseline 가중치가 아닌 별도 재학습 자산이다.

**다음 판단은 같은 후보·같은 실제 jerk 목적함수에서 calibration한 승인 규칙이 허용 오차를 지키면서 유용한 수정을 더 허용하는지**다. 물체 readout을 사용하는 변형에는 별도 지도학습과 데이터 비용을 명시해야 한다. 이 pilot은 그 전체 비교의 전제 진단이며, 아직 배포 가능한 DINO-WM 선택기나 보존 보장을 만들지 않았다.

근거: [실제 효과 분석](../artifacts/followup_v1/dinowm_retrained_effect_pilot/analysis.json), [원시 감사](../artifacts/followup_v1/dinowm_retrained_effect_pilot/raw_audit.json), [같은 후보의 기준선 비교](../artifacts/followup_v1/dinowm_retrained_matched_pilot/analysis.json), [점수 재현 감사](../artifacts/followup_v1/dinowm_retrained_matched_pilot/raw_audit.json).
<!-- END DINOWM EFFECT PILOT -->
'''
    p=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=p.read_text();begin='<!-- BEGIN DINOWM EFFECT PILOT -->';end='<!-- END DINOWM EFFECT PILOT -->'
    if begin in s:s=s[:s.index(begin)]+block+s[s.index(end)+len(end):]
    else:
        anchor='<!-- END DINOWM CPU PREFLIGHT -->';assert anchor in s;s=s.replace(anchor,anchor+'\n\n'+block,1)
    p.write_text(s);write_json(OUT/'report_evidence.json',{'inputs':{str((folder/name).relative_to(resolve('.'))):sha256(folder/name) for folder in [OUT,MATCH] for name in ['manifest.json','analysis.json','raw_audit.json','state_metrics.csv']},'source_sha256':sha256(Path(__file__)),'scope':'Exploratory3family/task actualeffect pilot; no U/R or superiorityclaim.'})
    print('REPORTED full DINO effect pilot',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-pid',type=int);args=p.parse_args();start=time.monotonic()
    if args.dependency_pid:
        while not (MATCH/'raw_audit.json').exists():
            try:os.kill(args.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError('Matched analysis ended before audit; inspect failure')
            if time.monotonic()-start>3600:raise RuntimeError('Observation window expired; inspect same process')
            time.sleep(5)
    main_report()
if __name__=='__main__':main()
