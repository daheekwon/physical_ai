"""Publish complete independently audited replacement chains by model family."""
import argparse,json,hashlib,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',required=True,choices=['lewm','dino']);args=ap.parse_args();out=BASE/('output_replacement_'+args.family);done=read(out/'pipeline_complete.json');audit=read(out/'raw_audit.json');analysis=read(out/'analysis.json');cfg=read(out/'protocol.json')['config'];assert done['states']==audit['states']==402 and audit['selected_rows']==17820 and not audit['smoke']
    assert done['raw_audit_sha256']==sha256(out/'raw_audit.json') and audit['analysis_sha256']==sha256(out/'analysis.json') and audit['rules_sha256']==sha256(out/'rules.json');assert audit['source_sha256']==sha256(resolve('scripts/audit_followup_output_replacement.py'))
    for p,h in read(out/'pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p
    selected=pd.read_csv(out/'selected.csv');assert len(selected)==17820
    summaries=[r for r in analysis['summary'] if r['split']=='development' and r['phase']=='all'];lookup={(r['environment'],r['representation'],r['target'],r['mode'],r['level']):r for r in summaries};models=sorted(set(k[:3] for k in lookup));assert len(models)==6
    for r in summaries:
        f=selected
        for k in ['environment','representation','target','mode','level','split']:f=f[f[k]==r[k]]
        assert len(f)==r['states']
        for key,col in [('joint','joint'),('active','active'),('jerk_gain','jerk_gain'),('candidate_coverage','candidate_coverage')]:np.testing.assert_allclose(f[col].mean(),r[key]['mean'],atol=1e-12)
        active=f[f.active]
        if len(active):np.testing.assert_allclose(active.violation.mean(),r['selected_risk']['mean'],atol=1e-12)
        else:assert r['selected_risk']['mean'] is None
    label='LeWM' if args.family=='lewm' else '재학습 DINO-WM';marker='OUTPUT REPLACEMENT '+args.family.upper()
    lines=['<!-- BEGIN '+marker+' -->',f'## {label}: 예측을 실제 결과로 바꾸는 선택 진단','',
    f'계획서 4.6절·F5의 치환 진단을 **PushT 210개 상태와 Cube 192개 상태, 총 402개 상태**에서 완료했다. 모델별 6개 고정 ridge readout 설정으로 {audit["candidate_readout_rows"]:,}개 후보–readout 행과 17,820개 상태별 선택을 검증했다. 기존 개발·calibration의 동일 물리 후보를 재사용했으며 새로운 물리 실행이나 최종 test는 아니다. 아래는 개발 PushT 105개 상태·35개 family 및 Cube 96개 상태·24개 family의 결과다.','',
    '비교 사슬은 **예측 latent → 실제 endpoint 영상의 latent → 실제 endpoint pose → 전체 구간 pose → 전체 보존 조건**이다. 처음 두 단계는 같은 frozen readout을 사용한다. 이후 단계는 실제 simulator 값을 사용하며, 모든 단계에서 후보와 후보별 수정 크기, 예상 제어기 jerk 순위는 같다. 선택되는 후보와 실제 적용 크기는 수준에 따라 달라질 수 있다. 실제 영상·pose를 사용하는 단계는 미래 정보를 아는 사후 진단이므로 배포 가능한 선택기 성능이 아니다.','',
    '물체 궤적 보존은 같은 시각 위치·회전의 최대 편차와 PushT coverage 조건이다. PushT는 2px·2도·coverage 감소 0.02, 100Hz의 5개 수정 행동＋3개 원래 후속 행동을 평가했다. Cube는 2mm·full SO(3) 2도, 500Hz의 같은 행동 개수이며 gripper는 원래대로 고정했다. Cube의 symmetry readout도 물리 정답을 대칭 회전으로 완화하지 않았다.','',
    '아래 셀은 **U / R / A (%)**다. U는 궤적 보존과 실제 jerk 10% 감소의 동시 달성률, R은 선택한 활성 수정 중 물체 위반율, A는 전체 상태 중 수정 적용 비율이다. R의 `—`는 적용이 없어 조건부 위반율이 정의되지 않는 경우다. 후보 전체 승인율 C와 조건별 위험·실제 jerk 감소·family 구간은 원본 분석에 별도로 제공한다.']
    if args.family=='lewm':
        fr=lookup[('pusht','lewm_frame','raw','separately_calibrated','predicted')];rr=lookup[('pusht','lewm_frame','raw','separately_calibrated','real_encoded')];ep=lookup[('pusht','lewm_frame','raw','unit_budget','true_endpoint')];dense=lookup[('pusht','lewm_frame','raw','unit_budget','dense_full')];ce=lookup[('ogb-cube','lewm_frame','raw','unit_budget','true_endpoint')];cd=lookup[('ogb-cube','lewm_frame','raw','unit_budget','dense_full')]
        lines+=['', '**현재 확인한 결론.** 실제 encoded endpoint로 교체하는 것만으로는 유용한 선택이 일관되게 회복되지 않았다. 이는 predictor가 정확하다는 증거가 아니라, 고정 readout과 endpoint 판정까지 포함한 경로를 함께 개선해야 하는지 검토할 근거다.', '',
        f"- PushT frame readout의 별도 보정 후 U는 예측과 실제 encoded endpoint 모두 {100*fr['joint']['mean']:.2f}%였다. History readout도 두 수준의 U가 같았다. 동등성의 증거가 아니며 paired 구간은 아래에 제시한다.",
        f"- 실제 endpoint pose를 정확히 알아도 τ=1에서 선택한 수정의 궤적 위반율은 PushT {100*ep['selected_risk']['mean']:.2f}%, Cube {100*ce['selected_risk']['mean']:.2f}%였다. 끝점 보존은 중간 및 후속 궤적 보존을 대신하지 못한다.",
        f"- 실제 전체 보존 판정으로 필터링하고 같은 예상 jerk 순위를 쓰면 U는 PushT {100*dense['joint']['mean']:.2f}%, Cube {100*cd['joint']['mean']:.2f}%이고 이 유한 bank의 선택 후 위반은 0이었다. 미래 물리 trace를 사용하는 진단 결과이므로 현재 LeWM의 실행 전 성능으로 주장하지 않는다.",
        '- Calibration의 위험 목표 10%를 충족한 규칙도 일부 개발 조건에서 그 목표를 넘었다. 따라서 표의 U와 R을 함께 보고, 미래 정보 치환이나 재보정으로 얻은 U 증가만으로 배포 가능성을 판단하지 않는다.']
    mode_names={'unit_budget':'1. 물리 단위 정규화 점수의 기준값 τ=1 고정','predicted_rule_transfer':'2. 예측 점수로 보정한 규칙을 모든 수준에 그대로 전이','separately_calibrated':'3. 각 수준을 같은 탐색 예산·위험 목표로 별도 보정'}
    def value(r):
        risk=r['selected_risk']['mean'];rr='—' if risk is None else f'{risk*100:.1f}'
        return f"{r['joint']['mean']*100:.1f} / {rr} / {r['active']['mean']*100:.1f}"
    for mode in cfg['modes']:
        lines+=['','**'+mode_names[mode]+'**','','| 환경·표현·target | 예측 latent | 실제 encoded latent | 실제 endpoint | Dense pose | 전체 보존 |','| --- | --- | --- | --- | --- | --- |']
        for env,rep,kind in models:
            lines.append(f"| {env} / {rep} / {kind} | "+' | '.join(value(lookup[(env,rep,kind,mode,level)]) for level in cfg['levels'])+' |')
    lines+=['',
    '별도 보정은 calibration 상태에서 경험적 선택 후 위반율 10% 이하와 적용 비율 25% 이상을 요구하고, 예상 jerk 10% 개선 상태 비율·평균 예상 개선량 순으로 threshold를 고른다. PushT는 접촉 상태의 선택 후 위반율에도 같은 목표를 적용한다. 각 수준에 같은 41개 quantile 지점과 τ=0·1을 요청하고 중복 threshold는 제거한다. 조건을 만족하는 규칙이 없으면 disabled로 남긴다. 예측 규칙이 disabled이면 전이 비교 전체도 fallback이며, 이를 실제 결과 치환이 무효라는 증거로 해석하지 않는다.','',
    '보정된 τ는 1을 넘을 수 있으므로 미래 값을 쓰는 조건도 일부 위반을 승인할 수 있다. 반면 τ=1의 전체 보존 필터는 기록된 유한 후보의 실제 보존 판정을 그대로 적용한다. 여기서도 순위는 예상 jerk를 사용하므로 실제 jerk까지 아는 유한 후보 oracle과 다르며 전역 최적 성능도 아니다.','',
    '| 별도 보정: 실제 encoded latent − 예측 latent | U 차이, %p [family 95% 구간] |','| --- | --- |']
    for env,rep,kind in models:
        r=next(r for r in analysis['paired_adjacent_levels'] if (r['environment'],r['representation'],r['target'])==(env,rep,kind) and r['mode']=='separately_calibrated' and r['split']=='development' and r['from']=='predicted' and r['to']=='real_encoded' and r['metric']=='joint');lo,hi=r['ci95'];lines.append(f"| {env} / {rep} / {kind} | {r['mean']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}] |")
    lines+=['',
    '해석 범위: 예측→실제 표현의 변화는 해당 고정 readout에서 예측을 실제 표현으로 바꿀 때 선택이 회복되는지, 실제 표현→실제 pose는 readout 측정 경로, endpoint→dense pose는 중간·후속 궤적의 누락, dense pose→전체 조건은 coverage 누락을 진단한다. 서로 다른 수준의 점수와 보정이 선택에 비선형으로 작용하므로 차이를 독립적인 오차 기여도처럼 더하지 않는다. Readout 용량·alpha를 고를 때 이미 사용한 개발 family이므로 독립 최종 증거가 아니며, 새 selector를 채택하면 새 calibration·test가 필요하다.','',
    '비용과 정보 범위: 공통 bank에는 LeWM 방향이 포함돼 있다. LeWM을 호출하지 않는 verifier라고 해서 전체 후보 생성도 LeWM-free인 것은 아니다. Cube의 공통 예상 비용은 병진과 yaw를 포함한 과거 관측 기반 surrogate이며, 주 실제 jerk 평가는 EEF 병진이다. 이 목적함수 차이와 예측 오차는 순위 실패를 만들 수 있으므로 치환 결과를 dynamics 오차만으로 설명하지 않는다. DINO-WM은 별도로 재학습된 호환 checkpoint이며 원 논문의 원본 weight 재현으로 부르지 않는다.','',
    f'독립 감사는 {audit["decoded_arrays"]:,}개 decoded 배열, 실제 효과와 dense pose, PushT 원시 pose의 polygon coverage, 실제 jerk, 예상 비용, {audit["calibration_trials"]:,}개 calibration 후보와 최종 선택, family 단위 구간을 재계산했다. 미래 정보는 사후 진단 수준에만 제공했고 새 학습·정책 변경은 수행하지 않았다.','',
    f'근거: [고정 설정](../configs/followup_output_replacement.json), [실행 잠금](../artifacts/followup_v1/output_replacement_{args.family}/pipeline_lock.json), [분석](../artifacts/followup_v1/output_replacement_{args.family}/analysis.json), [보정 규칙](../artifacts/followup_v1/output_replacement_{args.family}/rules.json), [상태별 선택](../artifacts/followup_v1/output_replacement_{args.family}/selected.csv), [원시 감사](../artifacts/followup_v1/output_replacement_{args.family}/raw_audit.json).','<!-- END '+marker+' -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();start=lines[0];end=lines[-1]
    if start in s:i=s.index(start);j=s.index(end,i)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN GRADIENT CONTROLS -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(out/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'raw_audit_sha256':sha256(out/'raw_audit.json'),'analysis_sha256':sha256(out/'analysis.json'),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()});print('REPORTED',args.family,flush=True)
if __name__=='__main__':main()
