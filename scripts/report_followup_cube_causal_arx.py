"""Publish only independently audited causal ARX diagnostic results."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_causal_arx';read=lambda p:json.loads(p.read_text())
def main():
    audit=read(OUT/'raw_audit.json');analysis=read(OUT/'analysis.json');lock=read(OUT/'protocol.json')
    for p,h in {**lock['inputs'],**audit['inputs'],**audit['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    assert audit['source_sha256']==sha256(resolve('scripts/audit_followup_cube_causal_arx.py')) and audit['selections']==576
    pct=lambda x:'미정의' if x is None else f'{100*x:.2f}%'
    lines=['<!-- BEGIN CUBE CAUSAL ARX -->','## Cube: 과거 관측 기반 반응 모델을 바꾸면 jerk 선택이 개선되는가?','',
    '기존 Cube policy의 **12개 development family·48개 상태·899개 후보**를 그대로 사용해 과거 PD, 축별 affine ARX, 축간 결합 affine ARX를 비교했다. 3개 모델의 2,697개 후보 예측과 576개 선택을 독립 검증했다. 새로운 후보·물리 실행·calibration·최종 test를 추가한 결과는 아니다.','',
    '각 상태 직전 0.5초의 EEF 위치·증분과 이미 전달된 목표 위치만으로 모델을 맞췄다. 미래 robot/object trace는 fitting에 사용하지 않았다. 고정 ridge와 excitation·rank·discrete stability 조건을 사용하고, 조건을 만족하지 않으면 기존 과거 PD로 fallback했다. 축별 모델은 48/48개 상태의 모든 축에서, 결합 모델은 43/48개 상태에서 채택됐다. 이 채택 여부는 수치적 식별·안정성 검사이며, 미래 예측의 정확성을 보장하는 검사는 아니다.','',
    '아래는 상태별 진단 지표의 평균이다. 위치 오차가 줄어드는 것과 jerk 감소 후보의 순위를 올바르게 매기는 것은 구분해야 한다.','',
    '| 예상 실행 모델 | Nominal EEF 위치 RMSE, mm | 후보 효과 EEF 위치 RMSE, mm | 실제 jerk 감소량과의 순위 상관 | 감소량 절대 오차, %p |',
    '| --- | --- | --- | --- | --- |']
    for variant in ['past_pd','diagonal_arx','coupled_arx']:
        x=next(z for z in analysis['fidelity'] if z['variant']==variant);lines.append(f"| {variant} | {1000*x['nominal_position_RMSE_m']['mean']:.2f} | {1000*x['candidate_effect_position_RMSE_m']['mean']:.3f} | {x['gain_rank_spearman']['mean']:.3f} | {100*x['gain_mean_absolute_error']['mean']:.2f} |")
    lines+=['','후보 생성과 세 verifier의 점수·허용 규칙을 유지하고 예상 jerk 순위만 바꿨다. **어느 ARX 변형도 세 verifier의 동시 달성률 U를 높이지 못했다.** 축별 모델의 위치 예측 개선을 유용한 smoothing 개선으로 해석하지 않는다. 아래 R은 수정이 적용된 선택 중 시간 정렬 물체 궤적 위반율이다.','',
    '| 예상 실행 모델 | 승인 규칙 | U | 수정 적용 비율 | 선택 후 위반율 R | 평균 실제 jerk 감소 |','| --- | --- | --- | --- | --- | --- |']
    for variant in ['past_pd','diagonal_arx','coupled_arx']:
        for verifier in ['controller_displacement','proprio_contact','lewm_global','true_preservation']:
            x=next(z for z in analysis['summary'] if z['variant']==variant and z['verifier']==verifier and z['phase']=='all');name='실제 궤적 보존 oracle' if verifier=='true_preservation' else verifier;lines.append(f"| {variant} | {name} | {pct(x['joint']['mean'])} | {pct(x['active']['mean'])} | {pct(x['selected_risk']['mean'])} | {pct(x['jerk_gain']['mean'])} |")
    lines+=['',
    '**Oracle 행은 미래 물리 결과로 후보를 거른 진단이며 배포 가능한 방법이 아니다.** 실제 보존을 미리 알려줘도 축별 ARX의 U는 6.25%에서 2.08%로 떨어졌고 결합 ARX는 6.25%를 유지했다. 같은 bank에서 실제 보존과 실제 jerk로 모두 선택한 기존 유한 oracle의 U는 12.50%였다. 따라서 이 두 ARX 변형만으로 jerk 순위의 병목을 해결했다고 볼 수 없다.','',
    '주 보존 기준은 H5＋원래 suffix 3행동 동안 같은 물리 시각의 물체 중심 2mm·full SO(3) 회전 2도 이내다. U는 이에 실제 XYZ EEF jerk 10% 감소와 0이 아닌 correction을 함께 요구한다. Gripper action은 원래 bank와 동일하다. 후보별 물리 trace는 원래 감사를 재사용하며, 이번 감사에서 dense pose와 jerk를 다시 계산했다.','',
    '학습 방정식은 별도의 augmented least-squares로 검증했고 모든 모델의 재귀 예측, excitation·rank·stability·fallback, 선택과 family bootstrap 구간을 재계산했다. 12개 family의 탐색적 결과이며 개선이 없다는 점추정으로 모집단 동등성을 주장하지 않는다. 순위 상관·효과 오차의 전체 구간, 접근/파지/운반/해제별 결과는 원본 분석에 있다.','',
    '다음 판단은 후보 생성과 예측 순위를 분리해서 내려야 한다. 현재 후보는 기존 PD로 생성했으므로 새 모델로 후보를 다시 만들었을 때의 가능성까지 부정하지 않는다. 다만 현재 검증에서는 새 모델을 기본 구성으로 교체할 근거가 없다. 추가 후보 생성 또는 접촉 상태별 반응 모델을 개발한다면 별도 calibration과 새 family 평가를 거쳐야 한다.','',
    '근거: [진단 사전 설정](../artifacts/followup_v1/cube_causal_arx/protocol.json), [전체 분석·paired 구간](../artifacts/followup_v1/cube_causal_arx/analysis.json), [독립 감사](../artifacts/followup_v1/cube_causal_arx/raw_audit.json), [개별 선택](../artifacts/followup_v1/cube_causal_arx/selected.csv).','<!-- END CUBE CAUSAL ARX -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:
        anchor='<!-- BEGIN CUBE POLICY ORACLE -->'
        if anchor not in s:anchor='<!-- BEGIN POLICY LATENCY FINAL TEST -->'
        i=s.index(anchor);s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['analysis.json','raw_audit.json','protocol.json','selected.csv','fidelity.csv']},'source_sha256':sha256(Path(__file__)),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()});write_json(OUT/'pipeline_complete.json',{'states':48,'families':12,'candidates':899,'selections':576,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'scope':lock['scope'],'unix':time.time()});print('REPORTED CAUSAL ARX',flush=True)
if __name__=='__main__':main()
