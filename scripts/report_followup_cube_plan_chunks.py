"""Publish only fully audited nominal causal compiler development results."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_plan_chunks';read=lambda p:json.loads(p.read_text())
def main():
    audit=read(OUT/'raw_audit.json');a=read(OUT/'analysis.json');complete=read(OUT/'pipeline_complete.json');assert not audit['smoke'] and audit['episodes']==36 and audit['families']==12 and audit['native_exact_episodes']==12
    assert complete['raw_audit_sha256']==sha256(OUT/'raw_audit.json') and audit['analysis_sha256']==sha256(OUT/'analysis.json') and audit['source_sha256']==sha256(resolve('scripts/analyze_followup_cube_plan_chunks.py'))
    lock=read(OUT/'pipeline_lock.json')
    for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
    f=pd.read_csv(OUT/'evaluation.csv');pred=pd.read_csv(OUT/'prediction_errors.csv');assert len(f)==36 and len(pred)==10800
    for row in a['summary']:
        g=f[f['mode']==row['mode']];assert len(g)==row['episodes']==12
        for k,v in row.items():
            if k not in ['mode','episodes']:np.testing.assert_allclose(g[k].mean(),v,rtol=1e-12,atol=1e-12)
    lines=['<!-- BEGIN CUBE CAUSAL CHUNKS -->','## Cube H=5 nominal 행동 구간: 미래 물리 분기 없이 생성·검증','',
    'Native 계획 추종 정책을 H=5 행동 구간으로 바꾸는 두 compiler를 같은 개발 scene 12개에서 평가했다. 알려진 EEF·gripper 목표 계획에서 상대 명령을 계산하되, 하나는 목표를 즉시 추종한다고 가정하고(`ideal_h5`), 다른 하나는 과거 로봇 움직임으로 적합한 유효 PD 및 gripper 응답으로 다음 상태를 예측한다(`past_pd_h5`). **모든 다섯 명령은 해당 구간을 실행하기 전에 고정하며 미래 simulator 분기나 미래 물체 상태를 입력받지 않는다.**','',
    '처음 15개 제어는 H=1이고 이후 5개씩 생성한다. 계획 자체는 초기 privileged 물체·목표 상태로 만든 native oracle 계획이다. 따라서 이미지 정책이나 물체 정보를 사용하지 않는 정책으로 해석하지 않는다. 모든 방식은 같은 15초·300개 제어·500Hz 물리 계측을 사용한다.','',
    '| Nominal 방식 | 종료 목표 성공 | 종료 성공＋접촉 없음 | 목표 위치 오차 평균, mm | EEF jerk energy 평균, m²/s⁵ | 원래 H=1 물체 궤적 보존 |','| --- | --- | --- | --- | --- | --- |']
    for mode in ['native_h1','ideal_h5','past_pd_h5']:
        g=f[f['mode']==mode];lines.append(f"| {mode} | {int(g.success_final.sum())}/12 | {int(g.released_success_final.sum())}/12 | {g.final_position_error_m.mean()*1000:.2f} | {g.eef_jerk_energy.mean():.3e} | {int(g.native_trajectory_preserved.sum())}/12 |")
    lines+=['','목표 성공은 환경의 위치 40mm 기준이다. 마지막 열은 같은 초기 상태에서 기존 H=1 실행 대비 **물체 위치 2mm·full 회전 2도 이내를 500Hz 전체 시각에서 유지했는지**다. Chunk 생성 자체가 nominal 움직임을 바꾸므로 목표 성공과 궤적 보존은 다를 수 있다. 여기서 방식 간 jerk 차이를 물체 궤적을 보존한 smoothing 효과로 세지 않는다.','',
    '| H=5 방식 | 1번째 행동 끝 EEF 위치 예측 오차 평균, mm | 5번째 행동 끝, mm | 5번째 행동 끝 yaw 오차 평균, deg |','| --- | --- | --- | --- |']
    for mode in ['ideal_h5','past_pd_h5']:
        one=pred[(pred['mode']==mode)&(pred.horizon==5)&(pred.offset==1)];five=pred[(pred['mode']==mode)&(pred.horizon==5)&(pred.offset==5)];lines.append(f'| {mode} | {one.position_prediction_error_m.mean()*1000:.2f} | {five.position_prediction_error_m.mean()*1000:.2f} | {np.rad2deg(five.yaw_prediction_error_rad.mean()):.2f} |')
    lines+=['','두 compiler의 예측 오차는 각각 자기 실행에서 측정하므로 동일 궤적에서의 순수 predictor 비교는 아니다. Gripper 예측 오차와 offset별 전체 결과도 원본에 저장했다. 이 오차는 신선한 dense kinematics와 비교하며, 환경 제어기가 읽는 적분 전 cached site 관측과 구분한다.','',
    f"원시 검증은 **36개 episode·{audit['blocks']:,}개 구간·{audit['physics_samples']:,}개 물리 표본**을 다뤘다. H=1의 12개 실행은 기존 검증된 native 결과와 action·물리 trace가 정확히 같다. 저장된 qpos에서 controller의 cached EEF와 gripper 관측을 재구성하고, 각 구간의 과거 입력·명령 목표·fit·계획 index를 확인했다. Compiler를 호출하지 않는 별도 수식 구현으로 전체 명령과 예측을 재계산했다.",'',
    '첫 smoke 검증기의 action 역정규화는 환경의 float32 계산 순서를 다르게 처리해 과거 target에서 약 9×10⁻⁹ 오차를 검출했다. 해당 검증 코드·실패 로그를 보존하고 환경의 연산 순서로 수정한 뒤 smoke 및 전체 감사를 수행했다. 실행 데이터나 기준 허용 오차를 바꾼 것은 아니다.','',
    '**완료 범위는 개발용 nominal compiler 검증이다.** 다음 smoothing 비교에서는 선택한 compiler의 수정 없는 실행을 기준으로 두고 같은 compiler에 correction을 추가한다. 기존 H=1 정책과의 품질 차이도 유지해 보고한다. 반복 smoothing·조건 적응·실제 지연·독립 최종 평가까지 완료한 결과가 아니며, 12개 개발 scene으로 정책의 일반적 성공률을 보장하지 않는다.','',
    '근거: [설정](../configs/followup_cube_plan_chunks.json), [고정 실행](../artifacts/followup_v1/cube_plan_chunks/pipeline_lock.json), [분석](../artifacts/followup_v1/cube_plan_chunks/analysis.json), [원시 감사](../artifacts/followup_v1/cube_plan_chunks/raw_audit.json), [episode 결과](../artifacts/followup_v1/cube_plan_chunks/evaluation.csv), [예측 오차](../artifacts/followup_v1/cube_plan_chunks/prediction_errors.csv).','<!-- END CUBE CAUSAL CHUNKS -->']
    section='\n'.join(lines)+'\n';doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();start='<!-- BEGIN CUBE CAUSAL CHUNKS -->';end='<!-- END CUBE CAUSAL CHUNKS -->'
    if start in s:
        first=s.index(start);last=s.index(end,first)+len(end);s=s[:first]+section.rstrip()+s[last:]
    else:
        where=s.index('<!-- BEGIN CUBE FINAL TEST -->');s=s[:where]+section+'\n'+s[where:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'episodes':36,'source_sha256':sha256(Path(__file__)),'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'analysis_sha256':sha256(OUT/'analysis.json'),'document':str(doc.relative_to(resolve('.'))),'section_sha256':__import__('hashlib').sha256(section.rstrip().encode()).hexdigest(),'unix':time.time()});print('REPORTED Cube causal H5 baseline',flush=True)
if __name__=='__main__':main()
