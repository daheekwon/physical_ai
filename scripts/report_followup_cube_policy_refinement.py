"""Report validated Cube repeated-refinement scope without extrapolating smoke."""
import argparse,json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args();out=BASE/('cube_policy_refinement_smoke' if args.smoke else 'cube_policy_refinement');a=read(out/'raw_audit.json');analysis=read(out/'analysis.json');assert a['smoke']==args.smoke and a['analysis_sha256']==sha256(out/'analysis.json') and a['source_sha256']==sha256(resolve('scripts/audit_followup_cube_policy_refinement.py'))
    if not args.smoke:
        done=read(out/'pipeline_complete.json');assert done['raw_audit_sha256']==sha256(out/'raw_audit.json') and a['episodes']==48 and a['families']==12 and a['local_blocks']==2736
        for p,h in read(out/'pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p
    else:assert a['episodes']==4 and a['local_blocks']==228 and a['optimizer_attempts']==855
    frame=pd.read_csv(out/'evaluation.csv');local=pd.read_csv(out/'local_evaluation.csv');assert len(frame)==a['episodes'] and len(local)==a['local_blocks'];suffix=' SMOKE' if args.smoke else '';marker='CUBE POLICY REFINEMENT'+suffix
    lines=['<!-- BEGIN '+marker+' -->','## Cube 반복 smoothing: '+('전체 길이 실행 검증' if args.smoke else '12개 초기 상태의 개발 비교'),'']
    if args.smoke:
        lines+=['한 초기 상태에서 nominal·controller·접촉 기반·LeWM verifier의 **4개 15초 episode와 228개 국소 비교**를 검증했다. 방법 간 성능 결론을 내리기 위한 표본이 아니라, 검증된 H5 명목 실행에 반복 smoothing을 연결한 실행·감사 검사다. 전체 비교는 별도의 12개 초기 상태·48개 episode 프로토콜로 진행한다.','',
        '수정 없는 실행은 기존 `past_pd_h5`의 행동·물리 상태를 정확히 재현했다. 최초로 통과한 소규모 실행은 별도 보존했고, optimizer 종료 상태·반복 횟수 기록을 보강한 버전이 네 방법 모두 같은 행동과 물리 궤적을 재현하는 것도 확인했다.']
    else:
        lines+=['동일한 12개 초기 상태에서 **4방법·48개 15초 episode, 2,736개 국소 H5＋3 비교**를 완료했다. 기준은 각 초기 상태의 수정 없는 `past_pd_h5` 실행이며, native H1 궤적을 새 명목 기준과 혼동하지 않는다. 기존 expert non-WM 후보 bank에서 보정한 verifier 규칙을 그대로 옮긴 개발 진단이다. 새 policy calibration이나 독립 최종 test는 아니다.','',
        '| 방법 | 전체 U, % | 활성 episode 중 누적 위반율, % | 평균 전체 jerk 감소, % | 최종 목표 성공, % | 국소 U, % | 활성 국소 수정 중 위반율, % |','| --- | --- | --- | --- | --- | --- | --- |']
        labels={'nominal':'명목 실행','controller_displacement':'제어기','proprio_contact':'접촉 proxy','lewm_global':'LeWM verifier'}
        for r in analysis['summary']:
            f=frame[frame.method==r['method']]
            for metric in ['joint','jerk_gain','task_success_final']:np.testing.assert_allclose(f[metric].mean(),r[metric]['mean'],atol=1e-12)
            percentage=lambda x:'—' if x['mean'] is None else f"{100*x['mean']:.2f}"
            lines.append(f"| {labels[r['method']]} | {percentage(r['joint'])} | {percentage(r['global_selected_risk'])} | {percentage(r['jerk_gain'])} | {percentage(r['task_success_final'])} | {percentage(r['local_joint'])} | {percentage(r['local_selected_risk'])} |")
        lines+=['','U는 원래 명목 실행의 시간 정렬된 물체 위치·full SO(3) 회전 궤적을 2mm·2도 안에서 유지하고 실제 EEF 병진 jerk를 10% 이상 줄인 비율이다. 누적 기준은 episode 시작에 고정한다. 목표 성공은 최종 위치가 목표에서 40mm 이내인 별도 지표이며 궤적 보존과 같지 않다. 각 방법의 자체 관측으로 다음 명목 chunk를 다시 계산하므로 누적 차이는 correction과 이후 compiler 반응을 함께 포함한다.','',
        '| 제어기 대비 | 전체 U 차이, %p [family 95% 구간] | 전체 jerk 감소율 차이, %p [family 95% 구간] |','| --- | --- | --- |']
        for method in ['proprio_contact','lewm_global']:
            cells=[]
            for metric in ['joint','jerk_gain']:
                r=next(r for r in analysis['paired_contrasts'] if r['method']==method and r['metric']==metric);lo,hi=r['ci95'];cells.append(f"{100*r['mean']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
            lines.append(f"| {labels[method]} | {' | '.join(cells)} |")
        lines+=['','독립 family는 12개이며 국소 chunk 수를 독립 표본 수로 세지 않는다. 관측 위험 0이나 폭 0의 bootstrap 구간은 모집단 안전 또는 동등성의 보증이 아니다. 국소 보존을 누적 보존의 대체 지표로 사용하지 않는다.']
    lines+=['','설정은 다음과 같다. 첫 15개 제어는 명목 실행이며 이후 5개 행동 단위로 현재까지의 robot 관측·RGB history를 사용한다. 알려진 절대 EEF/gripper 계획으로 다음 행동을 생성하고, non-WM 다항식 후보 중 선택한 correction만 더한다. 세 verifier의 후보 생성식·radii·예상 jerk 순위는 같으며 LeWM은 verifier일 때만 추론한다. 각 방법이 도달한 상태가 달라진 이후에는 후보의 실제 값도 달라질 수 있다.','',
    'Gripper action 좌표는 수정하지 않고 알려진 계획의 gripper 목표 시각도 유지한다. 실제로 적용할 float32 명령으로 점수와 norm을 다시 계산하며, 다항식 소속은 사전 지정한 양자화 오차 안에서 확인한다. 예상 비용은 실제 주 지표와 맞춘 EEF 병진 jerk다. 기존 expert 후보의 XYZ/yaw 결합 비용과 행동 분포가 달라졌으므로 과거의 경험적 위험 수준이 유지된다고 가정하지 않는다.','',
    '국소 평가는 선택을 먼저 확정한 뒤 같은 현재 상태에서 nominal H5＋원래 suffix 3개와 수정 H5＋같은 suffix를 분기 실행한다. 이후 원래 현재 상태로 복원해 선택한 H5만 실제로 실행한다. 해당 실제 구간이 평가 분기의 첫 H5와 정확히 일치하는지 검사하며, 미래 분기 결과를 다음 관측 history나 selector 입력에 넣지 않는다.','',
    f'감사 범위는 명목 재현 {a["exact_nominal_episodes"]}개 episode, 기록된 실제 시간·상태, 현재/과거 입력과 독립 compiler, 실제 적용 후보의 bounds·다항식·gripper·예상 비용, {a["fresh_LeWM_candidate_predictions"]:,}개 LeWM 후보의 fresh 예측, 선택 index, 국소/누적 보존 및 family 통계다. Optimizer {a["optimizer_attempts"]:,}회 중 미수렴 {a["optimizer_failures"]:,}회와 그중 유효 후보로 남은 {a["unconverged_candidates_retained"]:,}회를 모두 기록했다. 미수렴이더라도 제약·개선 검사를 통과한 후보를 남기는 기존 생성 규칙을 유지하며 전역 최적해라고 부르지 않는다.','',
    '계산 시간은 측정하지만 이번 실험의 물리 시계에는 삽입하지 않았다. 초기 plan은 native oracle의 privileged 상태를 사용하므로 pixel-only policy로 해석하지 않는다. 정책용 재보정·새 최종 평가·조건 적응·실제 지연은 별도 후속 단계로 남는다.','',
    f'근거: [고정 설정](../configs/followup_cube_policy_refinement.json), [실행 manifest](../artifacts/followup_v1/{out.name}/manifest.json), [원시 감사](../artifacts/followup_v1/{out.name}/raw_audit.json), [국소 결과](../artifacts/followup_v1/{out.name}/local_evaluation.csv), [optimizer 기록](../artifacts/followup_v1/{out.name}/solver_diagnostics.csv), [분석](../artifacts/followup_v1/{out.name}/analysis.json).','<!-- END '+marker+' -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();start=lines[0];end=lines[-1]
    if start in s:i=s.index(start);j=s.index(end,i)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN CUBE CAUSAL CHUNKS -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(out/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'raw_audit_sha256':sha256(out/'raw_audit.json'),'analysis_sha256':sha256(out/'analysis.json'),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()});print('REPORTED cube policy refinement',args.smoke,flush=True)
if __name__=='__main__':main()
