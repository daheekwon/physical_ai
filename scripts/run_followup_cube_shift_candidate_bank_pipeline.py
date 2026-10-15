"""Freeze audited shift-bank smoke and transfer nominal-only calibration rules."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_shift_candidate_bank';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_shift_candidate_bank.json','configs/followup_cube_policy_selection.json','scripts/cube_shift_candidate_bank_core.py','scripts/run_followup_cube_shift_candidate_bank.py','scripts/audit_followup_cube_shift_candidate_bank.py','scripts/analyze_followup_cube_shift_candidate_bank.py','scripts/run_followup_cube_shift_candidate_bank_pipeline.py','scripts/output_replacement_core.py','scripts/audit_followup_output_replacement.py','scripts/analyze_followup_response_selection.py','scripts/audit_retrained_dinowm_readouts.py']
def verify(record):
    for p,h in record.items():assert sha256(resolve(p))==h,p

def report():
    a=read(OUT/'raw_audit.json');s=read(OUT/'selection_audit.json');r=read(OUT/'analysis.json');assert a['contexts']==s['contexts']==336 and s['selections']==1008 and s['analysis_sha256']==sha256(OUT/'analysis.json');verify(read(OUT/'manifest.json')['inputs']);verify(a['inputs']);verify(s['inputs']);verify(s['helpers']);assert sha256(resolve(r['rule_source']))==r['rule_sha256'];pct=lambda x:'미정의' if x is None else f'{100*x:.2f}%'
    lines=['<!-- BEGIN CUBE SHIFT CANDIDATE TRANSFER -->','## Cube 물성 변화 후 고정 승인 규칙의 단발 correction','',f"기존12개 development family의48개 phase 상태에서 물성을10개 행동 전에 변경한7개 조건, 총336개 이력·{a['candidates']}개 다항식 후보를 실행했다. 변경 후 현재 robot 관측과 과거 robot/gripper 응답으로 H5＋3 nominal 명령을 먼저 계산하고, 모든 verifier가 동일한 다항식 후보와 예상 XYZ jerk 순위를 공유한다. Gripper 명령은 수정하지 않는다.",'', '허용 threshold는 새24개 nominal calibration family에서 정한 규칙을 그대로 가져왔다. 조건별 threshold·radius 재튜닝을 하지 않았다. 조건 변화 이력이나 미래 물리 결과는 규칙 보정에 사용하지 않았으며, 각 후보 행동과 점수는 실제 물리 분기 실행 전에 저장했다. 다음 값은 기존12개 development family에서의 결과이며 독립 최종 평가가 아니다.','', '| 조건 | 방법 | 동시 달성 U | 수정 적용 | 선택 후 위반율 | 실제 jerk 감소 평균 |','| --- | --- | --- | --- | --- | --- |']
    for row in r['summary']:lines.append(f"| {row['condition']} | {row['method']} | {pct(row['joint']['mean'])} | {pct(row['active']['mean'])} | {pct(row['selected_risk']['mean'])} | {pct(row['jerk_gain']['mean'])} |")
    lines+=['','U는 해당 조건의 같은 root에서 correction 없이 실행한 nominal과 비교해, H5＋3의 모든2ms 표본에서 물체 위치2mm·전체 회전2도 오차를 지키며 robot EEF XYZ jerk를10% 이상 줄인 상태의 비율이다. 최종 물체 위치만 같아지는 것은 보존으로 인정하지 않는다. 조건마다 현재 관측으로 다시 계산한 nominal을 기준으로 하므로, 서로 다른 조건의 물체 궤적 자체가 같다는 뜻은 아니다.','', '물성 변화의 대조는 같은 forward 처리를 수행한 sham_forward다. Original nominal(no-forward)도 재현 대조로 남겼다. 명목 반응을 바꾸지 않았던 조건도 제외하지 않았으며, 변화가 없는 결과를 적응 성공으로 세지 않는다. 조건별·verifier별 paired family 구간은 원본 분석에 포함한다.','',f"모든 후보의 geometry·예상/실제 jerk·물체 궤적 오차와 {a['fresh_LeWM_predictions']}개 LeWM 점수를 검증했다. 전체336개 root의 과거 물리·이미지를 재현하고 {a['fresh_physical_branches']}개 후보 분기를 새로 실행했다. 1,008개 선택과 모든 family 통계 구간은 독립 구현과 대조했다.",'', '**이 결과는 고정 threshold의 조건 전이 평가다.** 모든 방법이 갱신한 robot/gripper compiler를 공유한다. 고정/allowance만/계산량만/둘 다 적응하는 네 조건, 반복 episode 보존과 지연을 포함한 실행은 별도 검증이 필요하다. 후보 bank의 사후 oracle 가능성을 실제 선택기의 성능으로 대체하지 않는다.','', '근거: [실행 프로토콜](../artifacts/followup_v1/cube_shift_candidate_bank/manifest.json), [원시 감사](../artifacts/followup_v1/cube_shift_candidate_bank/raw_audit.json), [조건별 결과·paired 구간](../artifacts/followup_v1/cube_shift_candidate_bank/analysis.json), [선택 감사](../artifacts/followup_v1/cube_shift_candidate_bank/selection_audit.json).','<!-- END CUBE SHIFT CANDIDATE TRANSFER -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=doc.read_text()
    if lines[0] in text:i=text.index(lines[0]);j=text.index(lines[-1])+len(lines[-1]);text=text[:i]+section+text[j:]
    else:i=text.index('<!-- BEGIN CUBE SHIFT CAUSAL COMMANDS -->');text=text[:i]+section+'\n\n'+text[i:]
    doc.write_text(text);write_json(OUT/'report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['analysis.json','raw_audit.json','selection_audit.json','selected.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()})

def main(calibration_pid):
    smoke=BASE/'cube_shift_candidate_bank_smoke';a=read(smoke/'raw_audit.json');s=read(smoke/'selection_audit.json');assert a['smoke'] and s['smoke'] and a['contexts']==28 and a['fresh_physical_branches']==56 and s['selections']==84;assert a['source_sha256']==sha256(resolve('scripts/audit_followup_cube_shift_candidate_bank.py'));assert s['source_sha256']==sha256(resolve('scripts/analyze_followup_cube_shift_candidate_bank.py'));verify(read(smoke/'manifest.json')['inputs']);verify(a['inputs']);verify(s['helpers'])
    OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'smoke_bank_audit_sha256':sha256(smoke/'raw_audit.json'),'smoke_selection_audit_sha256':sha256(smoke/'selection_audit.json'),'nominal_calibration_dependency_pid':calibration_pid,'nominal_calibration_pipeline_lock_sha256':sha256(BASE/'cube_policy_calibration_bank/pipeline_lock.json'),'scope':'Previously inspected12developmentfamilies; fullstage uses24newnominalcalibration rules unchanged. No shifted-label tuning.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for stage,script in [('collect','scripts/run_followup_cube_shift_candidate_bank.py'),('audit','scripts/audit_followup_cube_shift_candidate_bank.py'),('analyze','scripts/analyze_followup_cube_shift_candidate_bank.py')]:
        if stage=='analyze':
            gate=BASE/'cube_policy_selection/pipeline_complete.json'
            while not gate.exists():
                try:os.kill(calibration_pid,0)
                except ProcessLookupError:raise RuntimeError('Nominal calibration ended before audit; inspect original run, do not restart')
                write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':'waiting_nominal_calibration','verified_dependency_pid':calibration_pid,'updated_unix':time.time()});time.sleep(10)
        verify(lock['inputs']);write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'status':'failed','stage':stage,'returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    verify(lock['inputs']);report();write_json(OUT/'pipeline_complete.json',{'contexts':336,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'selection_audit_sha256':sha256(OUT/'selection_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibration-pid',type=int,required=True);a=p.parse_args();main(a.calibration_pid)
