"""Audit nominal actualselectors, then evaluate frozen factors on audited shifts."""
import argparse,json,hashlib,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_adaptation_factorial';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_adaptation_factorial.json','scripts/cube_adaptation_factorial_core.py','scripts/cube_adaptation_selection_core.py','scripts/run_followup_cube_adaptation_factorial.py','scripts/audit_followup_cube_adaptation_factorial.py','scripts/run_followup_cube_adaptation_factorial_pipeline.py','scripts/output_replacement_core.py','zj_wam/cube_experiment.py','scripts/analyze_followup_response_selection.py','scripts/audit_retrained_dinowm_readouts.py','artifacts/followup_v1/cube_adaptation_calibration/settings.json']
def verify(values):
    for p,h in values.items():assert sha256(resolve(p))==h,p

def report(split):
    folder=OUT/split;a=read(folder/'raw_audit.json');r=read(folder/'analysis.json');assert a['analysis_sha256']==sha256(folder/'analysis.json') and a['actual_selector_calls']==(828 if split=='nominal' else 2016);verify(read(folder/'protocol.json')['inputs']);verify(a['inputs']);verify(a['helpers']);tag='CUBE ADAPTATION '+split.upper();pct=lambda x:'미정의' if x is None else f'{100*x:.2f}%'
    lines=[f'<!-- BEGIN {tag} -->',f"## Cube 적응 네 조건의 {'nominal/development' if split=='nominal' else '물성 변화'} 실제 selector 실행",'',f"고정한 nominal 보정값 α=0·반경0.4와 contact verifier로 **{a['states']}개 상태·{a['actual_selector_calls']}회 실제 selector 실행**을 평가했다. 고정/allowance만/계산량만/둘 다와 고정 반경·optimizer-only 대조군을 같은 potential 후보에 적용했다. 조건별 재보정은 하지 않았다. α=0이므로 allowance는 과거 오차 계산 비용을 쓰지만 실제 허용 반경을 줄이지 않는다.",'', '| 집합 | 조건 | 방법 | U | 선택 후 위반율 | 평균 실제 jerk 감소 | 조기 종료 | 후보 점수 계산 평균 | p50 / p95 시간(ms) |','| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for x in r['summary']:
        if split=='nominal' and x['split']=='calibration':continue
        lines.append(f"| {x['split']} | {x['condition']} | {x['mode']} | {pct(x['joint']['mean'])} | {pct(x['selected_risk']['mean'])} | {pct(x['jerk_gain']['mean'])} | {pct(x['early_stop']['mean'])} | {x['scored_candidates']['mean']:.2f} | {1000*x['seconds_p50']:.2f} / {1000*x['seconds_p95']:.2f} |")
    lines+=['', 'U는 동일 조건의 원래 H5＋3 실행 대비 모든2ms 물리 표본에서 물체 위치2mm·전체 회전2도를 지키며 실제 EEF XYZ jerk를10% 이상 줄인 비율이다. 최종 상태만 같은 것을 궤적 보존으로 세지 않는다. Nominal calibration 재대입 결과는 원본에 별도로 구분했고 development나 조건 변화 성공으로 제시하지 않는다.','', '매 호출에서 실제로 optimizer 후보를 생성·평가한 후, 규칙이 요구할 때만 random 후보를 추가 생성·평가했다. 기록된 행동·계수·점수·예상 반응이 기존 감사된 후보 bank와 정확히 같음을 확인한 뒤 해당 후보의 물리 결과를 재사용했다. 따라서 이 단계의 새 실행은 selector 연산이며 새 물리 분기나 이미지를 생성한 실험은 아니다.','', '독립 감사는 과거125/125 구간 예측 오차, 반경, 조기 종료, 허용 조건과 cost/norm/index 순위, solver 호출·iteration, 시간 내역 합, 선택된 dense 물체 오차·실제 jerk를 다시 계산했다. Family bootstrap과 고정 조건 대비 paired 차이도 독립 구현으로 대조했다.','', '시간은 입력 context를 받은 뒤의 과거 오차 계산·후보 생성·응답 및 score 계산·선택이다. 입력 수집·파일 I/O·물리 평가를 포함하지 않는다. 같은 CPU process에서 상태마다 여섯 조건의 순서를 회전시킨1회 측정이며 다른 작업이 실행되는 shared host다. p50/p95 및 deadline 초과율은 관측된 selector 지연으로, 실제 control loop 지연 주입이나 독립 장치 deadline 보장이 아니다.','']
    if r['additional_matched_budget_control_needed']:lines.append('Compute/both가 일부 상태에서 조기 종료했으므로 같은 평균 계산량의 고정 대조군을 추가해야 계산 적응의 효과를 구분할 수 있다. 현재 결과만으로 그 비교가 완료됐다고 보지 않는다.')
    else:lines.append('Compute/both는 이 평가에서 조기 종료하지 않았다. 모든 상태에서 fixed와 같은 전체 후보·solver 계산 예산을 사용하므로 연산량 적응이 계산을 줄였다는 근거가 없다. Walltime 차이를 유용한 계산 절감으로 해석하지 않는다.')
    lines+=['', '반경 축소가 선택되지 않았거나 계산량이 줄지 않은 결과를 적응 성공으로 바꾸어 부르지 않는다. 반복 episode·독립 최종·실제 지연 성능은 별도 범위다.','',f'근거: [실행 프로토콜](../artifacts/followup_v1/cube_adaptation_factorial/{split}/protocol.json), [결과·paired 구간](../artifacts/followup_v1/cube_adaptation_factorial/{split}/analysis.json), [독립 감사](../artifacts/followup_v1/cube_adaptation_factorial/{split}/raw_audit.json).',f'<!-- END {tag} -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=doc.read_text()
    if lines[0] in text:i=text.index(lines[0]);j=text.index(lines[-1])+len(lines[-1]);text=text[:i]+section+text[j:]
    else:i=text.index('<!-- BEGIN CUBE ADAPTATION CALIBRATION -->');text=text[:i]+section+'\n\n'+text[i:]
    doc.write_text(text);write_json(folder/'report_evidence.json',{'inputs':{p:sha256(folder/p) for p in ['analysis.json','raw_audit.json','evaluation.csv','protocol.json']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()})

def wait_gate(path,pid,stage):
    while not path.exists():
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError((stage,'Dependency ended before gate; inspect original, no restart'))
        write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'verified_dependency_pid':pid,'updated_unix':time.time()});time.sleep(10)

def main(nominal_pid,shift_pid):
    a=read(BASE/'cube_adaptation_smoke/raw_audit.json');assert a['actual_selector_calls']==48 and a['fullbank_exact_calls']==40;verify(read(BASE/'cube_adaptation_smoke/protocol.json')['inputs']);cal=read(BASE/'cube_adaptation_calibration/raw_audit.json');assert cal['trials']==1620;lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'nominal_source_pid':nominal_pid,'shift_bank_pid':shift_pid,'smoke_audit_sha256':sha256(BASE/'cube_adaptation_smoke/raw_audit.json'),'calibration_audit_sha256':sha256(BASE/'cube_adaptation_calibration/raw_audit.json')}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    wait_gate(OUT/'nominal/execution_complete.json',nominal_pid,'waiting_nominal_selectors')
    stages=[('audit_nominal','scripts/audit_followup_cube_adaptation_factorial.py','nominal'),('collect_shifted','scripts/run_followup_cube_adaptation_factorial.py','shifted'),('audit_shifted','scripts/audit_followup_cube_adaptation_factorial.py','shifted')]
    for stage,script,split in stages:
        if stage=='collect_shifted':wait_gate(BASE/'cube_shift_candidate_bank/pipeline_complete.json',shift_pid,'waiting_shift_bank_audit')
        verify(lock['inputs']);write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script)),'--split',split],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'status':'failed','stage':stage,'returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        if stage.startswith('audit'):report(split)
        print('COMPLETE',stage,flush=True)
    verify(lock['inputs']);write_json(OUT/'pipeline_complete.json',{'nominal_actual_calls':828,'shifted_actual_calls':2016,'nominal_audit_sha256':sha256(OUT/'nominal/raw_audit.json'),'shifted_audit_sha256':sha256(OUT/'shifted/raw_audit.json'),'nominal_report_sha256':sha256(OUT/'nominal/report_evidence.json'),'shifted_report_sha256':sha256(OUT/'shifted/report_evidence.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nominal-pid',required=True,type=int);p.add_argument('--shift-pid',required=True,type=int);a=p.parse_args();main(a.nominal_pid,a.shift_pid)
