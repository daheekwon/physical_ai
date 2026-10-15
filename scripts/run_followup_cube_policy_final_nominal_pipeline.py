"""Lock Cube independent test methods before any reserved nominal policy action."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_policy_final_nominal';FINAL=BASE/'cube_policy_final';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_policy_final.json','configs/followup_cube_policy_selection.json','configs/followup_cube_policy_calibration_bank.json','configs/followup_cube_plan_chunks.json','configs/followup_cube_native_policy.json','configs/followup_cube.json','scripts/cube_policy_final_core.py','scripts/run_followup_cube_policy_final_nominal.py','scripts/audit_followup_cube_policy_final_nominal.py','scripts/run_followup_cube_policy_final_nominal_pipeline.py','scripts/run_followup_cube_plan_chunks.py','scripts/cube_plan_chunk_core.py','scripts/run_followup_cube_native_policy.py','scripts/run_followup_cube.py','scripts/cube_policy_refinement_core.py','scripts/cube_policy_calibration_bank_core.py','scripts/output_replacement_core.py','scripts/audit_followup_output_replacement.py','scripts/analyze_followup_response_selection.py','scripts/audit_retrained_dinowm_readouts.py','scripts/analyze_followup_cube_plan_chunks.py','scripts/audit_followup_dinowm_policy_final.py','scripts/analyze_followup_cube_policy_selection.py','zj_wam/cube_execution.py','zj_wam/cube_experiment.py','artifacts/followup_v1/cube/manifest.json','data/checkpoints/lewm-cube/weights.pt','data/checkpoints/lewm-cube/config.json','artifacts/followup_v1/cube_policy_selection/rules.json','artifacts/followup_v1/cube_policy_selection/raw_audit.json','artifacts/followup_v1/cube_policy_selection/pipeline_complete.json','artifacts/followup_v1/cube_policy_calibration_nominal/initial_registry.json']
def verify(values):
    for p,h in values.items():assert sha256(resolve(p))==h,p

def report():
    a=read(OUT/'raw_audit.json');r=read(OUT/'analysis.json');assert a['episodes']==64 and a['fresh_physics_samples']==480000 and a['blocks']==4608 and a['analysis_sha256']==sha256(OUT/'analysis.json');verify(read(OUT/'manifest.json')['inputs']);verify(a['inputs']);verify(a['audit_helpers']);pct=lambda x:f'{100*x:.2f}%'
    lines=['<!-- BEGIN CUBE POLICY FINAL NOMINAL -->','## Cube 독립64개 family의 nominal reference 검증','', '새24개 calibration과 기존12개 development 분석을 마친 뒤, 예약해 둔64개 Cube family의 정책 실행 전에 방법·규칙·주 비교를 고정했다. LeWM−controller의 U 차이를 주 비교로 정했고 접촉 기준 비교는 탐색적으로 구분했다. 모든 방법은 기존 native plan, 같은 causal H5 compiler, 동일 다항식 후보와 예상 XYZ jerk 순위를 사용한다.','',f"64개 nominal episode의 전체480,000개 물리 표본과4,608개 compiler block, 초기 plan과 등록된 초기 상태를 재현했다. 원래 정책의 종료 목표 도달 비율은 {pct(r['summary']['success_final']['mean'])}였다. 실패 episode도 제외하지 않았다. 이 목표 도달은40mm 위치 기준이며, 이후 correction의 물체 궤적 보존2mm·전체 회전2도와 다르다.",'', '접근·잡기·운반·놓기 상태의 선정식도 test 실행 전에 잠갔다. 해당 phase가 없는 episode는 누락 내역을 보고하며, correction 결과를 보고 상태를 골라내지 않는다. 알려진 초기 상태와 목표로 생성한 privileged native plan을 쓰므로 이미지 정책이라고 부르지 않는다. 초기 family 분리는 모델 사전학습 미노출의 증명이 아니다.','', '**이 단계는 독립 평가용 원래 실행의 검증이다.** 아직64개 family에서 correction 효과를 검증한 결과가 아니다. 잠긴 규칙으로 공통 후보를 평가하고 선택·물리 오차·jerk·family 신뢰구간을 감사하는 단계가 뒤따른다.','', '근거: [최종 방법 잠금](../artifacts/followup_v1/cube_policy_final/final_protocol.json), [nominal 감사](../artifacts/followup_v1/cube_policy_final_nominal/raw_audit.json), [nominal 분석](../artifacts/followup_v1/cube_policy_final_nominal/analysis.json).','<!-- END CUBE POLICY FINAL NOMINAL -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=doc.read_text()
    if lines[0] in text:i=text.index(lines[0]);j=text.index(lines[-1])+len(lines[-1]);text=text[:i]+section+text[j:]
    else:i=text.index('<!-- BEGIN CUBE NEW POLICY CALIBRATED SELECTION -->');text=text[:i]+section+'\n\n'+text[i:]
    doc.write_text(text);write_json(OUT/'report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['manifest.json','raw_audit.json','analysis.json','evaluation.csv']},'final_protocol_sha256':sha256(FINAL/'final_protocol.json'),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()})

def main():
    smoke=BASE/'cube_policy_final_nominal_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['episodes']==1 and a['blocks']==72 and a['fresh_physics_samples']==7500;assert a['source_sha256']==sha256(resolve('scripts/audit_followup_cube_policy_final_nominal.py'));verify(read(smoke/'manifest.json')['inputs']);verify(a['inputs']);verify(a['audit_helpers']);selection=BASE/'cube_policy_selection';sa=read(selection/'raw_audit.json');assert sa['states']==138 and sa['candidate_rows']==2698 and sa['selections']==414 and read(selection/'pipeline_complete.json')['selection_audit_sha256']==sha256(selection/'raw_audit.json');verify(read(selection/'protocol.json')['inputs'])
    # Verify the phase implementation against all36 audited pre-test source families.
    from scripts.cube_policy_final_core import phase_roots
    from scripts.cube_policy_calibration_bank_core import source_dir
    import numpy as np
    cfg=read(resolve(SOURCES[0]));catalogue=read(BASE/'cube_policy_calibration_bank/manifest.json')['phases']
    for seed,phases in catalogue.items():
        with np.load(source_dir(int(seed))/f'{seed}_nominal.npz') as src:assert phase_roots(src,cfg)==phases
    reg=read(BASE/'cube_policy_calibration_nominal/initial_registry.json');seeds=list(range(cfg['test_seed_start'],cfg['test_seed_start']+cfg['test_families']));assert all(reg['families'][str(seed)]['split']=='reserved_test' for seed in seeds)
    FINAL.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);protocol={'config':cfg,'rules':read(selection/'rules.json')['rules'],'test_seeds':seeds,'inputs':{p:sha256(resolve(p)) for p in SOURCES},'nominal_smoke_audit_sha256':sha256(smoke/'raw_audit.json'),'phase_fixture_families':len(catalogue),'scope':cfg['scope']};target=FINAL/'final_protocol.json'
    if target.exists():assert read(target)==protocol
    else:
        assert not list(OUT.glob('*_nominal.json')),'Reserved nominal outputs predate method lock';write_json(target,protocol)
    for stage,script in [('collect','scripts/run_followup_cube_policy_final_nominal.py'),('audit','scripts/audit_followup_cube_policy_final_nominal.py')]:
        verify(protocol['inputs']);write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'status':'failed','stage':stage,'returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    verify(protocol['inputs']);report();write_json(OUT/'pipeline_complete.json',{'episodes':64,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'final_protocol_sha256':sha256(target),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
