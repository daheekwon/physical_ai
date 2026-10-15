"""Freeze audited Cube candidate/selection smoke, then calibrate on new families."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_policy_calibration_bank';SEL=BASE/'cube_policy_selection';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_policy_calibration_bank.json','configs/followup_cube_policy_selection.json','scripts/cube_policy_calibration_bank_core.py','scripts/run_followup_cube_policy_calibration_bank.py','scripts/audit_followup_cube_policy_calibration_bank.py','scripts/analyze_followup_cube_policy_selection.py','scripts/audit_followup_cube_policy_selection.py','scripts/run_followup_cube_policy_calibration_bank_pipeline.py','scripts/output_replacement_core.py','scripts/audit_followup_output_replacement.py','scripts/audit_retrained_dinowm_readouts.py','scripts/analyze_followup_response_selection.py','scripts/cube_policy_refinement_core.py','scripts/cube_plan_chunk_core.py','scripts/analyze_followup_cube_plan_chunks.py','scripts/audit_followup_cube_policy_refinement.py','artifacts/followup_v1/cube/manifest.json']
def verify_inputs(record):
    for p,h in record.items():assert sha256(resolve(p))==h,p

def report():
    bank=read(OUT/'raw_audit.json');audit=read(SEL/'raw_audit.json');result=read(SEL/'analysis.json');rules={x['method']:x['rule'] for x in read(SEL/'rules.json')['rules']};assert bank['states']==audit['states']==result['states'];assert bank['candidates']==audit['candidate_rows'];assert audit['analysis_sha256']==sha256(SEL/'analysis.json');assert audit['calibration_trials']>0
    verify_inputs(read(OUT/'manifest.json')['inputs']);verify_inputs(bank['inputs']);verify_inputs(bank['audit_helpers']);verify_inputs(read(SEL/'protocol.json')['inputs']);verify_inputs(audit['helpers'])
    missing=result['missing_phases'];missing_count=sum(len(x) for x in missing.values());pct=lambda x:'미정의' if x is None else f'{100*x:.2f}%'
    lines=['<!-- BEGIN CUBE NEW POLICY CALIBRATED SELECTION -->','## Cube 새 calibration family에서 보정한 correction 선택','',f"새24개 calibration family와 기존12개 development family에서 접근·잡기·운반·놓기 시점을 nominal 궤적으로 정했다. 총 {result['states']}개 상태·{bank['candidates']}개 공통 후보를 검증했다. 예정한 phase 중 {missing_count}개는 nominal에서 해당 조건이 발생하지 않아 존재하지 않으며, 누락 내역을 원본 분석에 기록했다. Nominal 작업 실패 family도 제외하지 않았다.",'','초기 상태를 아는 native CubePlanOracle의 고정 plan과 현재·과거 robot 관측으로 H5＋후속3명령을 미리 계산한다. 모든 방법은 같은 비-WM 다항식 후보, 고정된 float32 gripper 명령, 예상 XYZ 실행 jerk 순위를 사용한다. LeWM은 재학습하지 않고 native CPU 예측을 사용한다. 접촉 기준은 gripper proprioception/명령과 간단한 물체 크기 근사에 의존하며, 정확한 물체 동역학 모델은 아니다.','', '허용 threshold는 새24개 calibration family만으로 결정했다. 경험적 선택 후 위반율10% 이하·수정 적용 비율25% 이상을 만족하는 threshold 중 예상10% jerk 감소 비율, 이어 평균 예상 감소량을 최대화했다. 충족하는 threshold가 없으면 비활성화한다. 같은 데이터의 calibration 성능은 재대입 결과이며 위험 인증이 아니다.','', '| 방법 | 규칙 채택 | Development U | 수정 적용 비율 | 선택 후 위반율 | 평균 실제 jerk 감소 |','| --- | --- | --- | --- | --- | --- |']
    for method,rule in rules.items():
        r=next(x for x in result['summary'] if x['split']=='development' and x['phase']=='all' and x['method']==method)
        lines.append(f"| {method} | {'예' if rule['enabled'] else '적합 규칙 없음'} | {pct(r['joint']['mean'])} | {pct(r['active']['mean'])} | {pct(r['selected_risk']['mean'])} | {pct(r['jerk_gain']['mean'])} |")
    lines+=['','여기서 **물체 궤적 보존**은 같은 초기 상태의 원래 H5＋3 실행과 비교해 모든2ms 물리 표본에서 물체 위치 차이2mm·전체3D 회전 차이2도 이내임을 뜻한다. 최종 위치만 맞는 것은 통과하지 않는다. U는 이 조건을 지키면서 실제 robot EEF XYZ jerk 에너지를10% 이상 줄인 상태의 비율이며, 무수정 fallback은 U로 세지 않는다. 실제 jerk는 수정 직전75개 표본과 이후200개 표본에서 계산한다. 원래 작업 성공의 목표 위치40mm 기준과 구분한다.','',f"전체 후보의 형상·gripper 고정·과거 response fit·예상/실제 jerk·fullSO3 물체 오차를 독립 재계산했다. 모든 {bank['fresh_native_LeWM_predictions']}개 LeWM 예측을 새로 실행했고, 모든 상태의 RGB history·root contact·8명령의 인과적 조립과 {bank['fresh_physical_branches_all_families']}개 물리 분기를 재현했다. 선택 단계의 전체 threshold grid·선택·family bootstrap 구간도 독립 검증했다.",'', '기존12개 development family는 이전에 검사한 상태이므로 독립 최종 평가로 부르지 않는다. 위 U의 분모는 실제 존재하는 nominal phase 상태이며, 서로 같은 family의 상태는 bootstrap에서 묶었다. 예약된64개 test family의 정책 결과는 이 단계에서 사용하지 않았다. 이 결과만으로 반복 수정의 전체 궤적 보존·조건 적응·실시간 지연 성능을 주장하지 않는다.','', '근거: [후보 감사](../artifacts/followup_v1/cube_policy_calibration_bank/raw_audit.json), [보정 규칙](../artifacts/followup_v1/cube_policy_selection/rules.json), [분석과 paired 구간](../artifacts/followup_v1/cube_policy_selection/analysis.json), [독립 선택 감사](../artifacts/followup_v1/cube_policy_selection/raw_audit.json).','<!-- END CUBE NEW POLICY CALIBRATED SELECTION -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text()
    if lines[0] in s:i=s.index(lines[0]);j=s.index(lines[-1])+len(lines[-1]);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN CUBE NEW POLICY CALIBRATION NOMINAL -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(SEL/'report_evidence.json',{'inputs':{p:sha256(SEL/p) for p in ['analysis.json','raw_audit.json','rules.json','selected.csv','protocol.json']},'bank_raw_audit_sha256':sha256(OUT/'raw_audit.json'),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()})

def main():
    smoke=BASE/'cube_policy_calibration_bank_smoke';ss=BASE/'cube_policy_selection_smoke';a=read(smoke/'raw_audit.json');b=read(ss/'raw_audit.json');assert a['smoke'] and b['smoke'] and a['states']==b['states']==8 and a['candidates']==163 and a['fresh_physical_branches_all_families']==16 and b['selections']==24 and b['calibration_trials']==0
    assert a['source_sha256']==sha256(resolve('scripts/audit_followup_cube_policy_calibration_bank.py'));assert b['source_sha256']==sha256(resolve('scripts/audit_followup_cube_policy_selection.py'))
    for p in [smoke/'manifest.json',ss/'protocol.json']:verify_inputs(read(p)['inputs'])
    verify_inputs(a['inputs']);verify_inputs(a['audit_helpers']);verify_inputs(b['helpers']);assert b['analysis_sha256']==sha256(ss/'analysis.json')
    OUT.mkdir(exist_ok=True);SEL.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'candidate_smoke_audit_sha256':sha256(smoke/'raw_audit.json'),'selection_smoke_audit_sha256':sha256(ss/'raw_audit.json'),'scope':'New24 calibration plus prior12 development; no reservedfinal execution.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    stages=[('collect','scripts/run_followup_cube_policy_calibration_bank.py'),('audit_bank','scripts/audit_followup_cube_policy_calibration_bank.py'),('calibrate','scripts/analyze_followup_cube_policy_selection.py'),('audit_selection','scripts/audit_followup_cube_policy_selection.py')]
    for stage,script in stages:
        verify_inputs(lock['inputs']);write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'status':'failed','stage':stage,'returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    verify_inputs(lock['inputs']);report();complete={'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'selection_audit_sha256':sha256(SEL/'raw_audit.json'),'report_evidence_sha256':sha256(SEL/'report_evidence.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()};write_json(OUT/'pipeline_complete.json',complete);write_json(SEL/'pipeline_complete.json',complete);write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
