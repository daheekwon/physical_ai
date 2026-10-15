"""Freeze reproduced Cube compiler smoke; collect/audit new24calibration nominals."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_policy_calibration_nominal';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_policy_calibration_nominal.json','configs/followup_cube.json','scripts/run_followup_cube_policy_calibration_nominal.py','scripts/audit_followup_cube_policy_calibration_nominal.py','scripts/run_followup_cube_policy_calibration_nominal_pipeline.py']
def verify():
    for p,h in read(OUT/'pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p

def report():
    a=read(OUT/'raw_audit.json');r=read(OUT/'analysis.json');reg=read(OUT/'initial_registry.json');assert a['episodes']==24 and a['fresh_physics_samples']==180000 and a['blocks']==1728 and a['analysis_sha256']==sha256(OUT/'analysis.json')
    for p,h in {**read(OUT/'manifest.json')['inputs'],**a['inputs'],**a['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    pct=lambda value:f'{100*value:.2f}%';s=r['summary'];lines=['<!-- BEGIN CUBE NEW POLICY CALIBRATION NOMINAL -->','## Cube용 새 calibration family의 nominal 검증','',
    '기존12개 policy development family와 별도로 **24개 calibration family**에서 같은 native CubePlanOracle와 인과적 past-PD H5 compiler를 실행했다. 각 episode는15초·300action으로 고정했다. 기존 family1개의 smoke는 원본의 모든 저장 배열과 일치했고, 새24개 실행에서는 원래 plan과 전체180,000개 물리 표본,1,728개 compiler block을 검증했다.','',
    f"Native 목표 도달 경험 비율은 {pct(s['success_any']['mean'])}, 종료 시 목표 도달은 {pct(s['success_final']['mean'])}, 종료 시 목표에 도달하고 로봇 접촉이 없는 비율은 {pct(s['released_success_final']['mean'])}였다. 이 환경의 목표 도달 기준은 위치40mm이며, 수정 전후 물체 궤적 보존의2mm·full 회전2도와 다르다. 실패 episode도 제외하지 않았다.",'',
    '새24개 calibration 초기 상태와 이후 평가를 위해 예약한64개 초기 상태는 원래10,000개 expert 초기 qpos/qvel 및 기존12개 native policy 초기 상태와 비교했다. 새 family끼리도 초기 상태가 겹치지 않는다. 예약된 test family에서는 정책 행동이나 그 결과를 실행하지 않았으며, 방법과 규칙을 잠근 뒤 평가해야 한다. 초기 상태 분리는 모델의 사전학습 미노출 증명이 아니다.','',
    '이 단계에서 계획은 초기 실제 상태와 목표를 아는 privileged native oracle로 생성된다. 이미지 정책으로 부르지 않는다. Compiler는 각 chunk 시작의 실제 robot 관측과 과거 명령으로 다음5개 상대 행동을 미리 확정한다. 감사는 controller의 캐시된 좌표를 pre-integration qpos에서 재구성하고, 과거 robot/gripper history·목표·계수 피팅·명령 변환을 독립 수식으로 확인했다.','',
    'Nominal 검증은 correction의 효과나 보존 승인 규칙의 calibration을 완료한 것이 아니다. 다음 단계는 이24개 family의 공통 후보에서 승인 규칙을 정하고, development와 조건 변화에 적용한 뒤 별도로 잠근 새 test에서 확인하는 것이다.','',
    '근거: [실행 프로토콜](../artifacts/followup_v1/cube_policy_calibration_nominal/manifest.json), [초기 family 등록](../artifacts/followup_v1/cube_policy_calibration_nominal/initial_registry.json), [원시 감사](../artifacts/followup_v1/cube_policy_calibration_nominal/raw_audit.json), [nominal 분석](../artifacts/followup_v1/cube_policy_calibration_nominal/analysis.json).','<!-- END CUBE NEW POLICY CALIBRATION NOMINAL -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in text:i=text.index(begin);j=text.index(end)+len(end);text=text[:i]+section+text[j:]
    else:i=text.index('<!-- BEGIN CUBE SHIFT HISTORY -->');text=text[:i]+section+'\n\n'+text[i:]
    doc.write_text(text);write_json(OUT/'report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['manifest.json','initial_registry.json','raw_audit.json','analysis.json','evaluation.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()})

def main():
    smoke=BASE/'cube_policy_calibration_nominal_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['episodes']==1 and a['blocks']==72 and a['fresh_physics_samples']==7500 and a['source_sha256']==sha256(resolve(SOURCES[3]));assert a['analysis_sha256']==sha256(smoke/'analysis.json')
    for p,h in {**read(smoke/'manifest.json')['inputs'],**a['inputs'],**a['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'smoke_raw_audit_sha256':sha256(smoke/'raw_audit.json'),'scope':'New24Cube nominalcalibration only; reserve64testinitialfamilies, no policytest outcomes before later method lock.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for stage,script in [('collect',SOURCES[2]),('audit',SOURCES[3])]:
        verify();write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'status':'failed','stage':stage,'returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    verify();report();write_json(OUT/'pipeline_complete.json',{'episodes':24,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
