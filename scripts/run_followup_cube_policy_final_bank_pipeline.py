"""Run separately audited final wrappers under the pre-nominal Cube method lock."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_policy_final_bank';FINAL=BASE/'cube_policy_final';read=lambda p:json.loads(p.read_text())
SOURCES=['scripts/cube_policy_final_bank_adapter.py','scripts/run_followup_cube_policy_final_bank.py','scripts/audit_followup_cube_policy_final_bank.py','scripts/analyze_followup_cube_policy_final.py','scripts/run_followup_cube_policy_final_bank_pipeline.py']
def verify(values):
    for p,h in values.items():assert sha256(resolve(p))==h,p

def report():
    a=read(FINAL/'raw_audit.json');r=read(FINAL/'analysis.json');ba=read(OUT/'raw_audit.json');cfg=read(FINAL/'final_protocol.json')['config'];assert a['families']==64 and a['states']==ba['states']==r['states'] and a['selections']==3*a['states'] and a['analysis_sha256']==sha256(FINAL/'analysis.json');verify(read(FINAL/'final_protocol.json')['inputs']);verify(read(OUT/'manifest.json')['inputs']);verify(ba['inputs']);verify(ba['audit_helpers']);verify(a['helpers']);primary=next(x for x in r['paired'] if (x['method'],x['baseline'],x['metric'])==(cfg['primary_method'],cfg['primary_baseline'],cfg['primary_metric']));pct=lambda x:'미정의' if x is None else f'{100*x:.2f}%';lo,hi=primary['ci95'];missing=sum(len(x) for x in r['missing_phases'].values())
    lines=['<!-- BEGIN CUBE POLICY INDEPENDENT FINAL -->','## Cube 새64개 family의 독립 correction 평가','',f"Nominal 정책 실행 전에 모델·승인 규칙·후보 생성·phase 선정식·주 비교를 잠근64개 family에서 **{a['states']}개 상태·{ba['candidates']}개 후보·{a['selections']}개 선택**을 실행하고 감사했다. Nominal 궤적에서 조건에 맞는 phase가 없는 경우는{missing}개였으며 목록을 원본 분석에 포함했다. 실패한 nominal episode도 제외하지 않았다.",'',f"주 비교인 **LeWM−controller의 U 차이는 {100*primary['mean']:+.2f}%p**, paired family95% 구간은[{100*lo:+.2f},{100*hi:+.2f}]%p다. 사전 지정한 실질 우위 조건(U10%p 이상·구간 하한>0·선택 후 위반율10% 이하·적용25% 이상)을 {'충족했다' if r['supported_practical_primary_advantage'] else '충족하지 못했다'}. 미충족은 방법 간 동등성이나 다른 조건에서의 무용성 증명이 아니다.",'','| 방법 | 동시 달성 U | 수정 적용 | 선택 후 위반율 | 평균 실제 jerk 감소 |','| --- | --- | --- | --- | --- |']
    for x in r['summary']:lines.append(f"| {x['method']} | {pct(x['joint']['mean'])} | {pct(x['active']['mean'])} | {pct(x['selected_risk']['mean'])} | {pct(x['jerk_gain']['mean'])} |")
    lines+=['','U는 같은 root에서 원래 H5＋고정3명령을 실행한 물체 궤적과 비교해, 모든2ms 표본에서 위치2mm·전체3D 회전2도를 지키면서 실제 robot EEF XYZ jerk를10% 이상 줄인 상태의 비율이다. 최종 상태만 맞는 것을 궤적 보존으로 세지 않는다. 무수정 fallback도 U에 포함하지 않는다. 모든 방법이 동일한 다항식 후보와 예상 실행 jerk 순위를 공유하며 gripper 명령은 고정했다.','', '규칙은 별도24개 calibration family에서 결정한 값을 그대로 적용했다. Test threshold·representation·radius 재선택은 하지 않았다. 모든 선택은 각 수정 물리 분기 전에 저장했고, 감사에서 동일한 cost/norm/index 순위를 독립 구현으로 재계산했다. 모델·방법은 test nominal 전에 잠갔으며 최종 실행·감사 wrapper는 기존 development family의 smoke 통과 후 새 correction 실행 전에 고정했다.','',f"모든{ba['candidates']}개 후보의 dense 물체 오차·실제 jerk·다항식 제약을 검증했고 모든 LeWM 예측을 새로 계산했다. 각 상태의 RGB 이력·명령 조립·물리 prefix를 확인하고{ba['fresh_physical_branches_all_families']}개 물리 분기를 새로 재현했다. 선택 및 모든 family 통계 구간과 주 판정도 독립 대조했다.",'', '접촉 기준과의 비교는 탐색적 보조 결과다. U의 분모는 존재하는 nominal phase 상태이며 bootstrap은 같은 family의 상태를 함께 묶었다. 이 실험은 privileged 초기 plan에 기반한 단발 correction이고 pixel policy·반복 episode 보존·조건 적응·지연 실행 성능을 뜻하지 않는다. 초기 family 분리는 모델 사전학습 미노출이나 모집단 위험 인증도 아니다.','', '근거: [최종 잠금](../artifacts/followup_v1/cube_policy_final/final_protocol.json), [독립 결과](../artifacts/followup_v1/cube_policy_final/analysis.json), [후보 감사](../artifacts/followup_v1/cube_policy_final_bank/raw_audit.json), [선택·통계 감사](../artifacts/followup_v1/cube_policy_final/raw_audit.json).','<!-- END CUBE POLICY INDEPENDENT FINAL -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=doc.read_text()
    if lines[0] in text:i=text.index(lines[0]);j=text.index(lines[-1])+len(lines[-1]);text=text[:i]+section+text[j:]
    else:i=text.index('<!-- BEGIN CUBE POLICY FINAL NOMINAL -->');text=text[:i]+section+'\n\n'+text[i:]
    doc.write_text(text);write_json(FINAL/'report_evidence.json',{'inputs':{p:sha256(FINAL/p) for p in ['final_protocol.json','analysis.json','raw_audit.json','selected.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()})

def main(nominal_pid):
    sm=BASE/'cube_policy_final_bank_smoke';a=read(sm/'raw_audit.json');sa=read(BASE/'cube_policy_final_analysis_smoke/raw_audit.json');assert a['smoke'] and sa['smoke'] and a['states']==sa['states']==4 and a['candidates']==84 and a['fresh_physical_branches_all_families']==8 and sa['selections']==12;assert a['source_sha256']==sha256(resolve(SOURCES[2])) and sa['source_sha256']==sha256(resolve(SOURCES[3]));verify(read(sm/'manifest.json')['inputs']);verify(a['inputs']);verify(a['audit_helpers']);verify(sa['helpers']);OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'smoke_bank_audit_sha256':sha256(sm/'raw_audit.json'),'smoke_selection_audit_sha256':sha256(BASE/'cube_policy_final_analysis_smoke/raw_audit.json'),'final_protocol_sha256':sha256(FINAL/'final_protocol.json'),'nominal_dependency_pid':nominal_pid}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    gate=BASE/'cube_policy_final_nominal/pipeline_complete.json'
    while not gate.exists():
        try:os.kill(nominal_pid,0)
        except ProcessLookupError:raise RuntimeError('Finalnominal terminated before audit; inspect original, no restart')
        write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':'waiting_nominal_audit','verified_dependency_pid':nominal_pid,'updated_unix':time.time()});time.sleep(10)
    for stage,script in [('collect',SOURCES[1]),('audit_bank',SOURCES[2]),('analyze_audit',SOURCES[3])]:
        verify(lock['inputs']);verify(read(FINAL/'final_protocol.json')['inputs']);write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'status':'failed','stage':stage,'returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    verify(lock['inputs']);report();complete={'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'selection_audit_sha256':sha256(FINAL/'raw_audit.json'),'report_evidence_sha256':sha256(FINAL/'report_evidence.json'),'final_protocol_sha256':sha256(FINAL/'final_protocol.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()};write_json(OUT/'pipeline_complete.json',complete);write_json(FINAL/'pipeline_complete.json',complete);write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nominal-pid',type=int,required=True);a=p.parse_args();main(a.nominal_pid)
