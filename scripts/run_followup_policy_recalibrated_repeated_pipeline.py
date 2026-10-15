"""Freeze successful smoke and run full repeated new-policy-rule diagnosis."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'policy_recalibrated_repeated';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_policy_recalibrated_repeated.json','scripts/run_followup_policy_recalibrated_repeated.py','scripts/audit_followup_policy_recalibrated_repeated.py','scripts/run_followup_policy_recalibrated_repeated_pipeline.py']
def verify():
    for p,h in read(OUT/'pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p

def report():
    audit=read(OUT/'raw_audit.json');analysis=read(OUT/'analysis.json');assert audit['analysis_sha256']==sha256(OUT/'analysis.json') and audit['episodes']==108 and audit['families']==36 and audit['blocks']==3780;manifest=read(OUT/'manifest.json')
    for p,h in {**manifest['inputs'],**audit['inputs'],**audit['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    pct=lambda x:'미정의' if x is None else f'{100*x:.2f}%';lines=['<!-- BEGIN RECALIBRATED POLICY REPEATED -->','## Policy calibration 규칙의 반복 실행과 누적 보존','',
    '**새 policy calibration에서 정한 규칙**을 같은 정책의 자기 관측 기반 반복 correction에 적용했다. 기존 expert 규칙 전이와 구분하는 실험이다. Calibration24개·development12개 family에서 제어기·접촉 모델·LeWM의 **108개 episode·3,780개 국소 구간**을 실행하고 감사했다. 비교 기준은 이미 독립 검증한 같은36개 family의 nominal 정책이다.','',
    '각 episode는300action(30초), 매8action마다 현재 자기 관측으로 원래 frozen DP를 다시 호출한다. H5 다항식 수정 뒤3개 상대 action은 유지하고 다음 policy 호출 시점도 동일하다. 모든 방법이 같은 비-WM 후보 생성기를 사용하지만 correction 이후 상태가 달라지므로 이후의 실제 후보 배열까지 같다고 주장하지 않는다. LeWM은 고정 nativeCPU, 정책은 고정 GPU checkpoint다.','',
    '| Split | 방법 | 전체 episode U | 한 번 이상 수정 | 수정 episode 중 전체 위반율 | Task 성공 | 평균 전체 jerk 감소 |','| --- | --- | --- | --- | --- | --- | --- |']
    for split in ['calibration','development']:
        for method in ['controller','contact','lewm_global']:
            x=next(y for y in analysis['summary'] if y['split']==split and y['method']==method);lines.append(f"| {split} | {method} | {pct(x['joint']['mean'])} | {pct(x['active']['mean'])} | {pct(x['selected_risk']['mean'])} | {pct(x['any_success']['mean'])} | {pct(x['jerk_gain']['mean'])} |")
    lines+=['','주 보존 대상은 episode 시작에서 고정한 nominal의 **시간 정렬된 물체 위치·회전 궤적**이다.100Hz 위치2px·회전2도,10Hz coverage 감소0.02를 요구한다. U에는 실제 pusher jerk10% 감소와0이 아닌 수정도 필요하다. Task 성공은 별도 지표다.','',
    '| Split | 방법 | 국소 H5＋suffix3 U | 국소 수정 적용 | 국소 수정 중 위반율 |','| --- | --- | --- | --- | --- |']
    for split in ['calibration','development']:
        for method in ['controller','contact','lewm_global']:
            x=next(y for y in analysis['local_summary'] if y['split']==split and y['method']==method);lines.append(f"| {split} | {method} | {pct(x['joint']['mean'])} | {pct(x['active']['mean'])} | {pct(x['selected_risk']['mean'])} |")
    lines+=['',
    '국소 비교는 해당 방법의 현재 실제 상태에서 원래8action을 실행한 반사실이 기준이다. 수정 종료·suffix 종료 coverage와 dense 물체 궤적을 검사한다. 전체 비교는 episode 처음부터 고정한 nominal이 기준이므로 이후 정책 반응·누적 편차까지 포함한다. 국소 U와 전체 U를 서로 대체하지 않는다.','',
    f"모든 후보 점수와 선택은 해당 correction 물리를 실행하기 전에 저장했다. 감사에서 전체32만4천개 물리 표본·32,508개 RGB를 재실행하고, 모든3,780개 국소 nominal 분기를 다시 실행했다. {audit['fresh_native_LeWM_predictions']:,}개 native LeWM 후보 점수와324개 정책 출력을 새 forward로 확인했다. 모든 입력 history·후보·제어기 응답·선택·독립 polygon coverage·jerk를 확인했다.", '',
    '같은 family에서 많은 구간을 관측했다고 독립 표본 수가 늘어난 것으로 세지 않는다. Family bootstrap 구간, 수정 episode 위반율의 정확 이항 구간 및 탐색적 paired 비교는 원본에 제공한다. Calibration family 결과는 학습된 규칙과 같은 family의 반복 분포에 대한 진단이며 독립 test가 아니다. Development 결과도 반복 실행 방법의 최종 검증으로 부르지 않는다.','',
    '이 실행은 기존 single-correction 규칙을 반복 적용한 고정 규칙 조건이다. Episode 위험을 새로 보정한 규칙, allowance/계산량 적응, 실제 지연 및 DINO 반복 성능을 검증한 결과가 아니다. 이후 방법 변경은 별도 development/calibration을 거쳐 새 최종 family에서 평가해야 한다.','',
    '근거: [고정 실행 프로토콜](../artifacts/followup_v1/policy_recalibrated_repeated/manifest.json), [분석·family 구간](../artifacts/followup_v1/policy_recalibrated_repeated/analysis.json), [전체 원시 감사](../artifacts/followup_v1/policy_recalibrated_repeated/raw_audit.json), [episode 지표](../artifacts/followup_v1/policy_recalibrated_repeated/evaluation.csv), [국소 지표](../artifacts/followup_v1/policy_recalibrated_repeated/local_evaluation.csv).','<!-- END RECALIBRATED POLICY REPEATED -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN POLICY FINAL TEST -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['analysis.json','raw_audit.json','manifest.json','evaluation.csv','local_evaluation.csv']},'source_sha256':sha256(Path(__file__)),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()})

def main():
    smoke=BASE/'policy_recalibrated_repeated_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['episodes']==3 and a['blocks']==9 and a['fresh_local_nominal_branches']==9 and a['all_episode_physics_and_RGB_replayed_exact'];assert a['source_sha256']==sha256(resolve(SOURCES[2])) and a['analysis_sha256']==sha256(smoke/'analysis.json')
    for p,h in {**read(smoke/'manifest.json')['inputs'],**a['inputs'],**a['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'smoke_raw_audit_sha256':sha256(smoke/'raw_audit.json'),'scope':'Fixed new-policy rules, full development/calibration repeated transfer; no finaltest or adaptation claim.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for stage,script in [('collect',SOURCES[1]),('audit',SOURCES[2])]:
        verify();write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'running','updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'failed','returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((script,r.returncode))
        print('COMPLETE',stage,flush=True)
    verify();report();write_json(OUT/'pipeline_complete.json',{'episodes':108,'families':36,'blocks':3780,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
