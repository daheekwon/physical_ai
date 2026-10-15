"""Wait for locked full policy bank audit, then calibrate/audit/report native verifiers."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_policy_selection';read=lambda p:json.loads(p.read_text())

def report():
    a=read(OUT/'raw_audit.json');r=read(OUT/'analysis.json');rules={x['method']:x['rule'] for x in read(OUT/'rules.json')['rules']};assert a['states']==108 and a['selections']==540 and a['analysis_sha256']==sha256(OUT/'analysis.json')
    pct=lambda v:'미정의' if v is None else f'{100*v:.2f}%'
    lines=['<!-- BEGIN DINO FRESH POLICY SELECTION -->','## 새로운 정책 family에서 보정한 동일 후보 verifier 비교','', '새 frozen Diffusion Policy nominal의 36개 초기 family에서 사전 지정한 제어 시점16·96·192를 사용했다. **24개 calibration family·72개 상태**로 각 승인 규칙을 별도로 보정하고, **12개 development family·36개 상태**에서 비교했다. Nominal 성공 여부로 상태를 고르지 않았다. 같은 비-WM H5 다항식 후보와 원래 후속3행동, 같은 예상 실행 jerk 순위를 모든 방법이 공유한다.','', '경험적 선택 후 위반율10% 이하, 접촉 상태의 위반율10% 이하, 최소 수정 적용 비율25%를 동시에 만족하는 threshold만 채택했다. 적합한 threshold가 없으면 그 방법은 비활성화한다. 이 수치는 소표본 위험 인증이 아니다. LeWM·DINO는 native CPU 예측을 사용하며 DINO는 별도 검증한 third-party retrain이다.','', '| 방법 | 규칙 채택 | Development U | 수정 적용 비율 | 선택 후 위반율 | 평균 실제 jerk 감소 |','| --- | --- | --- | --- | --- | --- |']
    for method,rule in rules.items():
        x=next(x for x in r['summary'] if x['split']=='development' and x['phase']=='all' and x['method']==method);lines.append(f"| {method} | {'예' if rule['enabled'] else '적합 규칙 없음'} | {pct(x['joint']['mean'])} | {pct(x['active']['mean'])} | {pct(x['selected_risk']['mean'])} | {pct(x['jerk_gain']['mean'])} |")
    lines+=['','U는 같은 시각의 물체 위치2px·회전2도와 endpoint/suffix coverage 감소0.02 안에서 실제 pusher jerk를10% 이상 줄인 상태의 비율이다. 모든 상태가 분모이며 fallback은 유용한 수정으로 세지 않는다. 이 결과는 한 번의 H5＋3 수정 평가이고 전체 정책 episode 궤적 보존이나 반복 실행 성공을 뜻하지 않는다.','', '같은 이미지/history에서 고정 모델이 후보를 평가했으며, 접촉 모델은 영상으로 추정한 물체 geometry와 pusher proprioception을 사용했다. Physics 미래는 평가·calibration label에만 사용했다. 전체 후보 물리 감사에 이어 모든 threshold·540개 선택·family 구간을 독립 재계산했다. 같은 후보 비교이며 전체 계산 비용을 맞춘 비교는 별도다.','', '개발 결과와 calibration 재대입 결과는 원본에 구분했다. 예약한64개 정책 test family의 결과는 이 단계에서 실행하지 않았다. 방법 잠금 후 독립 평가, 자기 관측으로 재계획하는 반복 correction, 조건 변화·실행 지연 검사가 남아 있다.','', '근거: [후보 물리 감사](../artifacts/followup_v1/dinowm_policy_bank/raw_audit.json), [보정 규칙](../artifacts/followup_v1/dinowm_policy_selection/rules.json), [분석·paired 구간](../artifacts/followup_v1/dinowm_policy_selection/analysis.json), [독립 선택 감사](../artifacts/followup_v1/dinowm_policy_selection/raw_audit.json).','<!-- END DINO FRESH POLICY SELECTION -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN DINO COMMON VERIFIER -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{p:sha256(OUT/p) for p in ['analysis.json','raw_audit.json','rules.json','selected.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()})

def main(pid):
    smoke=BASE/'dinowm_policy_selection_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['states']==3 and a['selections']==15 and a['calibration_trials']==0;assert a['source_sha256']==sha256(resolve('scripts/audit_followup_dinowm_policy_selection.py'))
    files=['scripts/analyze_followup_dinowm_policy_selection.py','scripts/audit_followup_dinowm_policy_selection.py','scripts/run_followup_dinowm_policy_selection_pipeline.py'];OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in files},'smoke_audit_sha256':sha256(smoke/'raw_audit.json'),'bank_dependency_pid':pid}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    while not (BASE/'dinowm_policy_bank/pipeline_complete.json').exists():
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError('Bank parent ended before fullrawaudit; inspectoriginal, no restart')
        write_json(OUT/'status.json',{'pid':os.getpid(),'stage':'waiting_bank_audit','status':'running','verified_dependency_pid':pid,'updated_unix':time.time()});time.sleep(10)
    for stage in ['analyze','audit']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'running','updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve('scripts/'+stage+'_followup_dinowm_policy_selection.py'))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'status':'failed','stage':stage,'returncode':r.returncode,'unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    report();write_json(OUT/'pipeline_complete.json',{'states':108,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--bank-pid',type=int,required=True);args=ap.parse_args();main(args.bank_pid)
