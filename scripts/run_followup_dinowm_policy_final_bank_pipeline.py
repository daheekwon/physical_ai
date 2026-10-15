"""Finish old-family smoke, wait for locked test nominal audit, collect final corrections."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_policy_final_bank';FINAL=BASE/'dinowm_policy_final';read=lambda p:json.loads(p.read_text())
SOURCES=['scripts/run_followup_dinowm_policy_final_bank.py','scripts/audit_followup_dinowm_policy_final_bank.py','scripts/analyze_followup_dinowm_policy_final.py','scripts/audit_followup_dinowm_policy_final.py','scripts/run_followup_dinowm_policy_final_bank_pipeline.py']

def verify():
    for p,h in read(FINAL/'final_protocol.json')['inputs'].items():assert sha256(resolve(p))==h,p
    if (OUT/'pipeline_lock.json').exists():
        for p,h in read(OUT/'pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p

def run_stage(script,logpath,args=()):
    verify()
    with logpath.open('a') as log:r=subprocess.run([sys.executable,str(resolve(script)),*args],stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:write_json(OUT/'status.json',{'status':'failed','stage':script,'returncode':r.returncode,'pid':os.getpid(),'updated_unix':time.time()});raise RuntimeError((script,r.returncode))
    print('COMPLETE',script,*args,flush=True)

def report():
    a=read(FINAL/'raw_audit.json');r=read(FINAL/'analysis.json');bank=read(OUT/'raw_audit.json');assert a['states']==192 and a['families']==64 and a['selections']==960 and a['analysis_sha256']==sha256(FINAL/'analysis.json');cfg=read(FINAL/'final_protocol.json')['config'];p=next(x for x in r['paired'] if (x['method'],x['baseline'],x['metric'])==(cfg['primary_method'],cfg['primary_baseline'],cfg['primary_metric']));lo,hi=p['ci95'];pct=lambda x:'미정의' if x is None else f'{100*x:.2f}%';verdict='사전 지정한 경험적 실질 개선 조건을 충족했다.' if r['supported_practical_primary_advantage'] else '사전 지정한 실질적 추가 우위 조건을 충족하지 못했다. 동등성이나 다른 조건에서의 무용성을 뜻하지 않는다.'
    lines=['<!-- BEGIN DINO POLICY INDEPENDENT FINAL -->','## 새 정책64개 family의 독립 단발 correction 평가','',f"Frozen Diffusion Policy의 새64개 초기 family에서 사전 지정한16·96·192 시점, **192개 상태·{bank['candidates']}개 후보·960개 선택**을 실행하고 감사했다. 후보 생성·모델·보정 규칙·family·주 비교는 test nominal 실행 전에 잠갔다. Test에서 threshold·representation·radius를 다시 선택하지 않았다.",'',f"주 비교인 **DINO patch RMS−LeWM의 U 차이는 {100*p['mean']:+.2f}%p**, paired family95% 구간 [{100*lo:+.2f},{100*hi:+.2f}]%p다. {verdict} 사전 기준은 U10%p 이상, 구간 하한>0, 경험적 선택 후 위반율10% 이하, 수정 적용25% 이상이다.",'','| 방법 | 물체 보존·jerk 동시 달성 U | 수정 적용 비율 | 선택 후 위반율 | 평균 실제 jerk 감소 |','| --- | --- | --- | --- | --- |']
    for x in r['summary']:lines.append(f"| {x['method']} | {pct(x['joint']['mean'])} | {pct(x['active']['mean'])} | {pct(x['selected_risk']['mean'])} | {pct(x['jerk_gain']['mean'])} |")
    lines+=['','모든 방법은 동일한 비-WM 다항식 후보와 예상 실행 jerk 순위를 공유한다. 주 보존 기준은 H5＋동일 후속3행동의 시간 정렬된 물체 위치2px·회전2도와 두 종료점 coverage 감소0.02다. U에는 실제 pusher jerk10% 감소 및 실제 수정 적용도 필요하다. Nominal 성공 여부로 표본을 제외하지 않았다.','', '모든 선택은 modified physics를 실행하기 전에 저장했다. 전체 후보의 dense 물리 오차·jerk·독립 coverage·입력 history·action 제약을 확인하고 모든 LeWM 점수를 nativeCPU로 다시 계산했다. DINO는 각 상태의 nominal/마지막 후보를 새 native forward로 확인했다. 같은 두 후보의 물리 분기도 다시 실행했다. DINO는 검증한 third-party retrain이며 원본 공개 모델의 재현 결과로 부르지 않는다.','', '분석 감사는960개 선택의 기준과 물리 지표, family bootstrap 구간 및 주 비교 결정을 독립 재계산했다. 보조 비교는 탐색적이며 많은 후보를 독립 family로 세지 않는다. 낮은 경험적 위반율은 모집단 위험 인증이 아니다. 모델 사전학습 미노출까지 증명한 결과도 아니다.','', '이 실험은 원래 정책 실행에서 한 번 correction을 넣고 원래 suffix를 실행한 평가다. 이후 자기 관측으로 재계획하는 반복 정책, 조건 변화·적응, 실제 지연 및 계산 비용을 맞춘 비교는 별도 단계다.','', '근거: [최종 방법 잠금](../artifacts/followup_v1/dinowm_policy_final/final_protocol.json), [모든 결과와 paired 구간](../artifacts/followup_v1/dinowm_policy_final/analysis.json), [전체 후보 원시 감사](../artifacts/followup_v1/dinowm_policy_final_bank/raw_audit.json), [선택·통계 독립 감사](../artifacts/followup_v1/dinowm_policy_final/raw_audit.json).','<!-- END DINO POLICY INDEPENDENT FINAL -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN DINO POLICY FINAL NOMINAL -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(FINAL/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{p:sha256(FINAL/p) for p in ['analysis.json','raw_audit.json','final_protocol.json','selected.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()})

def main(nominal_pid,smoke_pid):
    OUT.mkdir(exist_ok=True);smoke=BASE/'dinowm_policy_final_bank_smoke'
    while not (smoke/'raw_audit.json').exists():
        try:os.kill(smoke_pid,0)
        except ProcessLookupError:raise RuntimeError('Smoke audit ended without raw audit; inspect original, no restart')
        write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':'waiting_smoke_audit','verified_dependency_pid':smoke_pid,'updated_unix':time.time()});time.sleep(10)
    a=read(smoke/'raw_audit.json');assert a['smoke'] and a['states']==3 and a['fresh_native_DINO_predictions']==a['candidates']==165 and a['fresh_physical_branches']==6;assert a['source_sha256']==sha256(resolve(SOURCES[1]))
    for p,h in {**read(smoke/'manifest.json')['inputs'],**a['inputs'],**a['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    for script in [SOURCES[2],SOURCES[3]]:run_stage(script,OUT/'smoke_analysis.log',['--smoke'])
    a=read(BASE/'dinowm_policy_final_analysis_smoke/raw_audit.json');assert a['smoke'] and a['states']==3 and a['selections']==15
    lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'smoke_bank_audit_sha256':sha256(smoke/'raw_audit.json'),'smoke_selected_audit_sha256':sha256(BASE/'dinowm_policy_final_analysis_smoke/raw_audit.json'),'final_method_protocol_sha256':sha256(FINAL/'final_protocol.json'),'nominal_dependency_pid':nominal_pid,'scope':'New execution/statistics wrapper passed oldfamily smoke and frozen before any final correction physics; model/rules/initialfamilies already locked before testnominal.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    gate=BASE/'dinowm_policy_final_nominal/pipeline_complete.json'
    while not gate.exists():
        try:os.kill(nominal_pid,0)
        except ProcessLookupError:raise RuntimeError('Final nominal ended without complete audit; inspect original, no restart')
        write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':'waiting_final_nominal_audit','verified_dependency_pid':nominal_pid,'updated_unix':time.time()});time.sleep(10)
    assert read(gate)['raw_audit_sha256']==sha256(BASE/'dinowm_policy_final_nominal/raw_audit.json')
    for stage,script in [('collect_bank',SOURCES[0]),('audit_bank',SOURCES[1]),('analyze',SOURCES[2]),('audit_selection',SOURCES[3])]:
        write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()});run_stage(script,OUT/(stage+'.log'))
    verify();report();write_json(OUT/'pipeline_complete.json',{'states':192,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()});write_json(FINAL/'pipeline_complete.json',{'states':192,'families':64,'selections':960,'raw_audit_sha256':sha256(FINAL/'raw_audit.json'),'report_evidence_sha256':sha256(FINAL/'report_evidence.json'),'final_protocol_sha256':sha256(FINAL/'final_protocol.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--nominal-pid',type=int,required=True);ap.add_argument('--smoke-pid',type=int,required=True);args=ap.parse_args();main(args.nominal_pid,args.smoke_pid)
