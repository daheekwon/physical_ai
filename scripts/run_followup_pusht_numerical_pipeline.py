"""Freeze a checked numerical diagnostic, collect all8states, audit and report."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'pusht_numerical';read=lambda p:json.loads(p.read_text())

def report():
    a=read(OUT/'raw_audit.json');r=read(OUT/'analysis.json');assert a['states']==r['states']==8 and a['analysis_sha256']==sha256(OUT/'analysis.json') and a['exact_100Hz_repeats_and_parent']
    lines=['<!-- BEGIN PUSHT NUMERICAL SENSITIVITY -->','## 物理 적분 간격에 따른 보존·jerk 판정의 민감도'.replace('物理','물리'),'',f"새 정책 development bank에서 nominal free/contact 각각4개 상태, **8개 상태·{r['families']}개 family·{r['candidates']}개 후보**를 사전 규칙으로 선정했다. 접촉 phase 안에서 정렬된 family별 첫 window를 사용했다. 성공·실패 결과로 표본을 고르지 않았다. 각 후보를100Hz 두 번 및200·400Hz에서 실행한 **{r['fresh_collection_branches']}개 물리 분기**와 감사의 추가{r['fresh_audit_branches']}개 분기를 확인했다.",'', '100Hz 두 반복은 모든 물리 상태·접촉 기록·coverage에서 서로 일치했고 기존 후보 bank와도 정확히 일치했다. 적분률은 correction 시작에서만 바꾸고, prefix를 재생해 동일한 전체 root 상태에서 분기했다. 제어율10Hz, PD gain100/20, 상대 행동, H5와 후속3행동의0.8초 길이는 고정했다. 따라서 이는 국소 적분 민감도이며 episode 전체의 적분 수렴 검사가 아니다.','', '| Nominal phase | 적분률 | 활성 후보 보존 판정 변경률 | 유용 후보 존재 상태 | Jerk 순위 상관 평균 | Nominal 물체 위치 최대 변화 |','| --- | --- | --- | --- | --- | --- |']
    for x in r['summary']:
        rho=x['jerk_gain_rank_spearman']['mean'];rho='미정의' if rho is None else f'{rho:.3f}'
        lines.append(f"| {x['phase']} | {x['rate']}Hz | {100*x['preservation_flip_fraction']['mean']:.2f}% | {x['states_with_useful_candidate']}/{x['states']} | {rho} | {x['nominal_position_change_px']['max']:.4f}px |")
    lines+=['', '보존 판정 변경은 각 적분률에서 다시 실행한 nominal 대비2px·2도 및 수정/suffix 종료 coverage 감소0.02를 적용한 결과를100Hz 판정과 비교한 것이다. Jerk는 모든 적분률을 동일한100Hz 시각에 맞추고 같은 prefix·차분 stencil로 계산했다. 더 촘촘한 물리 시점에서만 검출되는 위반, jerk 개선 순위, 접촉 시작 시각 및 존재 변화는 원본 표에 따로 제공한다. 유용 후보는 보존과 실제 jerk10% 감소를 함께 만족해야 한다.','', '적분 간격은 접촉 solver뿐 아니라 PD 수치 적분에도 영향을 준다. 같은 상대 명령이라도 이후 pusher 위치가 달라지면 실제 절대 목표가 달라질 수 있다. 높은 적분률을 자동으로 정답으로 보거나, 이 소규모 진단만으로 기존 모델 비교의 우열을 바꾸지 않는다. 후보들은 독립 표본이 아니며 같은 family의 두 phase도 독립 family로 세지 않는다. 기존 test·허용 오차·선택 규칙은 변경하지 않았다.','', '근거: [고정 표본·설정](../artifacts/followup_v1/pusht_numerical/protocol.json), [전체 분석](../artifacts/followup_v1/pusht_numerical/analysis.json), [후보별 오차·jerk](../artifacts/followup_v1/pusht_numerical/candidate_metrics.csv), [독립 감사](../artifacts/followup_v1/pusht_numerical/raw_audit.json).','<!-- END PUSHT NUMERICAL SENSITIVITY -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN PUSHT NATIVE CPU CALIBRATION -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{p:sha256(OUT/p) for p in ['analysis.json','raw_audit.json','protocol.json','candidate_metrics.csv','state_metrics.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()})

def main():
    smoke=BASE/'pusht_numerical_smoke';a=read(smoke/'raw_audit.json');assert a['states']==1 and a['candidates']==55 and a['fresh_physical_branches']==6 and a['exact_100Hz_repeats_and_parent'];assert a['source_sha256']==sha256(resolve('scripts/audit_followup_pusht_numerical.py'))
    for p,h in {**read(smoke/'protocol.json')['inputs'],**a['inputs'],**a['helpers']}.items():assert sha256(resolve(p))==h,p
    files=['configs/followup_pusht_numerical.json','scripts/run_followup_pusht_numerical.py','scripts/audit_followup_pusht_numerical.py','scripts/run_followup_pusht_numerical_pipeline.py'];OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in files},'smoke_raw_audit_sha256':sha256(smoke/'raw_audit.json')};write_json(OUT/'pipeline_lock.json',lock)
    for stage in ['run','audit']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'running','updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('w') as log:r=subprocess.run([sys.executable,str(resolve('scripts/'+stage+'_followup_pusht_numerical.py'))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'failed','returncode':r.returncode});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    report();write_json(OUT/'pipeline_complete.json',{'states':8,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
