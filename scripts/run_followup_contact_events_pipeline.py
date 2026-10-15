"""Freeze complete event-role smoke, run120windows, independently audit and publish."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'contact_events';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_contact_events.json','scripts/contact_event_core.py','scripts/prepare_followup_contact_events.py','scripts/audit_followup_contact_catalogue.py','scripts/run_followup_contact_events.py','scripts/audit_followup_contact_events.py','scripts/analyze_followup_contact_events.py','scripts/audit_followup_contact_event_analysis.py','scripts/run_followup_contact_events_pipeline.py']

def verify():
    lock=read(OUT/'pipeline_lock.json')
    for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
    return lock

def report():
    r=read(OUT/'analysis.json');a=read(OUT/'analysis_raw_audit.json');raw=read(OUT/'raw_audit.json');assert r['states']==a['states']==raw['states']==120 and r['families']==12 and a['analysis_sha256']==sha256(OUT/'analysis.json');pct=lambda v:'미정의' if v is None else f'{100*v:.2f}%';roles=['pre','transition','post','stable_free','stable_contact'];physical={(x['event_type'],x['role']):x for x in r['physical_summary']};selected={(x['event_type'],x['role'],x['method']):x for x in r['selection_summary']};effects={(x['event_type'],x['role'],x['representation'],x['method']):x for x in r['effect_summary']}
    lines=['<!-- BEGIN CONTACT EVENT LOCALIZATION -->','## 접촉 시작·해제 전후의 물리 자유도와 LeWM 효과 예측','',f"이미 노출된 정책 development12개 family에서 각 family의 첫 유효 접촉 시작·해제 시점을 정했다. 각 event의 correction 직전·전환 포함·직후와 안정된 free/contact 대조를 구성해 **120개 고유 상태·{r['candidates']}개 후보·{r['selection_rows']}개 선택**을 평가했다. Nominal 접촉 기록만으로 시점을 잠갔으며 수정 성공·실패 결과로 고르지 않았다.",'', '접촉은100Hz 충돌 기록을10Hz 제어 구간별 존재 여부로 묶고, 양쪽 두 제어 구간의 상태가 일정한 전환만 사용했다. H5 시작 시점은 event 제어 index의−5·−2·＋1이고 후속3행동은 동일하게 실행했다. 따라서 pre라는 이름은 correction 구간의 위치를 뜻하며 suffix가 전환을 포함할 수 있다. 다른 전환도 같은 window에 있을 수 있다. 대조는 같은 family의8제어 구간 동안 action-level contact 상태가 일정하고 event 주변과 겹치지 않는 window 중 nominal pusher 이동 RMS가 가장 가까운 것을 사용했다. 미세 물리 시점의 contact chatter까지 없다는 뜻은 아니다.','', '| Event | 구간 | 유용한 후보 존재 비율 | 제어기 U | LeWM U | 접촉 모델 U | LeWM 선택 후 위반율 |','| --- | --- | --- | --- | --- | --- | --- |']
    for event in ['onset','release']:
        for role in roles:
            x=physical[event,role];cs=[selected[event,role,m] for m in ['controller','lewm_global','contact']];lines.append(f"| {event} | {role} | {pct(x['has_useful_candidate']['mean'])} | {pct(cs[0]['joint']['mean'])} | {pct(cs[1]['joint']['mean'])} | {pct(cs[2]['joint']['mean'])} | {pct(cs[1]['selected_risk']['mean'])} |")
    lines+=['', '각 셀은12개 family의12개 시점이다. 유용한 후보 존재 비율은 같은 유한 후보 중 물체 궤적 보존과 실제 jerk10% 감소를 함께 달성한 것이 하나라도 있는 상태의 비율이며, 실행 전에 그것을 고를 수 있다는 뜻은 아니다. 선택기는 앞선 별도 정책 calibration 규칙을 그대로 사용했다. Event label이나 미래 물체 상태는 선택기에 제공하지 않았다.','', '| Event | 구간 | Frame 실제/예측/zero 효과 오차(px) | History 실제/예측/zero 효과 오차(px) | Frame/History 해상도 초과 효과 비율 |','| --- | --- | --- | --- | --- |']
    for event in ['onset','release']:
        for role in ['pre','transition','post']:
            vals=[];support=[]
            for rep in ['lewm_frame','lewm_history']:
                vals.append(' / '.join(f"{effects[event,role,rep,m]['position_effect_error']['mean']:.3f}" for m in ['real_encoded','predicted','zero_effect']));support.append(pct(effects[event,role,rep,'predicted']['position_resolvable_fraction']['mean']))
            lines.append(f"| {event} | {role} | {vals[0]} | {vals[1]} | {' / '.join(support)} |")
    rr=r['resolution'];lines+=['',f"두 고정 ridge readout의 실제 효과 측정 해상도는 기존 expert calibration105개 상태의 상태별 동일 가중90분위수다. Frame은 위치{rr['lewm_frame']['position']:.3f}px·회전{rr['lewm_frame']['rotation']*180/3.141592653589793:.3f}도, History는 위치{rr['lewm_history']['position']:.3f}px·회전{rr['lewm_history']['rotation']*180/3.141592653589793:.3f}도다. 이 해상도는 새 정책 event 결과에 맞춰 보정하지 않은 전이 기준이며 정책 분포에서의 위험 보증이 아니다.",'', '같은 고정 probe로 실제 encoded endpoint와 예측 endpoint를 읽었다. Current-state capture, 회전 효과, zero/persistence/크기·방향을 맞춘 wrong-action, 측정 가능한 효과 비율과 방향 평가 지원 비율, event 내부 paired 차이와 family 구간은 원본에 제공한다. 해상도 이하 사례는 방향 해석에서 구별하되 물리 보존·위반 통계에는 모두 유지했다. 실제 효과를 읽지 못하는 범위에서 예측 오류를 dynamics만의 문제로 분류하지 않는다.','', '주 보존 기준은 동일 물리 시각의 위치2px·회전2도와 수정/suffix 종료 coverage 감소0.02이고 pusher jerk는100Hz trace와 같은 경계 stencil로 계산했다. Event 시각으로 궤적을 시간 이동해 오차를 줄이지 않았다. 접촉 mask 불일치·존재 변화·window 내 첫 접촉 시각 차이도 물리 trace에서 계산했다. 시작 시 이미 접촉 중이면 첫 기록은 새 접촉 시작 시각이 아니므로 해당 의미를 구분한다.','', '모든 native LeWM 미래 latent와 실제 endpoint/history encoding을 새로 계산하고, 상태별 nominal/마지막 후보 물리 및 접촉 trace를 다시 실행했다. Probe 추론·효과 기하·선택 지표·family 구간도 독립 재계산했다. 이 결과는 같은 family 안의 개발 진단이며 배포 성능이나 새로운 독립 test가 아니다. 고정 probe·승인 규칙의 전이와 현재 candidate family 범위에서 해석한다.','', '근거: [event 선정·지원 범위](../artifacts/followup_v1/contact_events_catalogue/manifest.json), [선정·해상도 독립 감사](../artifacts/followup_v1/contact_events_catalogue/raw_audit.json), [전체 분석·paired 구간](../artifacts/followup_v1/contact_events/analysis.json), [물리·native 모델 감사](../artifacts/followup_v1/contact_events/raw_audit.json), [효과·통계 독립 감사](../artifacts/followup_v1/contact_events/analysis_raw_audit.json).','<!-- END CONTACT EVENT LOCALIZATION -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN PUSHT NUMERICAL SENSITIVITY -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{p:sha256(OUT/p) for p in ['analysis.json','analysis_raw_audit.json','raw_audit.json','protocol.json','selected.csv','state_effects.csv','physical_states.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()})

def main():
    smoke=BASE/'contact_events_smoke';a=read(smoke/'analysis_raw_audit.json');raw=read(smoke/'raw_audit.json');assert a['states']==10 and a['cases']==10 and a['selected_rows']==30 and raw['fresh_physical_branches']==20 and raw['fresh_native_LeWM_predictions']==raw['candidates'];assert a['source_sha256']==sha256(resolve('scripts/audit_followup_contact_event_analysis.py')) and raw['source_sha256']==sha256(resolve('scripts/audit_followup_contact_events.py'));assert a['analysis_sha256']==sha256(smoke/'analysis.json');assert len(read(smoke/'analysis.json')['paired_phase_errors'])==64
    for p,h in {**read(smoke/'protocol.json')['inputs'],**read(smoke/'analysis_complete.json')['inputs'],**a['helpers'],**raw['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'smoke_raw_audit_sha256':sha256(smoke/'raw_audit.json'),'smoke_analysis_audit_sha256':sha256(smoke/'analysis_raw_audit.json'),'catalogue_audit_sha256':sha256(BASE/'contact_events_catalogue/raw_audit.json'),'scope':'120predeclared development event windows; full firstfamily10role smoke audited before fullcollection.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for stage,script in [('collect','scripts/run_followup_contact_events.py'),('audit_raw','scripts/audit_followup_contact_events.py'),('analyze','scripts/analyze_followup_contact_events.py'),('audit_effects','scripts/audit_followup_contact_event_analysis.py')]:
        verify();write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'status':'failed','stage':stage,'returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    verify();report();a=read(OUT/'analysis_raw_audit.json');write_json(OUT/'pipeline_complete.json',{'states':120,'families':12,'candidates':read(OUT/'raw_audit.json')['candidates'],'effect_rows':a['effect_rows'],'selected_rows':a['selected_rows'],'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'analysis_raw_audit_sha256':sha256(OUT/'analysis_raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
