"""Complete matched-history Cube response-update preparation and diagnosis."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_shift_history';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_shift_history.json','scripts/cube_shift_history_core.py','scripts/run_followup_cube_shift_history.py','scripts/audit_followup_cube_shift_history.py','scripts/run_followup_cube_shift_history_pipeline.py']
def verify():
    for p,h in read(OUT/'pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p

def report():
    a=read(OUT/'raw_audit.json');r=read(OUT/'analysis.json');manifest=read(OUT/'manifest.json');assert a['condition_contexts']==336 and a['fresh_history_RGB']==1008 and a['fresh_physics_samples']==151200 and a['forecast_rows']==672 and a['analysis_sha256']==sha256(OUT/'analysis.json')
    for p,h in {**manifest['inputs'],**a['inputs'],**a['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    lines=['<!-- BEGIN CUBE SHIFT HISTORY -->','## Cube: 물성 변화 후 관측으로 반응 계수를 갱신하기','',
    '적응 실험의 입력과 반응 예측을 검증하기 위해 기존12개 development family·48개 phase에서 **7조건·336개 실제 history**를 만들었다. 물성을 correction root의10action(0.5초) 전에 바꾸고, 원래 행동을 그대로 실행하면서 이후의 robot 관측·목표 명령·이미지를 기록했다. 모든 history의250개 물리 표본과 nominal 미래8action의200개 표본, 세 RGB를 다시 실행해 **151,200개 물리 표본·1,008개 RGB**를 확인했다.','',
    '미래8action을 실행하기 전에 과거 입력을 확정했다. 다음 두 모델은 실제 과거 EEF와 이미 전달된 목표 명령만 사용한다. **변경 전 계수 유지**는 변경 직전 관측에서 맞춘 PD 계수를 그대로 사용하고, **변경 후 계수 갱신**은 변경 이후 관측으로 다시 맞춘다. 두 모델의 상태별 피팅 표본 수는 같으며 최대250개다. 이른 접근 phase에서는 변경 이전 history가 짧아125개를 사용한다. 모델의 초기 현재 위치·속도와 예측 행동은 동일하다.','',
    '| 조건 | 계수 유지 / 갱신 EEF 위치 RMSE, mm | 갱신의 위치 오차 감소, mm (family95% 구간) |','| --- | --- | --- |']
    for condition in manifest['conditions']:
        name=condition['name'];old=next(z for z in r['summary'] if z['condition']==name and z['model']=='frozen_prechange');new=next(z for z in r['summary'] if z['condition']==name and z['model']=='updated_postchange');delta=next(z for z in r['paired'] if z['condition']==name and z['metric']=='eef_position_RMSE_m');lo,hi=delta['ci95'];lines.append(f"| {name} | {1000*old['eef_position_RMSE_m']['mean']:.3f} / {1000*new['eef_position_RMSE_m']['mean']:.3f} | {1000*delta['mean']:+.3f} [{1000*lo:+.3f}, {1000*hi:+.3f}] |")
    lines+=['',
    '계수를 갱신하면 물성이 변하지 않아도 시간·접촉 phase가 진행되면서 예측이 달라질 수 있다. 이를 구분하기 위해 **조건별 갱신 이득에서 동일 절차 sham의 갱신 이득을 뺀 paired 비교**도 제공한다. 원래 no-forward nominal은 원본 재현 fixture이며, 물성 변화의 공정한 기준은 같은 forward 절차를 사용한 sham이다. 위치 오차 개선이 실제 jerk 감소나 물체 궤적 보존 개선을 뜻하지 않는다. 예상 jerk의 상대 오차와 모든 상태별 결과도 원본에 포함했다.','',
    'History 수집 smoke에서 원래 물리는 같지만 RGB 일부가 달라지는 문제를 발견했다. mj_setConst가 XML의 장면 중심·범위·시각 크기 통계까지 다시 계산한 것이 원인이었다. 원래 center/extent/meansize를 보존하고 물성 관련 meanmass/meaninertia 계산은 유지한 뒤, 원래 조건의 물리와 RGB가 모두 정확히 재현되는 것을 확인했다. 실패한 최초 실행은 별도로 보존했다. 물리 전용 앞선672개 분기 결과를 변경한 것은 아니다.','',
    '이 단계는 **변경된 관측 history 및 국소 반응 예측**의 검증이다. 다항식 correction 선택, 물체 보존–jerk 동시 달성, allowance/계산량 적응의 네 조건, 독립 최종 test를 완료했다는 뜻은 아니다. 다음 선택기는 이 causal history를 입력으로 사용하고, 같은 변경 조건의 nominal을 평가 기준으로 삼아야 한다. 원래 source phase는 조건 결과에 따라 다시 고르지 않았다.','',
    '근거: [입력 프로토콜](../artifacts/followup_v1/cube_shift_history/manifest.json), [반응 예측·paired 구간](../artifacts/followup_v1/cube_shift_history/analysis.json), [전체 history·미래 trace 감사](../artifacts/followup_v1/cube_shift_history/raw_audit.json), [개별 오차](../artifacts/followup_v1/cube_shift_history/response_fidelity.csv), [초기 렌더링 문제와 해결 기록](../artifacts/followup_v1/cube_shift_history_smoke_failure_v0/diagnosis.json).','<!-- END CUBE SHIFT HISTORY -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN CUBE SHIFT VALIDITY -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['manifest.json','analysis.json','raw_audit.json','response_fidelity.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()})

def main():
    smoke=BASE/'cube_shift_history_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['states']==4 and a['condition_contexts']==28 and a['forecast_rows']==56 and a['source_sha256']==sha256(resolve(SOURCES[3])) and a['analysis_sha256']==sha256(smoke/'analysis.json')
    for p,h in {**read(smoke/'manifest.json')['inputs'],**a['inputs'],**a['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'smoke_raw_audit_sha256':sha256(smoke/'raw_audit.json'),'scope':'Causal changedhistory/responseupdate diagnosis; future correction/adaptation study separate.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for stage,script in [('collect',SOURCES[2]),('audit',SOURCES[3])]:
        verify();write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'status':'failed','stage':stage,'returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    verify();report();write_json(OUT/'pipeline_complete.json',{'states':48,'families':12,'condition_contexts':336,'forecast_rows':672,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
