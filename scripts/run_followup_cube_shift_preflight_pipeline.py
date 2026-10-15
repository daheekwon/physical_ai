"""Freeze verified smoke, audit full Cube condition validity and publish evidence."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_shift_preflight';read=lambda p:json.loads(p.read_text())
SOURCES=['configs/followup_cube_shift_preflight.json','scripts/cube_shift_preflight_core.py','scripts/run_followup_cube_shift_preflight.py','scripts/audit_followup_cube_shift_preflight.py','scripts/run_followup_cube_shift_preflight_pipeline.py']
def verify():
    for p,h in read(OUT/'pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p

def report():
    a=read(OUT/'raw_audit.json');r=read(OUT/'analysis.json');assert a['states']==48 and a['fresh_physical_branches']==672 and a['analysis_sha256']==sha256(OUT/'analysis.json')
    for p,h in {**read(OUT/'manifest.json')['inputs'],**a['inputs'],**a['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    cfg=read(OUT/'manifest.json')['config'];lines=['<!-- BEGIN CUBE SHIFT VALIDITY -->','## Cube의 질량·마찰 변경이 실제 반응을 바꾸는가?','',
    '적응 비교의 전제 조건으로, 기존12개 development family의 접근·파지·운반·해제 **48개 상태**에서 질량·마찰 변화를 검사했다. 동일한 원래8action을7조건×2반복으로 실행한 **672개 분기**를 독립적으로 모두 재실행했다. Smoothing이나 적응 성능을 측정한 결과는 아니다.','',
    '과거 history는 동일하게 유지하고 현재 root에서 객체 물성만 갑자기 변경했다. 객체 질량과 세 주축 관성은 함께0.5배/2배로 바꾸고, 마찰은 객체 geom의 sliding 계수만0.5배/2배로 바꿨다. 원래 질량은 약79.36g, sliding 계수는1.0이다. 다른 표면 계수와 gripper 명령은 그대로 유지했다. 결합 조건은 질량2배·객체 마찰0.5배다.','',
    '모델 상수는 scratch data로 다시 계산하고, 원래 root를 복원한 뒤 변경/sham 조건에서만 mj_forward를 호출했다. 물리 시간을 진행시키거나 qpos/qvel을 바꾸지 않았다. 원래 nominal은 이미 감사한 parent trace를 정확히 재현했고, 동일 조건의 두 실행도 모든 기록 물리량에서 정확히 같았다.','',
    f"Sham은 파라미터를 바꾸지 않고 같은 상수·forward 절차만 적용한 대조다. 최대 sham 위치 차이는 {r['maximum_sham_position_m']:.3g}m, full 회전 차이는 {r['maximum_sham_rotation_rad']:.3g}rad였다. 효과가 있다고 분류하려면 위치 또는 회전 차이가 각각 **1μm/10μrad의 수치 하한과 해당 상태 sham 차이의10배**를 모두 넘어야 한다. 이 기준은 물체 보존 허용 오차2mm·2도와 별개다.",'',
    '| 조건 | 반응 변화가 검출된 상태 | 평균 최대 위치 변화, mm | 평균 최대 회전 변화, deg |','| --- | --- | --- | --- |']
    for condition in cfg['conditions']:
        x=next(z for z in r['summary'] if z['condition']==condition['name'] and z['phase']=='all');lines.append(f"| {condition['name']} | {100*x['effective_object_change']['mean']:.2f}% | {1000*x['object_position_effect_m']['mean']:.4f} | {180/3.141592653589793*x['object_rotation_effect_rad']['mean']:.4f} |")
    lines+=['','| 조건 | 접근 | 파지 | 운반 | 해제 |','| --- | --- | --- | --- | --- |']
    for condition in cfg['conditions'][2:]:
        values=[next(z for z in r['summary'] if z['condition']==condition['name'] and z['phase']==phase)['effective_object_change']['mean'] for phase in ['approach','grasp','transport','release']];lines.append('| '+condition['name']+' | '+' | '.join(f'{100*v:.2f}%' for v in values)+' |')
    lines+=['',
    '마찰 설정값의 변화와 실제 접촉 반응의 변화는 구분해야 한다. 접촉 상대의 마찰과 접촉 상태에 따라 객체 마찰만 낮춘 변경은 현재 window에서 null일 수 있다. 무변화 사례를 제거하지 않고 phase별 지원 범위로 보고하며, 향후 적응 성공의 근거로 세지 않는다. 여기서 변화가 검출되었다고 수정 허용량이나 적응 규칙이 검증된 것은 아니다.','',
    '다음 단계에서는 같은 조건에서 각 방법의 자체 nominal을 기준으로 수정 보존과 jerk를 측정하고, 고정 모델과 과거 관측 갱신 모델 및 allowance/계산량 적응을 비교해야 한다. 이 preflight의 root 즉시 변화와 episode 시작부터 바뀐 조건은 서로 다른 설계이므로 구분해 잠근다.','',
    '근거: [조건·프로토콜](../artifacts/followup_v1/cube_shift_preflight/manifest.json), [전체 분석·family 구간](../artifacts/followup_v1/cube_shift_preflight/analysis.json), [독립 원시 감사](../artifacts/followup_v1/cube_shift_preflight/raw_audit.json), [상태별 변화와 null 대조](../artifacts/followup_v1/cube_shift_preflight/effects.csv).','<!-- END CUBE SHIFT VALIDITY -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN CUBE CAUSAL ARX -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['manifest.json','analysis.json','raw_audit.json','effects.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()})

def main():
    smoke=BASE/'cube_shift_preflight_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['states']==4 and a['fresh_physical_branches']==56 and a['source_sha256']==sha256(resolve(SOURCES[3]));assert a['analysis_sha256']==sha256(smoke/'analysis.json')
    for p,h in {**read(smoke/'manifest.json')['inputs'],**a['inputs'],**a['audit_helpers']}.items():assert sha256(resolve(p))==h,p
    OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'smoke_raw_audit_sha256':sha256(smoke/'raw_audit.json'),'scope':'Parameter validity preflight only; no adaptation/smoothing performance claim.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for stage,script in [('collect',SOURCES[2]),('audit',SOURCES[3])]:
        verify();write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve(script))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'status':'failed','stage':stage,'returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    verify();report();write_json(OUT/'pipeline_complete.json',{'states':48,'families':12,'branches':672,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
