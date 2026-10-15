"""Locked Cube policy DINO score, audit, selection and report pipeline."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_cube_dino_policy import HELPERS
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_dino_policy_scores';SEL=BASE/'cube_dino_policy_selection';read=lambda p:json.loads(p.read_text())
SOURCES=['scripts/run_followup_cube_dino_policy_scores.py','scripts/analyze_followup_cube_dino_policy.py','scripts/run_followup_cube_dino_policy_pipeline.py','scripts/model_runtime_imports.py','scripts/retrained_dinowm_core.py',*HELPERS]
def verify(inputs):
    for p,h in inputs.items():assert sha256(resolve(p))==h,p

def report():
    a=read(SEL/'raw_audit.json');r=read(SEL/'analysis.json');scores=read(OUT/'raw_audit.json');assert a['states']==r['states']==scores['states']==138 and a['candidates']==scores['candidates']==2698 and a['selections']==690;verify(read(SEL/'protocol.json')['inputs']);verify(a['helpers']);assert a['analysis_sha256']==sha256(SEL/'analysis.json');pct=lambda v:'미정의' if v is None else f'{100*v:.2f}%'
    lines=['<!-- BEGIN CUBE DINO POLICY COMPARISON -->','## Cube 정책의 동일 후보에서 DINO-WM verifier 비교','', 'Cube의24개 calibration family와 기존12개 development family에서 같은138개 상태·2,698개 다항식 후보에 DINO-WM의 patch RMS 및 patch 평균 거리 승인을 추가했다. 기존 controller/contact/LeWM의 규칙과 선택을 그대로 유지하고, DINO의 두 threshold는 같은41분위수 탐색과 위험·적용 조건으로 calibration family에서만 정했다. 새 모델은 별도 제3자 재학습 no-proprioception checkpoint이며 원래 공개 모델과 동일한 자산이라고 부르지 않는다.','', '| Development 방법 | U | 수정 적용 | 선택 후 위반율 | 평균 실제 jerk 감소 |','| --- | --- | --- | --- | --- |']
    for x in r['summary']:
        if x['split']=='development' and x['phase']=='all':lines.append(f"| {x['method']} | {pct(x['joint']['mean'])} | {pct(x['active']['mean'])} | {pct(x['selected_risk']['mean'])} | {pct(x['jerk_gain']['mean'])} |")
    lines+=['','U는 동일 root에서 원래 H5＋후속3명령의 물체 움직임과 비교해 모든2ms 표본에서 위치2mm·전체3D 회전2도를 지키면서 실제 EEF XYZ jerk를10% 이상 줄이는 비율이다. 최종 pose만 같은 것은 궤적 보존이 아니다. 모든 방법이 같은 후보와 예상 jerk 순위를 사용하며 gripper 명령은 고정했다.','']
    for metric in ['dino_patch_rms','dino_patch_mean']:
        x=next(x for x in r['paired'] if x['split']=='development' and x['method']==metric and x['baseline']=='lewm_global' and x['metric']=='joint');lo,hi=x['ci95'];lines.append(f"{metric}−LeWM의 U 차이는{100*x['mean']:+.2f}%p, paired family95% 구간은[{100*lo:+.2f},{100*hi:+.2f}]%p다.")
    lines+=['',f"추가한 DINO 예측은 이미지3장·과거10명령·후보5명령만 입력받는다. 기존 expert fixture의 encoding과 nominal/마지막 후보 예측을 정확히 재현했고, 모든 상태의 history 및{scores['fresh_native_candidate_forwards']}개 후보를 독립 normalization·native predictor 호출로 다시 확인했다. 모든 후보의 두 native 거리, DINO calibration grid, 선택 및 family 통계도 별도 계산으로 대조했다.",'','물리 결과는 감사된 공통 후보 bank를 hash 검증 후 재사용했다. 이번 단계에서 새 물리 분기를 실행하지 않았다. Nominal plan은 privileged 초기 상태에 기반하며 DINO의 no-proprioception과 정책 자체의 정보 접근을 혼동하지 않는다. 이 비교는 기존 development에 대한 확장이므로 새로운 독립 최종 평가, 반복 episode 보존, 조건 적응 또는 전체 추론 비용의 근거가 아니다. 이미 다른 방법의 결과를 확인한64개 final family를 새 독립 test로 재명명하지 않는다.','', '근거: [예측 감사](../artifacts/followup_v1/cube_dino_policy_scores/raw_audit.json), [고정 입력](../artifacts/followup_v1/cube_dino_policy_selection/protocol.json), [보정 규칙](../artifacts/followup_v1/cube_dino_policy_selection/rules.json), [성능과 paired 구간](../artifacts/followup_v1/cube_dino_policy_selection/analysis.json), [선택·통계 감사](../artifacts/followup_v1/cube_dino_policy_selection/raw_audit.json).','<!-- END CUBE DINO POLICY COMPARISON -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=doc.read_text();assert lines[0] not in text;anchor='<!-- BEGIN CUBE POLICY INDEPENDENT FINAL -->';assert anchor in text;doc.write_text(text.replace(anchor,section+'\n\n'+anchor,1));write_json(SEL/'report_evidence.json',{'inputs':{n:sha256(SEL/n) for n in ['protocol.json','raw_audit.json','analysis.json','selected.csv','rules.json']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__))})

def main():
    smoke=BASE/'cube_dino_policy_scores_smoke';sa=read(smoke/'raw_audit.json');ss=read(BASE/'cube_dino_policy_selection_smoke/raw_audit.json');assert sa['smoke'] and ss['smoke'] and sa['states']==ss['states']==8 and sa['candidates']==sa['fresh_native_candidate_forwards']==163 and ss['selections']==40;assert sa['source_sha256']==sha256(resolve(SOURCES[0])) and ss['source_sha256']==sha256(resolve(SOURCES[1]));verify(read(smoke/'manifest.json')['inputs']);verify(read(BASE/'cube_dino_policy_selection_smoke/protocol.json')['inputs']);OUT.mkdir(exist_ok=True)
    lock={'inputs':{p:sha256(resolve(p)) for p in SOURCES},'smoke_prediction_audit_sha256':sha256(smoke/'raw_audit.json'),'smoke_selection_audit_sha256':sha256(BASE/'cube_dino_policy_selection_smoke/raw_audit.json')}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    jobs=[('collect',[SOURCES[0],'collect']),('audit',[SOURCES[0],'audit']),('analyze_audit',[SOURCES[1]])]
    for stage,args in jobs:
        verify(lock['inputs']);write_json(OUT/'status.json',{'pid':os.getpid(),'status':'running','stage':stage,'updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,*[str(resolve(args[0])),*args[1:]]],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'status':'failed','stage':stage,'returncode':r.returncode});raise RuntimeError((stage,r.returncode))
        print('COMPLETE CUBE DINO',stage,flush=True)
    verify(lock['inputs']);report();done={'scores_raw_audit_sha256':sha256(OUT/'raw_audit.json'),'selection_raw_audit_sha256':sha256(SEL/'raw_audit.json'),'report_evidence_sha256':sha256(SEL/'report_evidence.json'),'pipeline_lock_sha256':sha256(OUT/'pipeline_lock.json'),'unix':time.time()};write_json(OUT/'pipeline_complete.json',done);write_json(SEL/'pipeline_complete.json',done);write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
