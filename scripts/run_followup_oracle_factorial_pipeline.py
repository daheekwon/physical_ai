"""Freeze four-state A-D smoke and run full fixed-bank analysis/audit/report."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'oracle_factorial';read=lambda p:json.loads(p.read_text())

def report():
    a=read(OUT/'raw_audit.json');result=read(OUT/'analysis.json');assert a['states']==402 and a['selections']==8040 and a['analysis_sha256']==sha256(OUT/'analysis.json');assert a['source_sha256']==sha256(resolve('scripts/audit_followup_oracle_factorial.py'))
    def lookup(env,cell):return next(r for r in result['summary'] if r['environment']==env and r['split']=='development' and r['method']=='lewm_global' and r['cell']==cell)
    def pct(v):return '미정의' if v is None else f'{100*v:.2f}%'
    lines=['<!-- BEGIN ACCEPTANCE JERK FACTORIAL -->','## 보존 판정과 jerk 순위의 교차 진단: 고정 후보 A–D 비교','', '이미 독립 감사한 PushT210·Cube192개 development/calibration 상태의 **같은 비-WM 다항식 후보**에서 승인 기준과 jerk 순위만 교차했다. 새 threshold나 후보를 학습하지 않았고, A의 선택이 기존 controller/contact/LeWM/DINO 두 metric의 선택과 전부 일치하는지 확인했다. 아래는 development의 LeWM 승인 기준을 중심으로 한 결과이며 모든 방법·calibration 결과와 family 구간은 분석 파일에 있다.','','A는 현재 verifier+예상 jerk, B는 현재 verifier+실제 jerk, C는 실제 물체 보존 판정+예상 jerk, D는 실제 보존 판정+실제 jerk다. B·C·D는 미래 실행을 아는 사후 진단이다. B는 verifier가 잘못 승인한 unsafe 후보를 고를 수 있다. 모든 셀에 nominal fallback을 포함하고, 예상 순위에서는 예상 개선이 없는 수정, 실제 순위에서는 실제 개선이 없는 수정을 강제로 적용하지 않았다.','','| Task | 셀 | 보존·10% jerk 동시 달성 U | 수정 적용 비율 | 선택 후 위반율 | 평균 실제 jerk 감소 |','| --- | --- | --- | --- | --- | --- |']
    for env in ['pusht','ogb-cube']:
        for cell in ['A','B','C','D']:
            r=lookup(env,cell);lines.append(f"| {env} | {cell} | {pct(r['joint']['mean'])} | {pct(r['active']['mean'])} | {pct(r['selected_risk']['mean'])} | {pct(r['jerk_gain']['mean'])} |")
    lines+=['','각 셀의 U와 적용 비율은 전체 상태를 분모로 한다. 위반율은 실제 수정한 상태 중 비율이며, 적용이 없으면 정의되지 않는다. D는 탐색한 유한 후보 안의 최선이며 전역 물리 한계가 아니다. C–A는 같은 예상 순위에서 보존 판정 변경의 여지, D–C는 실제 보존 후보 안에서 jerk 순위를 개선할 여지를 보여준다. 이 차이들은 상호작용을 포함하므로 독립적인 오류 기여도로 더하지 않는다.','','고정 후보 자체에 유용한 수정이 없는 상태, 유용한 후보가 있지만 현재 verifier가 모두 거절한 상태, 승인 후 선택에서 놓친 상태도 함께 기록했다. 물리 보존 판정을 정확하게 알아도 예상 jerk 순위가 개선을 놓친다면, representation 분석만 확대하기보다 action→실행 응답 및 목적함수를 보완해야 한다. 실제 jerk 순위만 바꾼 B의 개선도 물체 위반을 함께 보아야 한다.','','검증은 모든 원시 dense 물체 오차·실제 jerk 재계산, 기존 감사된 모델 점수와 후보 입력의 hash 확인, 별도 정렬 구현의 8,040개 선택, 요약 및 paired family 신뢰구간 재현을 포함한다. 기존 개발 자료의 원인 진단이며 독립 최종 평가나 배포 가능한 oracle 성능으로 해석하지 않는다.','','근거: [고정 프로토콜](../artifacts/followup_v1/oracle_factorial/protocol.json), [전체 분석·paired 구간](../artifacts/followup_v1/oracle_factorial/analysis.json), [독립 감사](../artifacts/followup_v1/oracle_factorial/raw_audit.json), [선택 결과](../artifacts/followup_v1/oracle_factorial/selected.csv).','<!-- END ACCEPTANCE JERK FACTORIAL -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN DINO COMMON VERIFIER -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{p:sha256(OUT/p) for p in ['analysis.json','raw_audit.json','selected.csv','protocol.json']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()})

def main():
    smoke=BASE/'oracle_factorial_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['states']==4 and a['selections']==80;assert a['source_sha256']==sha256(resolve('scripts/audit_followup_oracle_factorial.py')) and a['analysis_sha256']==sha256(smoke/'analysis.json')
    for p,h in read(smoke/'protocol.json')['inputs'].items():assert sha256(resolve(p))==h,p
    files=['configs/followup_oracle_factorial.json','scripts/analyze_followup_oracle_factorial.py','scripts/audit_followup_oracle_factorial.py','scripts/run_followup_oracle_factorial_pipeline.py'];OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in files},'smoke_audit_sha256':sha256(smoke/'raw_audit.json')}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for stage in ['analyze','audit']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'running','updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve('scripts/'+stage+'_followup_oracle_factorial.py'))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'failed','returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    report();write_json(OUT/'pipeline_complete.json',{'states':402,'selections':8040,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
