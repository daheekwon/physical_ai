"""Freeze nominal smoke; collect/audit36 fresh development/calibration families."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_policy_nominal';read=lambda p:json.loads(p.read_text())


def report():
    audit=read(OUT/'raw_audit.json');assert not audit['smoke'] and audit['episodes']==36 and audit['fresh_native_policy_predictions']==108 and audit['all_physics_and_RGB_replayed_exact'];f=pd.read_csv(OUT/'evaluation.csv');assert len(f)==36 and f.seed.nunique()==36
    for p,h in audit['inputs'].items():assert sha256(resolve(p))==h,p
    lines=['<!-- BEGIN DINO FRESH POLICY NOMINAL -->','## 모델 비교 확장을 위한 새로운 PushT 정책 family','', '현재 expert HDF5의 18,685개 episode는 초기 pose 185개 family를 공유하며, 기존 downstream train/development/calibration/test 분할이 이 family들을 모두 사용했다. 다른 episode를 고르는 것만으로 새로운 초기 상태 평가가 되지 않는다. 따라서 별도의 frozen-policy 조건으로 모델 비교를 확장한다. Expert 결과와 같은 모집단의 최종 평가로 해석하지 않는다.','','새 초기 상태에서 frozen Diffusion Policy를 수정 없이 실행해 development 12개·calibration 24개, **36개 30초 episode**를 수집하고 검증했다. 새 100개 family는 기존 expert·수집된 policy 초기 pose 및 서로 간에 중복이 없음을 확인했다. 이 중 나머지 64개는 후속 test용으로 예약했으며 정책 추론·궤적·성공 결과는 아직 실행하지 않았다. 초기 상태의 중복 확인과 모델 pretraining 데이터 노출은 다른 문제다.','','| 역할 | Episode | 한 번 이상 목표 coverage 성공 | 최종 평균 coverage |','| --- | --- | --- | --- |']
    for split,g in f.groupby('split'):lines.append(f"| {split} | {len(g)} | {100*g.success_any.mean():.2f}% | {g.final_coverage.mean():.3f} |")
    lines+=['','정책과 checkpoint, 100-step diffusion 추론, 8행동 chunk, 검증된 absolute-target→relative-action 변환을 고정했다. 성공 뒤에도 300개 제어 시점까지 실행했다. 물리 108,000개 표본과 RGB 10,836개를 모두 새로 재생해 확인하고, action 변환·polygon coverage를 독립 계산했으며 고정 시점16/96/192에서 108개 native 정책 추론을 다시 확인했다.','','이 단계는 **수정 없는 정책의 데이터·인터페이스 검증**이다. 다음 단계에서 같은 H5＋3 후보의 LeWM·DINO·제어기·접촉 선택을 development/calibration에서 비교하고 규칙을 고정한 뒤 예약된 test로 진행한다. 이 nominal 결과만으로 smoothing이나 모델 우위를 주장하지 않는다.','','근거: [expert family 잔여 감사](../artifacts/followup_v1/protocol/dinowm_new_family_preflight.json), [새 family·설정 잠금](../artifacts/followup_v1/dinowm_policy_nominal/manifest.json), [nominal 평가](../artifacts/followup_v1/dinowm_policy_nominal/evaluation.csv), [독립 재생 감사](../artifacts/followup_v1/dinowm_policy_nominal/raw_audit.json).','<!-- END DINO FRESH POLICY NOMINAL -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN DINO COMMON VERIFIER -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['manifest.json','raw_audit.json','evaluation.csv']},'source_sha256':sha256(Path(__file__)),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()})


def main():
    smoke=BASE/'dinowm_policy_nominal_smoke';a=read(smoke/'raw_audit.json');assert a['episodes']==1 and a['fresh_native_policy_predictions']==3 and a['all_physics_and_RGB_replayed_exact'];assert a['source_sha256']==sha256(resolve('scripts/audit_followup_dinowm_policy_nominal.py'))
    for p,h in read(smoke/'manifest.json')['input_hashes'].items():assert sha256(resolve(p))==h,p
    paths=['configs/followup_dinowm_policy_nominal.json','scripts/run_followup_dinowm_policy_nominal.py','scripts/audit_followup_dinowm_policy_nominal.py','scripts/run_followup_dinowm_policy_nominal_pipeline.py','scripts/run_followup_policy_refinement.py','scripts/analyze_followup_policy_final.py','zj_wam/policy_refinement.py','zj_wam/diagnostics.py'];OUT.mkdir(exist_ok=True);lock={'inputs':{p:sha256(resolve(p)) for p in paths},'smoke_raw_audit_sha256':sha256(smoke/'raw_audit.json'),'scope':'36nominal development/calibration episodes only. Reserved64test policy outcomes remain unexecuted.'}
    if (OUT/'pipeline_lock.json').exists():assert read(OUT/'pipeline_lock.json')==lock
    else:write_json(OUT/'pipeline_lock.json',lock)
    for stage in ['run','audit']:
        for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
        write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'running','updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve('scripts/'+stage+'_followup_dinowm_policy_nominal.py'))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'failed','returncode':r.returncode,'updated_unix':time.time()});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    report();write_json(OUT/'pipeline_complete.json',{'episodes':36,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_sha256':sha256(OUT/'report_evidence.json'),'unix':time.time()});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})

if __name__=='__main__':main()
