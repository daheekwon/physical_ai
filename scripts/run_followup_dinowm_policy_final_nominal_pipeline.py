"""Lock policy correction comparison before test nominal rollouts; audit nominal interface."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
from scripts.run_followup_dinowm_policy_final_nominal import SOURCES
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_policy_final_nominal';FINAL=BASE/'dinowm_policy_final';read=lambda p:json.loads(p.read_text())

def verify():
    lock=read(FINAL/'final_protocol.json')
    for p,h in lock['inputs'].items():assert sha256(resolve(p))==h,p
    return lock

def freeze():
    if (FINAL/'final_protocol.json').exists():return verify()
    smoke=BASE/'dinowm_policy_final_nominal_smoke';a=read(smoke/'raw_audit.json');assert a['smoke'] and a['episodes']==1 and a['fresh_native_policy_predictions']==3 and a['all_physics_and_RGB_replayed_exact'];assert a['source_sha256']==sha256(resolve('scripts/audit_followup_dinowm_policy_final_nominal.py'))
    for p,h in {**read(smoke/'manifest.json')['input_hashes'],**a['inputs']}.items():assert sha256(resolve(p))==h,p
    for name in ['dinowm_policy_nominal','dinowm_policy_bank','dinowm_policy_selection']:
        folder=BASE/name;c=read(folder/'pipeline_complete.json');a=read(folder/'raw_audit.json');assert c['raw_audit_sha256']==sha256(folder/'raw_audit.json')
        if name=='dinowm_policy_selection':
            assert a['protocol_sha256']==sha256(folder/'protocol.json') and a['analysis_sha256']==sha256(folder/'analysis.json')
            checked=read(folder/'protocol.json')['inputs']
        else:checked=a['inputs']
        for p,h in {**checked,**a.get('audit_helpers',{}),**a.get('helpers',{})}.items():assert sha256(resolve(p))==h,p
    selection=BASE/'dinowm_policy_selection';a=read(selection/'raw_audit.json');assert a['states']==108 and a['selections']==540
    cfg=read(resolve('configs/followup_dinowm_policy_final.json'));parent=read(BASE/'dinowm_policy_nominal/manifest.json');seeds=list(range(cfg['test_seeds']['start'],cfg['test_seeds']['start']+cfg['test_seeds']['count']));assert len(seeds)==64 and all(parent['initials'][str(s)]['split']=='reserved_test' for s in seeds)
    rules={r['method']:r['rule'] for r in read(selection/'rules.json')['rules']};assert set(rules)==set(cfg['methods']);files=[resolve(p) for p in SOURCES+['scripts/audit_followup_dinowm_policy_final_nominal.py','scripts/run_followup_dinowm_policy_final_nominal_pipeline.py','scripts/original_expert_evaluation_core.py','scripts/run_followup_dinowm_policy_bank.py','scripts/audit_followup_dinowm_policy_bank.py','scripts/dinowm_runtime_core.py','scripts/output_replacement_core.py','scripts/model_runtime_imports.py','scripts/retrained_dinowm_core.py','scripts/analyze_followup_response_selection.py','scripts/analyze_followup_policy_final.py','scripts/lewm_runtime_reference.py','zj_wam/model_audit.py','zj_wam/response_models.py','zj_wam/repeated_refinement.py','zj_wam/followup.py','zj_wam/adapters.py','configs/followup_dinowm_policy_bank.json']]
    for folder,names in [('dinowm_policy_nominal',['manifest.json','raw_audit.json','pipeline_complete.json']),('dinowm_policy_bank',['manifest.json','raw_audit.json','pipeline_complete.json','export_recovery.json']),('dinowm_policy_selection',['protocol.json','rules.json','analysis.json','raw_audit.json','pipeline_complete.json']),('dinowm_policy_final_nominal_smoke',['manifest.json','raw_audit.json'])]:files.extend(BASE/folder/n for n in names)
    files+=sorted((BASE/'protocol/dinowm_retrained_inventory').glob('*.json'));files.append(resolve(read(resolve('configs/followup_dinowm_policy_nominal.json'))['policy_checkpoint']));files.append(resolve(read(resolve('configs/followup_v1.json'))['checkpoint'])/'weights.pt');OUT.mkdir(exist_ok=True);assert not list(OUT.glob('test/*')),'Test rollout existed before method lock';FINAL.mkdir(exist_ok=True)
    lock={'config':cfg,'rules':rules,'test_seeds':seeds,'test_initials':{str(s):parent['initials'][str(s)]['initial_state'] for s in seeds},'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'primary_decision':'DINOpatchRMS-LeWM paired family U difference >=.10, 95%interval lower>0, selected risk<=.10 and active>=.25. Other contrasts exploratory. Empiricalrisk not population certification.','analysis':'192fixedstate windows, family-level paired bootstrap2000; all64families included independent of nominaltask success; same55candidate bank and rules locked before testnominal rollout. Action/modeldecision before modifiedphysics; futurephysicaldata for evaluation only.','scope':'NativeCPU singlecorrection evaluation; no repeatedclosedloop/conditionchange/actualdelay/hardware claim. Nominal interface stage audited before new candidate branches. Downstream heldoutinitialfamilies, not guaranteed WMpretraining exclusion.','created_unix':time.time()}
    with (FINAL/'final_protocol.json').open('x') as f:json.dump(lock,f,indent=2);f.write('\n')
    print('LOCKED POLICY FINAL64FAMILIES192STATES',flush=True);return lock

def report():
    a=read(OUT/'raw_audit.json');assert a['episodes']==64 and a['all_physics_and_RGB_replayed_exact'];lines=['<!-- BEGIN DINO POLICY FINAL NOMINAL -->','## 예약한 정책64개 family의 nominal 독립 검증','', '새 정책 calibration24개·development12개 family의 후보 선택 감사를 완료한 뒤, 비교 방법·규칙·주 비교·지표·192개 평가 시점을 잠갔다. 그 이후 예약한64개 test 초기 family에서 frozen Diffusion Policy의 수정 없는300행동 실행을 수집하고, 전체192,000개 물리 시점 및19,264개 RGB를 재현했다. 각 episode의 사전 지정16·96·192 시점에서192개 native 정책 예측도 다시 계산했다.','', '이 단계는 독립 평가의 nominal 실행 및 입력 검증이다. **수정 후보의 최종 효과 비교는 아직 완료하지 않았다.** 동일한 H5＋후속3행동, 비-WM 다항식 후보, CPU LeWM·DINO 및 제어기·접촉 규칙으로 이어질 실험이며, test 결과에 맞춘 재보정은 하지 않는다. 정책 task 성공 여부로 family나 window를 제외하지 않는다.','', '개발 경로의 기존 family1개에서300행동·정책 예측·전체 물리·RGB가 원본과 정확히 일치하는 smoke를 먼저 통과했다. 정책 가중치, 명령 변환, 난수 규칙과 모든 calibration threshold를 고정했다. 원시 감사는 각 제어 구간의 absolute→relative 변환과 독립 polygon coverage, dense jerk도 확인했다.','', '근거: [최종 방법·분석 잠금](../artifacts/followup_v1/dinowm_policy_final/final_protocol.json), [nominal 입력 manifest](../artifacts/followup_v1/dinowm_policy_final_nominal/manifest.json), [독립 감사](../artifacts/followup_v1/dinowm_policy_final_nominal/raw_audit.json).','<!-- END DINO POLICY FINAL NOMINAL -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN DINO FRESH POLICY SELECTION -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{p:sha256(OUT/p) for p in ['manifest.json','raw_audit.json','evaluation.csv']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()})

def main():
    freeze()
    for stage in ['run','audit']:
        verify();write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'running','updated_unix':time.time()})
        with (OUT/(stage+'.log')).open('a') as log:r=subprocess.run([sys.executable,str(resolve('scripts/'+stage+'_followup_dinowm_policy_final_nominal.py'))],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:write_json(OUT/'status.json',{'pid':os.getpid(),'stage':stage,'status':'failed','returncode':r.returncode});raise RuntimeError((stage,r.returncode))
        print('COMPLETE',stage,flush=True)
    verify();report();write_json(OUT/'pipeline_complete.json',{'episodes':64,'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'report_evidence_sha256':sha256(OUT/'report_evidence.json'),'final_protocol_sha256':sha256(FINAL/'final_protocol.json'),'unix':time.time(),'scope':'Nominal interface only;192state correction evaluation pending.'});write_json(OUT/'status.json',{'status':'complete','updated_unix':time.time()})
if __name__=='__main__':main()
