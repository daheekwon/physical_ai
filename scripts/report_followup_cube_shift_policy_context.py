"""Record verified causal-command preparation without claiming adaptive efficacy."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
OUT=resolve('artifacts/followup_v1/cube_shift_policy_context');read=lambda p:json.loads(p.read_text())
def main():
    a=read(OUT/'raw_audit.json');assert a['contexts']==336 and a['nominal_command_exact_states']==48 and a['independent_command_rows']==2688;assert a['manifest_sha256']==sha256(OUT/'manifest.json');assert a['source_sha256']==sha256(resolve('scripts/prepare_followup_cube_shift_policy_context.py'))
    for p,h in {**read(OUT/'manifest.json')['inputs'],**a['inputs']}.items():assert sha256(resolve(p))==h,p
    lines=['<!-- BEGIN CUBE SHIFT CAUSAL COMMANDS -->','## Cube 조건 변화 후 사용할 명령의 인과적 재구성','', '이미 검증한12개 development family·48개 phase·7개 조건의 **336개 과거 관측 이력**에서 현재 robot 상태로 다음 H5＋3명령을 다시 구성했다. 명령 계산 함수에는 초기 상태에서 정한 privileged plan, 현재 robot 위치·yaw, 과거 robot 위치·명령 목표와 gripper 관측만 전달한다. 물성 변경값·물체 상태·이후의 물리 궤적·기록된 미래 정책 행동은 명령 계산 입력으로 사용하지 않는다.','', 'Robot PD 계수와 gripper 응답 계수는 실제 과거 관측으로 피팅한다. Controller가 읽는 현재 위치는 마지막 적분 이전 qpos의 cached kinematics로 복원하며, 평가용 fresh EEF 좌표와 구분했다. 모든2,688개 명령을 별도 compiler 수식으로 대조했고, 물성 변경 없는48개 상태의8명령은 기존 검증된 nominal 명령과 완전히 일치했다.','', '첫 검사에서는 float32 명령을 너무 일찍 float64로 변환해 작은 gripper 목표가 달라지는 오류가 발생했다. 실패 산출물을 보존한 뒤 simulator의 연산 순서와 동일하게 수정했다. 원래 수치 허용 오차와 명령 완전 일치 검사는 완화하지 않았다. 기존 nominal 및 조건 변화 물리 기록은 수정하지 않았다.','', '**완료 범위는 다음 correction 실험의 입력 준비다.** 이 단계에서 새 물리 분기나 이미지 재실행을 추가하지 않았으며, 앞서 전체 재현 감사를 통과한 이력을 사용했다. 원래 기록된 미래 행동을 그대로 사용하는 진단과 구분해, 이후 실제 correction 비교에서는 이 현재 관측 기반 명령을 사용해야 한다. 아직 allowance/계산량 적응의 네 조건에서 보존–jerk 이득을 확인한 결과가 아니다.','', '근거: [입력 프로토콜](../artifacts/followup_v1/cube_shift_policy_context/manifest.json), [명령 감사](../artifacts/followup_v1/cube_shift_policy_context/raw_audit.json), [실패 원인 기록](../artifacts/followup_v1/cube_shift_policy_context_failure_v0/diagnosis.json).','<!-- END CUBE SHIFT CAUSAL COMMANDS -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text()
    if lines[0] in s:i=s.index(lines[0]);j=s.index(lines[-1])+len(lines[-1]);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN CUBE SHIFT HISTORY -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'manifest_sha256':sha256(OUT/'manifest.json'),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()})
if __name__=='__main__':main()
