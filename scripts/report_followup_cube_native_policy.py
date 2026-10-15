"""Report verified native fixed-target Cube policy baselines with their input scope."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import beta
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_native_policy';read=lambda p:json.loads(p.read_text())


def main():
    raw=read(OUT/'raw_audit.json');done=read(OUT/'pipeline_complete.json');m=read(OUT/'manifest.json');a=read(OUT/'analysis.json');assert raw['episodes']==24 and raw['families']==12 and raw['physics_samples']==180000 and done['raw_audit_sha256']==sha256(OUT/'raw_audit.json')
    for name,h in m['inputs'].items():assert sha256(resolve(name))==h,name
    rows=[]
    for r in a['summary']:
        rows.append(f"| {r['policy']} | {round(r['success_any']['mean']*12)}/12 | {round(r['success_final']['mean']*12)}/12 | {round(r['released_success_final']['mean']*12)}/12 | {r['final_position_error_m']['mean']*1000:.2f} | {r['first_success_seconds']['mean']:.2f} | {r['eef_jerk_energy']['mean']:.3g} |")
    block=f'''<!-- BEGIN CUBE NATIVE POLICY BASELINE -->
## Cube의 자기 관측 기반 정책 baseline 확보

Cube의 반복 실행을 위한 **native Markov oracle과 계획 추종 oracle baseline**을 새 개발 scene 12개에서 각각 실행했다. 두 정책 모두 12/12개 scene에서 15초 종료 시점의 목표 위치 조건을 만족했고, 종료 시 로봇–물체 접촉도 없었다. 아직 smoothing을 적용한 결과는 아니다.

저장소의 데이터 수집기가 사용하는 OGBench `CubeMarkovOracle`과 `CubePlanOracle`을 사용했다. 두 정책 모두 현재 로봇 proprioception과 privileged 물체/목표 상태를 입력받는다. Markov는 매 제어 시점의 관측에 반응하고, plan은 초기 물체·목표로 만든 EEF 계획을 현재 proprioception에 맞춰 추종한다. 미래 simulator 분기 결과를 정책 입력으로 제공하지 않았다. 학습된 이미지 정책과 같은 정보 조건으로 비교하지 않는다.

설정은 20Hz 제어·500Hz 물리, 한 scene당 300개 행동이다. 원래 noisy multi-target 수집 wrapper와 구분해 **추가 action/plan noise 없이 한 목표를 15초 동안 유지**했다. Oracle의 done이 켜져도 새 목표로 전환하거나 조기 종료하지 않았다. Action은 원래 wrapper처럼 [-1,1]로 제한한 float32다. Cube 시작·목표·EEF 위치를 명시적인 seeded variation으로 지정하고, 시작–목표 거리가 10cm 이상인 scene을 정책 실행 전에 구성했다. 같은 seed의 두 정책은 초기 상태와 목표가 정확히 일치했다.

| 정책 | 한 번 이상 성공 | 종료 성공 | 종료 성공＋접촉 없음 | 종료 위치 오차 평균, mm | 첫 성공 시각 평균, s | 전체 EEF jerk energy 평균, m²/s⁵ |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

여기서 native task 성공은 **목표와 위치 차이 40mm 이내**라는 환경의 기준이다. 물체 궤적 보존의 **nominal 대비 2mm·full 회전 2도** 조건과 다르다. 목표 회전 오차·접촉·물체 높이·EEF angular jerk·joint jerk도 원본 결과에 따로 저장했다. 두 정책의 jerk 차이는 서로 다른 nominal 동작에서 나온 값이므로 물체 궤적을 보존하는 smoothing 효과로 해석하지 않는다. 12개 개발 scene의 관측 성공이며, 0/12 실패에서도 이항 가정하의 양측 95% 구간의 실패율 상한은 약 {100*beta.ppf(.975,1,12):.1f}%다.

원시 검증은 24개 episode·180,000개 물리 표본 전체의 action 범위, 시간축, control 시점과 dense trace의 일치, 목표 위치에 따른 성공 판정을 포함한다. 첫 두 scene의 두 정책은 dense 계측을 끈 새 실행과 다시 비교했고, 초기 상태·행동·control 상태·성공 표식이 정확히 일치했다. 500Hz jerk는 같은 15초의 실제 EEF 위치에서 계산했으며 진입 전 history를 추가한 refinement 지표와는 구간이 다르다.

**다음 단계는 이 baseline을 H=5 nominal action chunk와 결합하는 것이다.** Native 정책은 매 시점 현재 상태를 받아 행동을 내므로, 미래 실제 물리 상태를 미리 읽어 다섯 행동을 만드는 방식은 사용하지 않는다. 계획의 알려진 목표와 검증된 causal controller 예측으로 chunk를 만드는 경우에는 그 compiler 자체의 nominal 성능을 먼저 검증해야 한다. 이후 같은 compiler에서 correction 유무·반복·조건 변화와 누적 물체 궤적 오차를 비교한다. 이번 baseline 검증으로 Cube closed-loop smoothing이나 적응까지 완료한 것은 아니다.

근거: [고정 설정](../configs/followup_cube_native_policy.json), [정책·소스 manifest](../artifacts/followup_v1/cube_native_policy/manifest.json), [baseline 결과](../artifacts/followup_v1/cube_native_policy/analysis.json), [원시 감사](../artifacts/followup_v1/cube_native_policy/raw_audit.json).
<!-- END CUBE NATIVE POLICY BASELINE -->
'''
    doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();start='<!-- BEGIN CUBE NATIVE POLICY BASELINE -->';end='<!-- END CUBE NATIVE POLICY BASELINE -->'
    if start in s:i=s.index(start);j=s.index(end,i)+len(end);s=s[:i]+block.rstrip()+s[j:]
    else:anchor='<!-- END CUBE RUNTIME IMPLEMENTATION -->';i=s.index(anchor)+len(anchor);s=s[:i]+'\n\n'+block+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{name:sha256(OUT/name) for name in ['manifest.json','raw_audit.json','analysis.json','pipeline_complete.json']},'document':str(doc.relative_to(resolve('.'))),'section_sha256':hashlib.sha256(block.rstrip().encode()).hexdigest()});print('REPORTED native Cube baselines',flush=True)
if __name__=='__main__':main()
