"""Wait for the existing Cube final job, independently audit effects, then report."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,write_json,sha256

BASE=resolve('artifacts/followup_v1/cube');DEST=BASE/'final_analysis'
read=lambda p:json.loads(p.read_text())

def report():
    from scripts.run_followup_cube_final import verify,LOCK
    lock=verify();a=read(DEST/'analysis.json');raw=read(DEST/'raw_audit.json');probe=read(DEST/'probe_effect_raw_audit.json')
    assert not a['partial'] and not raw['partial'] and not probe['validation_only']
    for item in [a,raw,probe]:assert item['lock_sha256']==sha256(LOCK)
    assert raw['states']==probe['states']==lock['planned_states']
    assert probe['effect_rows']==a['probes']['effect_rows']
    primary=next(x for x in a['selection']['paired'] if x['bank']=='common' and x['method']=='lewm_global' and x['baseline_bank']=='common' and x['baseline']=='controller_displacement')
    delta=primary['U_difference'];lo,hi=delta['ci95'];rows=[]
    names={'controller_displacement':'제어기 변위 제한','proprio_contact':'Proprioception 기반 접촉 proxy','lewm_global':'LeWM global','oracle':'유한 후보 oracle'}
    pct=lambda x:'미정의' if x is None else f'{100*x:.1f}%'
    for method,label in names.items():
        x=next(r for r in a['selection']['summary'] if r['bank']=='common' and r['method']==method and r['phase']=='all')
        rows.append(f"| {label} | {pct(x['joint']['mean'])} | {pct(x['risk']['mean'])} | {pct(x['active']['mean'])} |")
    decision=a['primary_decision']
    conclusion=('사전 선언한 경험적 위험·적용 조건과 실질적 개선 기준을 충족했다.' if decision['supported_practical_advantage'] else '사전 선언한 조건에서 실질적인 LeWM 추가 우위를 입증하지 못했다.')
    feasibility='LeWM은 선언한 경험적 위험·적용 조건을 충족했다.' if decision['empirical_feasible'] else 'LeWM은 선언한 경험적 위험·적용 조건을 충족하지 못했다.'
    captures=[]
    for name in ['frame__ridge_symmetry','history__ridge_symmetry']:
        x=next(r for r in a['probes']['capture'] if r['readout']==name);t=a['probes']['resolution_thresholds'][name]
        captures.append(f"| {name} | {1000*x['position_m']['mean']:.2f}mm | {math.degrees(x['rotation_mod_cube_symmetry_rad']['mean']):.2f}도 | {1000*t['position_m']:.2f}mm / {math.degrees(t['rotation_rad']):.2f}도 |")
    text=f'''<!-- BEGIN CUBE FINAL TEST -->
## 잠근 Cube expert 최종 평가

독립 downstream 초기 상태 family {len(lock['families'])}개·상태 {raw['states']}개에서 접근·grasp·운반·release를 평가했다. 최종 물리 후보 {raw['candidate_rows']:,}행을 원시 trace로 감사하고, frozen readout의 효과 {probe['effect_rows']:,}행을 별도 회전 행렬 계산으로 다시 검증했다. Readout·threshold·radius는 최종 데이터에서 학습하거나 선택하지 않았다. World model 사전학습 데이터 비노출을 보장하는 분리는 아니다.

물체 위치 2mm·full SO(3) 회전 2도, 500Hz 같은 시각의 궤적 보존과 실제 EEF jerk 10% 감소 기준을 유지했다. Gripper 사건은 원래 값·시점으로 고정했다. Probe 진단의 Cube 대칭 처리가 물리 보존 허용 오차를 완화하지 않는다.

**주 비교인 LeWM−제어기의 U 차이는 {100*delta['mean']:+.2f}%p, family bootstrap 95% 구간 [{100*lo:+.2f}, {100*hi:+.2f}]%p다.** {conclusion} {feasibility} 신뢰구간이 0을 포함하는 결과를 동등성으로 해석하지 않는다.

| 같은 후보의 선택기 | 보존·jerk 동시 달성률 U | 선택 후 위반율 | 수정 적용 비율 |
| --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

접촉 proxy의 위반율 11.7%는 경험적 목표 10%를 넘었다. U가 높다는 이유만으로 위험 조건을 충족한 방법보다 우수하다고 판단하지 않는다.

공통 후보에 LeWM 방향도 포함되므로 제어기 열을 완전히 LeWM 없는 pipeline이라고 부르지 않는다. LeWM 방향을 제거한 별도 bank와, 후보 수·action norm을 맞춘 low/high/random bank의 고정 verifier 비교를 원본 분석에 함께 제공한다. 이들은 실제 EEF 움직임이나 계산 비용까지 맞춘 비교는 아니다. Oracle도 실행한 유한 후보 중 최선이며 배포 가능한 선택기는 아니다.

| 고정 readout | 실제 상태 위치 오차 평균 | 대칭을 고려한 회전 오차 평균 | 개발에서 고정한 효과 해상도 위치 / 회전 |
| --- | --- | --- | --- |
'''+ '\n'.join(captures)+'''

같은 최종 bank에서 유용한 후보가 존재한 상태는 34.0%였다. LeWM이 그런 후보를 하나 이상 승인한 상태는 전체의 24.6%, 예상 jerk 개선 조건까지 통과한 것은 21.9%, 최종적으로 유용한 후보를 고른 것은 13.3%였다. 따라서 현재 bank의 후보 부족과 승인·순위 선택의 손실이 모두 남아 있다. 이 분해로 모든 oracle 차이를 모델 예측 오류 하나에 귀속하지 않는다.

이 표는 실제 상태를 읽는 정밀도와 작은 intervention 차이를 읽는 정밀도를 구분한다. 예측 효과와 real-endpoint·zero·wrong-action의 비교는 효과 분석 원본에 유지했다. 해상도보다 작은 사례는 방향 진단에서 unresolved로 표시하고 물리 보존 통계에서는 제거하지 않았다. Probe를 최종 correction 선택에 사용한 결과는 아니다.

후보 수를 독립 표본 수로 세지 않으며 구간은 초기 상태 family 단위다. 보조 비교는 탐색적이고, 실제 이동량 매칭의 공통 표본 부족은 별도 분석의 한계로 남는다. 이번 완료 범위는 Cube expert 최종 평가이며, Cube의 지연·반복·조건 적응·호환 정책과 추가 모델·task·실물은 별도 후속 작업이다.

근거: [Cube 최종 protocol](../artifacts/followup_v1/cube/final_protocol.json), [최종 선택·probe 분석](../artifacts/followup_v1/cube/final_analysis/analysis.json), [원시 물리 감사](../artifacts/followup_v1/cube/final_analysis/raw_audit.json), [독립 probe 효과 감사](../artifacts/followup_v1/cube/final_analysis/probe_effect_raw_audit.json).
<!-- END CUBE FINAL TEST -->
'''
    path=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');doc=path.read_text();begin='<!-- BEGIN CUBE FINAL TEST -->';end='<!-- END CUBE FINAL TEST -->'
    if begin in doc:
        start=doc.index(begin);stop=doc.index(end,start)+len(end);doc=doc[:start]+text.strip()+doc[stop:]
    else:
        anchor='<!-- BEGIN FINAL EXPERT TEST -->';assert anchor in doc;doc=doc.replace(anchor,text+'\n'+anchor,1)
    path.write_text(doc)
    write_json(DEST/'report_evidence.json',{'lock_sha256':sha256(LOCK),'inputs':{name:sha256(DEST/name) for name in ['analysis.json','raw_audit.json','probe_effect_raw_audit.json']},'source_sha256':sha256(Path(__file__)),'scope':'Cube expert final only; full F0-F11 remains incomplete','unix':time.time()})
    print('REPORTED Cube final',raw['states'],'states',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-pid',type=int,required=True);args=p.parse_args();start=time.monotonic()
    validation=read(BASE/'final_validation_development/probe_effect_raw_audit.json');assert validation['validation_only'] and validation['states']==4 and validation['effect_rows']==5760
    source=resolve('scripts/audit_followup_cube_final_probes.py');assert validation['inputs'][str(source.relative_to(resolve('.')))]==sha256(source)
    print('WAITING for original Cube final execution and analysis; audit validation passed',flush=True)
    while not (DEST/'analysis.json').exists() or not (DEST/'raw_audit.json').exists():
        try:os.kill(args.dependency_pid,0)
        except ProcessLookupError:raise RuntimeError('Cube parent ended before analysis; inspect failure, no automatic restart')
        if time.monotonic()-start>21600:raise RuntimeError('Six-hour observation window elapsed; revalidate parent')
        time.sleep(10)
    with (DEST/'independent_probe_audit.log').open('a') as log:
        subprocess.run([sys.executable,str(source)],stdout=log,stderr=subprocess.STDOUT,check=True)
    report()

if __name__=='__main__':main()
