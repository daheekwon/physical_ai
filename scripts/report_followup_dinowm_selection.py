"""Report audited common-candidate native DINO verifier calibration."""
import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_selection';read=lambda p:json.loads(p.read_text())

def main():
    a=read(OUT/'analysis.json');audit=read(OUT/'raw_audit.json');complete=read(OUT/'pipeline_complete.json');assert not audit['smoke'] and audit['states']==402 and audit['selected_rows']==2010;assert audit['analysis_sha256']==sha256(OUT/'analysis.json') and audit['source_sha256']==sha256(resolve('scripts/audit_followup_dinowm_selection.py'));assert complete['raw_audit_sha256']==sha256(OUT/'raw_audit.json')
    for p,h in read(OUT/'pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p
    def pct(v):return '—' if v is None else f'{100*v:.2f}%'
    def summary(env,method):return next(r for r in a['summary'] if r['environment']==env and r['method']==method and r['split']=='development' and r['phase']=='all')
    lines=['<!-- BEGIN DINO COMMON VERIFIER -->','## DINO-WM을 같은 후보의 승인기로 사용한 첫 비교','',
           f"**PushT 210개·Cube 192개 상태**의 기존 개발·calibration 물리 trace를 재사용했다. 공통 반경 제한 안의 {audit['candidate_rows']:,}개 후보 행과 다섯 방법의 **2,010개 상태별 선택**, {audit['calibration_trials']:,}개 threshold 시도를 감사했다. 모든 방법은 같은 상태에서 정확히 같은 후보와 예상 실행 jerk 순위를 사용한다. 후보 생성에는 world model을 사용하지 않았고, Cube의 기존 LeWM low/high 후보는 제거했다.", '',
           'DINO의 첫 점수는 후보 예측과 nominal 예측의 차이를 patch별 제곱거리로 평균한 RMS이고, 두 번째는 patch 평균 feature의 거리다. 두 구성을 사전에 정해 각각 보고하며 결과를 보고 하나만 선택하지 않았다. LeWM은 native global latent 거리, 제어기는 예상 로봇 변위, 접촉 모델은 기존 물리·기하 점수를 사용한다. 별도 물체 readout 학습은 추가하지 않았다. DINO는 검증된 third-party retrain으로 원래 공개 checkpoint와 구분한다.', '',
           '각 방법은 같은 calibration family에서 위험 목표 10%·최소 수정 적용 비율 25%를 사용해 독립적으로 보정했다. PushT는 접촉 표본의 선택 후 위험도 10% 이하여야 한다. 적합한 threshold가 없으면 규칙을 비활성화한다. 아래는 이미 연구에 노출된 development의 진단 결과이며, calibration 값이나 새 최종 test로 해석하지 않는다.', '',
           '| 환경·승인 방법 | U | 수정 적용 비율 | 선택 후 위반율 | 평균 실제 jerk 감소 |','| --- | --- | --- | --- | --- |']
    for env in ['pusht','ogb-cube']:
        for method in ['controller','contact','lewm_global','dino_patch_rms','dino_patch_mean']:
            r=summary(env,method);lines.append(f"| {env} / {method} | {pct(r['joint']['mean'])} | {pct(r['active']['mean'])} | {pct(r['selected_risk']['mean'])} | {pct(r['jerk_gain']['mean'])} |")
    lines+=['','DINO−LeWM의 U 차이는 다음과 같다. Family bootstrap 95% 구간이 0을 포함하면 우위를 입증하지 못한 것이며 동등성의 증거도 아니다. Controller·contact 대비 paired 차이와 접촉 단계별 결과는 원본에 제공한다.','','| 환경·DINO 점수 | U 차이, %p | 95% 구간, %p |','| --- | --- | --- |']
    for env in ['pusht','ogb-cube']:
        for method in ['dino_patch_rms','dino_patch_mean']:
            r=next(r for r in a['paired'] if r['environment']==env and r['split']=='development' and r['method']==method and r['baseline']=='lewm_global' and r['metric']=='joint');lo,hi=r['ci95'];lines.append(f"| {env} / {method} | {100*r['mean']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines+=['','물체 궤적 보존은 PushT 2px·2도·coverage 감소 0.02, Cube 2mm·full 회전 2도 조건이다. 각각 100Hz·500Hz의 수정 구간과 원래 후속 행동 3개를 검사한다. U는 보존과 실제 로봇 병진 jerk 10% 감소를 함께 요구한다. 모든 경우에 fallback을 포함한 전체 상태가 분모이며, 수정이 없을 때의 조건부 위반율은 미정의다.','',
            '상대 radius 0.2는 환경별 기존 정규화·후보 생성 규약을 따른다. 두 환경의 실제 물리 크기가 같다는 뜻은 아니다. PushT 접촉 모델은 현재 RGB의 기하 추정과 robot proprioception을, Cube 접촉 모델은 gripper 센서와 이미 알려진 명령을 사용한다. 추가 관측 입력을 숨기지 않는다. Cube의 후보 순위 목적함수는 기존 XYZ·yaw 비용이며 실제 주 jerk는 EEF 병진이다. 그 차이를 유지한 재사용 비교임을 명시한다.','',
            '모든 token 거리, 물리 pose·coverage·jerk·action norm·예상 비용, 후보 포함 조건, 전체 calibration과 최종 선택 및 family 구간을 감사했다. PushT 접촉 점수는 별도로 감사된 고정 원본을 재사용했다. 원시 점수를 CSV로 왕복하면서 threshold 경계가 바뀌지 않도록 검증된 원본 연산 정밀도로 선택을 재현했다.','',
            '**이 결과는 verifier의 개발 비교다.** 저장된 예측을 사용했으므로 전체 추론 비용이나 DINO 방향 생성의 가치를 측정한 결과가 아니다. 실제 계산 비용을 맞춘 비교, 새 family의 최종 평가, 정책·반복·조건 변화 평가가 남아 있다. Calibration에서 경험적 조건을 만족해도 development나 실제 배포의 위험을 보증하지 않는다.','',
            '근거: [설정](../configs/followup_dinowm_selection.json), [분석](../artifacts/followup_v1/dinowm_selection/analysis.json), [독립 감사](../artifacts/followup_v1/dinowm_selection/raw_audit.json), [승인 규칙](../artifacts/followup_v1/dinowm_selection/rules.json), [선택 결과](../artifacts/followup_v1/dinowm_selection/selected.csv).','<!-- END DINO COMMON VERIFIER -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=doc.read_text();start='<!-- BEGIN DINO COMMON VERIFIER -->';end='<!-- END DINO COMMON VERIFIER -->'
    if start in text:i=text.index(start);j=text.index(end)+len(end);text=text[:i]+section+text[j:]
    else:i=text.index('<!-- BEGIN OUTPUT REPLACEMENT LEWM -->');text=text[:i]+section+'\n\n'+text[i:]
    doc.write_text(text);assert section in doc.read_text();write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{p:sha256(OUT/p) for p in ['analysis.json','raw_audit.json','rules.json','selected.csv','pipeline_complete.json']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time(),'scope':'Common verifier development/calibration complete. Cost/new-familyfinal/repeated/policy remain.'});print('REPORTED DINO common402',flush=True)

if __name__=='__main__':main()
