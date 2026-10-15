"""Report completed capture repetitions while retaining pending F4/F5 scope."""
import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'probe_stability';read=lambda p:json.loads(p.read_text())
def main():
    complete=read(OUT/'capture_pipeline_complete.json');audit=read(OUT/'capture_raw_audit.json');a=read(OUT/'capture_analysis.json');assert complete['raw_audit_sha256']==sha256(OUT/'capture_raw_audit.json') and audit['analysis_sha256']==sha256(OUT/'capture_analysis.json');assert audit['models']==180 and audit['reused_models']==72 and audit['evaluations']==360 and audit['alpha_trials_recomputed']==210;assert audit['source_sha256']==sha256(resolve('scripts/audit_followup_probe_stability.py'))
    for p,h in read(OUT/'capture_pipeline_lock.json')['inputs'].items():assert sha256(resolve(p))==h,p
    f=pd.read_csv(OUT/'capture_metrics.csv');variations=a['training_variation']
    for r in variations:
        g=f[(f.environment==r['environment'])&(f.split==r['split'])&(f.representation==r['representation'])&(f.fraction==r['fraction'])&(f.metric==r['metric'])];g=g[g.learner==r['learner']] if r['axis']=='training_subset' else g[g.learner.str.startswith('mlp64_seed_')];assert len(g)==r['replicates']==3
        np.testing.assert_allclose([g.value.mean(),g.value.min(),g.value.max()],[r['mean'],r['min'],r['max']],atol=1e-12)
    def interval(env,rep,metric,axis):
        r=next(r for r in variations if r['environment']==env and r['representation']==rep and r['split']=='calibration' and r['metric']==metric and r['axis']==axis and (axis!='training_subset' or r['fraction']==.5 and r['learner']=='ridge'));scale=1000 if metric=='position_m' else 180/np.pi if 'rotation' in metric else 1;return f"{r['min']*scale:.2f}–{r['max']*scale:.2f}"
    pairs=[('pusht','lewm_frame'),('pusht','lewm_history'),('pusht','dino_projected_patch_mean'),('pusht','dino_projected_spatial_quadrants'),('ogb-cube','lewm_frame'),('ogb-cube','dino_projected_patch_mean')]
    lines=['<!-- BEGIN PROBE STABILITY CAPTURE -->','## 추가 B의 capture 단계: 학습 표본과 초기화 민감도','',
    '**6개 표현·180개 probe 구성·360개 평가를 검증했다.** 조건이 정확히 같은 기존 모델 72개는 배열·학습 subset을 확인해 재사용했고 108개를 새로 학습했다. 학습 family의 25%·50%·100%를 포함하는 중첩 부분집합을 세 seed로 만들었다. 100% deterministic ridge를 세 독립 반복으로 세지 않고 한 번만 사용했으며, 전체 학습 데이터의 MLP만 세 초기화를 비교했다. 각 subset의 MLP 초기화는 고정해 두 변동 요인을 분리했다.','',
    '아래는 세 번의 **학습 반복에서 나온 평균 오차의 최소–최대 범위**다. 각 모델은 같은 calibration family에서 평가했으며, 이 범위는 신뢰구간이 아니다. Ridge alpha는 development에서만 골랐고 이 표로 calibration 승자를 새로 선택하지 않았다. PushT 위치는 px, Cube 위치는 mm다. Cube 회전은 영상 대칭을 고려한 진단 오차이며 실제 물체 궤적 보존의 full 회전 2도 조건은 유지한다.','',
    '| 환경·표현 | 50% 학습 ridge 위치 범위 | 50% 학습 ridge 회전 범위, deg | 100% 학습 MLP 위치 범위 | 100% 학습 MLP 회전 범위, deg |','| --- | --- | --- | --- | --- |']
    for env,rep in pairs:
        pos='position_px' if env=='pusht' else 'position_m';rot='rotation_rad' if env=='pusht' else 'rotation_mod_cube_symmetry_rad';lines.append(f"| {env} / {rep} | {interval(env,rep,pos,'training_subset')} | {interval(env,rep,rot,'training_subset')} | {interval(env,rep,pos,'mlp_initialization')} | {interval(env,rep,rot,'mlp_initialization')} |")
    lines+=['','Cube의 LeWM MLP 위치 오차는 초기화에 따라 약 69–161mm로 변했다. 따라서 작은 nonlinear probe를 한 번 학습한 결과만으로 표현에 물체 정보가 없다고 결론내릴 수 없다. 반대로 nonlinear probe를 붙였다는 사실만으로 더 정밀한 측정기가 되었다고 볼 수도 없다. 모델·표현·학습량별 결과와 평균값·label-shuffle 대조를 원본에 유지했다. 현재 반복에서 새 수렴 경고는 없었지만, 그것이 충분한 예측 정밀도를 의미하지는 않는다.','',
    '표현은 native future prediction과 대응 가능한 소수로 고정했다. LeWM spatial capture는 해당 공간의 native 미래 출력이 없으므로 이번 capture–effect 연계 반복에서 제외했고, 기존 capture 결과는 유지했다. LeWM frame과 DINO projected mean은 192차원이다. History·spatial 구성은 입력 차원이 다르므로 모든 행을 동일 차원의 모델 비교라고 부르지 않는다. DINO는 원래 공개 checkpoint가 아닌 별도 검증한 third-party retrain이다.','',
    '독립 감사는 모든 학습·평가 family 분리, 세 seed의 중첩 subset, train-only scaler, 210개 ridge alpha 점수와 재적합, 재사용 모델 배열의 정확한 일치, 예측 재계산 및 물리 단위 오차·Cube 대칭·family bootstrap을 포함한다. MLP optimizer 자체를 별도 구현으로 재학습한 것은 아니며 저장된 가중치와 inference를 검증했다. 겹치는 학습 subset의 변동과 평가 family의 불확실성을 서로 다른 결과로 보고한다.','',
    '**추가 B 전체는 아직 완료되지 않았다.** 이 절은 절대 상태 capture의 재현성이다. 같은 180개 모델로 실제 intervention 차이와 예측 차이를 읽는 F4–F5 반복이 남아 있다. 절대 위치 오차를 작은 차이의 해상도로 대체하지 않으며, 이 결과만으로 safe correction 표현이나 threshold를 선택하지 않는다.','',
    '근거: [설정](../configs/followup_probe_stability.json), [고정 capture 실행](../artifacts/followup_v1/probe_stability/capture_pipeline_lock.json), [capture 분석](../artifacts/followup_v1/probe_stability/capture_analysis.json), [독립 감사](../artifacts/followup_v1/probe_stability/capture_raw_audit.json), [모든 반복의 지표](../artifacts/followup_v1/probe_stability/capture_metrics.csv).','<!-- END PROBE STABILITY CAPTURE -->']
    section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();start='<!-- BEGIN PROBE STABILITY CAPTURE -->';end='<!-- END PROBE STABILITY CAPTURE -->'
    if start in s:i=s.index(start);j=s.index(end,i)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- BEGIN DIRECTION UTILITY -->');s=s[:i]+section+'\n\n'+s[i:]
    doc.write_text(s);write_json(OUT/'capture_report_evidence.json',{'source_sha256':sha256(Path(__file__)),'raw_audit_sha256':sha256(OUT/'capture_raw_audit.json'),'analysis_sha256':sha256(OUT/'capture_analysis.json'),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()});print('REPORTED capture repetitions; effects pending',flush=True)
if __name__=='__main__':main()
