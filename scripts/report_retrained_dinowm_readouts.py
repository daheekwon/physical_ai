"""Publish audited capture diagnostics without claiming intervention fidelity."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_readouts';read=lambda p:json.loads(p.read_text())


def main():
    audit=read(OUT/'raw_audit.json');done=read(OUT/'pipeline_complete.json');assert done['raw_audit_sha256']==sha256(OUT/'raw_audit.json') and audit['counts']['evaluations']==152
    for name,h in audit['inputs'].items():assert sha256(OUT/name)==h,name
    rows=read(OUT/'analysis.json')['metrics'];selection=read(OUT/'selection.json')['readouts'];table=[];controls=[];paired=[]
    for env in ['pusht','ogb-cube']:
        kind='raw' if env=='pusht' else 'symmetry';position='position_px' if env=='pusht' else 'position_m';angle='rotation_rad' if env=='pusht' else 'rotation_mod_cube_symmetry_rad';scale=1 if env=='pusht' else 1000
        for r in rows:
            if r['environment']!=env or r['split']!='calibration' or r['target']!=kind or r['readout']!='ridge':continue
            mlp=next(x for x in rows if all(x[k]==r[k] for k in ['environment','split','target','representation']) and x['readout']=='mlp64')
            joint='joint_pose' if env=='pusht' else 'joint_pose_symmetry'
            table.append(f"| {env} | {r['representation']} | {r['input_dimension']} | {scale*r[position]['mean']:.2f} / {np.rad2deg(r[angle]['mean']):.2f} | {scale*mlp[position]['mean']:.2f} / {np.rad2deg(mlp[angle]['mean']):.2f} | {100*r[joint]['mean']:.2f}% |")
            for learner in ['mean','label_shuffle']:
                c=next(x for x in rows if all(x[k]==r[k] for k in ['environment','split','target','representation']) and x['readout']==learner)
                controls.append({'environment':env,'representation':r['representation'],'readout':learner,'position':scale*c[position]['mean'],'rotation_deg':float(np.rad2deg(c[angle]['mean']))})
        pairs=[('dino_projected_patch_mean','lewm_frame'),('dino_projected_history_patch_mean','lewm_history')]
        if env=='pusht':pairs.append(('dino_projected_spatial_quadrants','lewm_spatial'))
        for d,l in pairs:
            with np.load(OUT/env/(d+'__'+kind+'__ridge__calibration.npz')) as x,np.load(OUT/env/(l+'__'+kind+'__ridge__calibration.npz')) as y:
                np.testing.assert_array_equal(x['sample_ids'],y['sample_ids']);np.testing.assert_array_equal(x['families'],y['families'])
                f=pd.DataFrame({'initial_pose_family':x['families'],'position_difference':scale*(x[position]-y[position]),'rotation_difference_deg':np.rad2deg(x[angle]-y[angle])})
                paired.append({'environment':env,'dino':d,'lewm':l,**{k:mean_ci(f,k) for k in ['position_difference','rotation_difference_deg']}})
    extra={'paired_dimension_matched_ridge':paired,'controls':controls,'scope':'Exploratory family-paired capture diagnostics on calibration families; not final test or causal encoder-training comparison.'};write_json(OUT/'capture_comparisons.json',extra)
    pts=[]
    for r in paired:
        p=r['position_difference'];a=r['rotation_difference_deg'];pts.append(f"| {r['environment']} | {r['dino']} − {r['lewm']} | {p['mean']:.2f} [{p['ci95'][0]:.2f}, {p['ci95'][1]:.2f}] | {a['mean']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}] |")
    warnings=sum(len(r['warnings']) for r in selection.values());ca=read(BASE/'dinowm_capture/raw_audit.json')
    block=f'''<!-- BEGIN DINOWM CAPTURE READOUTS -->
## DINO-WM 물체 상태 capture: 같은 label·probe 예산의 LeWM 대조

**같은 데이터와 readout 절차에서 DINO의 물체 상태 decoding이 LeWM보다 전반적으로 좋아지는 결과는 확인되지 않았다.** PushT의 공간 ridge는 DINO native에서 평균 14.79px·14.52도, LeWM에서 8.20px·7.14도였다. Cube의 DINO native 공간 표현은 19.92mm·대칭 회전 19.16도, LeWM frame은 18.00mm·19.59도였다. Cube의 두 표현은 차원·집계 방식이 다르므로 이 숫자로 동등성이나 우위를 확정하지 않는다. 절대 상태 오차는 두 모델 모두 주 물리 허용량보다 크다. **이 결과만으로 작은 intervention 차이도 읽지 못한다고 결론낼 수는 없으며, 같은 frozen readout의 실제 차이와 예측 차이를 다음으로 평가해야 한다.**

같은 기존 probe family와 timestamp에서 DINO 특징을 수집했다. PushT는 학습 100·개발 35·calibration 35개 family, 각 20개 관측이며 Cube는 64·24·24개 family의 네 phase에서 현재·미래 관측을 사용했다. 총 618개 source 기록·4,296개 label이다. 모든 split은 family가 분리되어 있고, 기존 최종 test를 사용하지 않았다. Frozen encoder/predictor에 별도 지도학습 readout을 붙이는 진단이며 world model 가중치는 변경하지 않았다. DINO는 별도 재학습 no-proprio 자산과 CPU FP32 설정이다.

Native DINO의 채널 384를 label·데이터를 사용하지 않는 고정 직교 투영으로 192로 줄인 대조도 포함했다. 같은 변환을 공간별·history별로 반복해 LeWM과 입력 차원을 맞췄다. LeWM도 동일 scaler·ridge alpha 탐색·MLP recipe로 다시 학습했다. Alpha는 development 오차로만 선택했다. Calibration 결과에서 새 winner를 선택하지 않고 사전에 정한 모든 표현을 보고한다. Cube raw 회전 target과 대칭 canonical target을 모두 학습했으며 아래 Cube 표는 후자다. 대칭 진단은 물리 보존의 full SO(3) 2도 조건을 바꾸지 않는다.

아래 오차는 calibration의 평균 위치/회전 오차다. PushT는 px/도, Cube는 mm/대칭을 고려한 도 단위다. 동시 pose 통과율은 해당 관측에서 2px·2도 또는 2mm·대칭 2도 이내인 비율이며, **수정 승인율·실제 궤적 보존율·U가 아니다.**

| Task | 표현 | 차원 | Ridge 위치 / 회전 | MLP64 위치 / 회전 | Ridge 동시 pose 통과율 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(table)}

표현 차원을 맞춘 ridge의 paired 오차 차이는 다음과 같다. 양수는 DINO 오차가 더 크다는 뜻이다. 초기 상태 family 단위 bootstrap 95% 구간이며 다중 비교를 보정한 확증 결과가 아니다. Random projection이 정보를 버릴 수 있으므로 native 결과도 위 표에 유지했다. 데이터·학습 방식이 다른 checkpoint의 차이를 architecture 하나의 인과 효과로 해석하지 않는다.

| Task | DINO − LeWM | 위치 오차 차이 [95% 구간] | 회전 오차 차이 [95% 구간], 도 |
| --- | --- | --- | --- |
{chr(10).join(pts)}

평균값·label-shuffle 대조, 전체 오차 신뢰구간, Cube raw 회전 결과는 원본 분석에 포함했다. 작은 MLP는 ridge의 정밀도 문제를 해결하지 못했지만 한 가지 고정 capacity·학습 recipe의 결과다. 최적의 nonlinear readout이 없다는 결론은 아니다. 이번 본 학습의 수렴 경고는 {warnings}개다. 학습량 25%·50%·100% 곡선은 이 전체 학습량 비교와 별도 후속 진단이다.

**원시 감사:** 25개 표현·target 구성, 76개 learner, 152개 평가, 65,760개 관측–learner 평가행을 재계산했다. 125개 alpha 탐색, shuffled-label ridge 재학습, training-only scaler, 저장된 모델 배열의 예측, 회전 오차와 family 신뢰구간이 재현됐다. 최대 예측 재현 차이는 {audit['max_error']['inference']:.3g}, 원시 label·ID 정합성은 정확히 일치했다. MLP optimizer 전체를 두 번 학습한 감사는 아니며 sklearn 예측과 저장 배열의 일치는 fit 시 검사했다.

수집의 첫 전체 감사는 7개 PushT 기록에서 FP32 pooling의 output-relative 허용량을 넘겨 중단됐다. 해당 기록 140개 관측을 원본 영상에서 재인코딩한 mean·quadrant 특징은 저장값과 정확히 같았다. FP64 pooling 항등식과 원시 token의 절댓값을 이용한 FP32 합산 오차 한계를 확인한 별도 v2 감사로 재개했다. 물리 label·특징·기존 실패 로그·원래 프로토콜은 수정하지 않았다. 학습은 이 감사가 통과한 뒤 시작했다.

근거: [수집 감사](../artifacts/followup_v1/dinowm_capture/raw_audit.json), [감사 복구 기록](../artifacts/followup_v1/dinowm_capture/audit_recovery_protocol.json), [학습 프로토콜](../artifacts/followup_v1/dinowm_readouts/protocol.json), [전체 capture 결과](../artifacts/followup_v1/dinowm_readouts/analysis.json), [차원 대조와 대조군](../artifacts/followup_v1/dinowm_readouts/capture_comparisons.json), [독립 원시 감사](../artifacts/followup_v1/dinowm_readouts/raw_audit.json).
<!-- END DINOWM CAPTURE READOUTS -->
'''
    doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');text=doc.read_text();start='<!-- BEGIN DINOWM CAPTURE READOUTS -->';end='<!-- END DINOWM CAPTURE READOUTS -->'
    if start in text:i=text.index(start);j=text.index(end,i)+len(end);text=text[:i]+block.rstrip()+text[j:]
    else:
        anchor='<!-- END DINOWM EFFECT PILOT -->';assert anchor in text;i=text.index(anchor)+len(anchor);text=text[:i]+'\n\n'+block+text[i:]
    doc.write_text(text)
    write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{name:sha256(OUT/name) for name in ['protocol.json','raw_audit.json','analysis.json','selection.json','capture_comparisons.json','pipeline_complete.json']},'document':str(doc.relative_to(resolve('.'))),'section_sha256':__import__('hashlib').sha256(block.rstrip().encode()).hexdigest()})
    print('REPORTED DINO capture',len(table),'rows',len(paired),'paired contrasts',flush=True)
if __name__=='__main__':main()
