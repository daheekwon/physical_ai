"""Report nested family readout learning curves with explicit sampling limits."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_learning_curve';read=lambda p:json.loads(p.read_text())


def main():
    protocol=read(OUT/'protocol.json');audit=read(OUT/'raw_audit.json');a=read(OUT/'analysis.json');assert audit['new_learners']==152 and audit['metric_rows']==456 and audit['analysis_sha256']==sha256(OUT/'analysis.json')
    for name,h in protocol['inputs'].items():assert sha256(resolve(name))==h,name
    for p in OUT.glob('*/*/*.json'):
        r=read(p)
        for name,h in r['files'].items():assert sha256(p.parent/name)==h,name
    chosen={'pusht':['dino_spatial_quadrants','dino_projected_spatial_quadrants','lewm_spatial'],'ogb-cube':['dino_spatial_quadrants','dino_projected_patch_mean','lewm_frame']};table=[];comparisons=[]
    for env,reps in chosen.items():
        target='raw' if env=='pusht' else 'symmetry';pos='position_px' if env=='pusht' else 'position_m';rot='rotation_rad' if env=='pusht' else 'rotation_mod_cube_symmetry_rad';scale=1 if env=='pusht' else 1000
        for rep in reps:
            cells=[]
            for fraction in [.25,.5,1.]:
                r=next(r for r in a['metrics'] if r['environment']==env and r['split']=='calibration' and r['representation']==rep and r['fraction']==fraction and r['readout']=='ridge' and r['target']==target)
                cells.append(f"{r[pos]['mean']*scale:.2f} / {np.rad2deg(r[rot]['mean']):.2f}")
            table.append(f"| {env} | {rep} | "+' | '.join(cells)+' |')
        pair=('dino_projected_spatial_quadrants','lewm_spatial') if env=='pusht' else ('dino_projected_patch_mean','lewm_frame')
        for fraction in [.25,.5,1.]:
            folder=BASE/'dinowm_readouts'/env if fraction==1 else OUT/env/f'fraction_{fraction:.2f}'
            with np.load(folder/(pair[0]+'__'+target+'__ridge__calibration.npz')) as x,np.load(folder/(pair[1]+'__'+target+'__ridge__calibration.npz')) as y:
                np.testing.assert_array_equal(x['sample_ids'],y['sample_ids']);np.testing.assert_array_equal(x['families'],y['families']);f=pd.DataFrame({'initial_pose_family':x['families'],'position_difference':scale*(x[pos]-y[pos]),'rotation_difference_deg':np.rad2deg(x[rot]-y[rot])})
                comparisons.append({'task':env,'fraction':fraction,'pair':pair,**{k:mean_ci(f,k) for k in ['position_difference','rotation_difference_deg']}})
    write_json(OUT/'paired_comparisons.json',{'comparisons':comparisons,'scope':'Single nested training-family sample; evaluation-family bootstrap conditional on that training set. No claim of training-subset uncertainty or general sample efficiency.'})
    block=f'''<!-- BEGIN DINOWM LEARNING CURVE -->
## Readout 학습량 대조: 개선은 있지만 task에 따라 비교가 달라짐

전체 학습 family 중 고정된 무작위 순서의 25%·50%·100%를 사용하는 중첩 대조를 수행했다. PushT는 25·50·100개 family(500·1,000·2,000개 label), Cube는 16·32·64개 family(128·256·512개 label)다. Development와 calibration family는 그대로 유지했다. 모든 표현에 같은 subset을 제공하고 각 규모에서 같은 alpha 탐색 예산을 사용했다. 100% 결과는 앞서 감사한 readout을 그대로 재사용했다.

**PushT는 학습량이 늘수록 DINO·LeWM 공간 ridge가 모두 개선됐지만, 이번 범위에서 DINO가 LeWM의 정밀도를 넘지 않았다. Cube는 작은 학습량에서 DINO가 유리한 비교도 있었고, 전체 학습량에서는 관계가 달라졌다.** 예를 들어 입력 차원 192를 맞춘 Cube 위치 오차는 16개 family에서 DINO 51.94mm·LeWM 92.16mm, 64개 family에서 DINO 34.01mm·LeWM 18.00mm였다. 따라서 학습량 하나의 결과를 표현 전반의 우열로 확대하지 않는다. 아래는 calibration 평균 위치/회전 오차이며 PushT는 px/도, Cube는 mm/대칭 회전 도다.

| Task | Ridge 표현 | 학습 25% | 학습 50% | 학습 100% |
| --- | --- | --- | --- | --- |
{chr(10).join(table)}

전체 17개 표현에 ridge·MLP64·평균값·label-shuffle 대조를 유지했고 Cube raw 회전은 ridge로 별도 평가했다. 새 learner 152개와 기존 76개를 합쳐 총 456개 split 평가행을 기록했다. 새 alpha 탐색 250개, 저장 모델의 독립 행렬 연산 예측, 물리 회전 오차, family 신뢰구간, 선택된 ridge의 재학습 계수를 검사했다. 새 학습 수렴 경고는 {len(a['warnings'])}개다. 원시 평가 파일과 학습 subset ID도 남겼다.

**학습 family 추출은 한 개 seed의 중첩 순서다.** 제공한 paired 신뢰구간은 그 학습 집합에 조건부인 평가 family 변동이며, 학습 subset을 바꿨을 때의 변동까지 포함하지 않는다. Cube의 작은 표본에서 보인 역전을 일반적인 sample efficiency 이점으로 주장하려면 추가 subset seed에서 재현해야 한다. 학습량 증가 후에도 절대 상태 오차는 주 물리 허용량보다 크며, 실제 intervention 차이를 읽는 해상도는 다음 F4 진단에서 별도로 확인한다.

근거: [학습량 프로토콜](../artifacts/followup_v1/dinowm_learning_curve/protocol.json), [전체 결과](../artifacts/followup_v1/dinowm_learning_curve/analysis.json), [차원을 맞춘 paired 비교](../artifacts/followup_v1/dinowm_learning_curve/paired_comparisons.json), [원시 검증](../artifacts/followup_v1/dinowm_learning_curve/raw_audit.json).
<!-- END DINOWM LEARNING CURVE -->
'''
    doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();start='<!-- BEGIN DINOWM LEARNING CURVE -->';end='<!-- END DINOWM LEARNING CURVE -->'
    if start in s:i=s.index(start);j=s.index(end,i)+len(end);s=s[:i]+block.rstrip()+s[j:]
    else:anchor='<!-- END DINOWM CAPTURE READOUTS -->';i=s.index(anchor)+len(anchor);s=s[:i]+'\n\n'+block+s[i:]
    doc.write_text(s);write_json(OUT/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{name:sha256(OUT/name) for name in ['protocol.json','analysis.json','raw_audit.json','paired_comparisons.json']},'document':str(doc.relative_to(resolve('.'))),'section_sha256':hashlib.sha256(block.rstrip().encode()).hexdigest()});print('REPORTED learning curves',flush=True)
if __name__=='__main__':main()
