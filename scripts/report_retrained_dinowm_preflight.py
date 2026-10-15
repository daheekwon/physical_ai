"""Publish completed restricted-loader CPU implementation verification."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1');OUT=BASE/'dinowm_retrained_preflight';read=lambda p:json.loads(p.read_text())


def main():
    a=read(OUT/'audit.json');assert len(a['models'])==2
    for p,h in a['inputs'].items():assert sha256(resolve(p))==h,p
    rows=[]
    for r in a['models']:
        best=min(x['relative_error'] for x in r['directional_derivative_sweep'])
        rows.append(f"| {r['environment']} | {r['native_one_step_max_error']:.3g} | {r['native_two_step_max_error']:.3g} | {100*best:.3f}% |")
    block=f'''<!-- BEGIN DINOWM CPU PREFLIGHT -->
## DINO-WM 사전 실행 검증: CPU FP32에서 native rollout 재현

별도 재학습 PushT·Cube DINO-WM을 실제 저장된 개발용 history 한 개씩으로 실행했다. DINOv2 ViT-S/14의 공식 소스를 commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`으로 고정하고 공식 배포 가중치를 확보했다. Encoder 가중치의 로컬 SHA-256은 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`다. 이는 이번 평가의 고정 식별값이며 checkpoint 제작자가 사용한 encoder 소스 revision의 재현을 뜻하지 않는다.

영상 변환, 3-frame history·5-action grouping, action 정규화를 확인했다. 관측 영상 224px는 native 모델에서 196px로 resize되고, frame당 **196개 patch × 384차원**으로 표현된다. 모든 가중치를 고정한 상태에서 native `VWorldModel.rollout`과 adapter 출력을 비교했다.

| 모델 | 1회 예측 최대 절대 차이 | 2회 예측 최대 절대 차이 | Step sweep 중 최소 미분 상대 오차 |
| --- | --- | --- | --- |
{chr(10).join(rows)}

2회 예측 비교는 같은 미래 action을 두 번 연결한 **수치 검증 입력**이며 실제 두 구간의 예측 정확성 결과가 아니다. 자동미분은 한 무작위 출력 projection·한 action 방향에 대한 중앙 유한차분으로 확인했다. 양 모델 모두 입력 gradient가 유한하고 0이 아니었으며 반복 출력은 정확히 같았다. Step에 따라 유한차분 오차가 달라지므로 작은 값 하나를 모든 방향의 정밀도 보증으로 해석하지 않는다.

No-proprioception 조건은 **외부에서 실제 robot state를 공급하지 않고 native adapter의 상수 zero 입력을 유지**하는 방식이다. 외부 proprioception 인자를 바꿔도 이 adapter의 출력은 동일했다. 이는 영상에서 pusher 정보를 읽지 못한다거나 모델이 로봇 상태에 물리적으로 무관하다는 뜻은 아니다.

이번 실행은 **CPU FP32**이며 배포 설정의 GPU bfloat16을 재현한 결과가 아니다. 모델을 일반 pickle로 실행하는 경로는 자동 승인 검토에서 거부되어 사용하지 않았다. 대신 해시가 고정된 검토 완료 클래스만 명시적으로 허용한 `weights_only=True` 로딩으로 실행했다. 무제한 pickle 로딩 fallback은 없다.

여기까지는 구현·입력·미분 검증이다. 실제 물체 정보를 읽는 능력, 작은 물체 변화 구분, intervention 효과 예측, 수정 승인 성능은 별도 평가가 필요하다. 다음 진단은 두 task의 개발용 family 3개씩에서 이미 실행된 후보 전체를 대상으로 **실제 영상의 latent 변화와 예측 latent 변화를 구분**한다. 5행동 끝의 물체 변화와 8행동 보존 오차를 따로 비교하며, 전체 patch 변화가 pusher 변화에 반응하는지도 기록한다.

근거: [사전 감사](../artifacts/followup_v1/dinowm_retrained_preflight/audit.json), [고정 encoder 소스](../artifacts/followup_v1/protocol/dinowm_retrained_inventory/encoder_source.json), [encoder 가중치](../artifacts/followup_v1/protocol/dinowm_retrained_inventory/encoder_weights.json), [실제 intervention 진단 조건](../configs/followup_dinowm_effect_pilot.json).
<!-- END DINOWM CPU PREFLIGHT -->
'''
    p=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=p.read_text();begin='<!-- BEGIN DINOWM CPU PREFLIGHT -->';end='<!-- END DINOWM CPU PREFLIGHT -->'
    if begin in s:s=s[:s.index(begin)]+block+s[s.index(end)+len(end):]
    else:
        anchor='<!-- END DINOWM RETRAINED ASSETS -->';assert anchor in s;s=s.replace(anchor,anchor+'\n\n'+block,1)
    p.write_text(s);write_json(OUT/'report_evidence.json',{'audit_sha256':sha256(OUT/'audit.json'),'source_sha256':sha256(Path(__file__)),'scope':'CPU implementation audit only, no accuracy or safety claim.'})
    print('REPORTED DINO CPU preflight',flush=True)
if __name__=='__main__':main()
