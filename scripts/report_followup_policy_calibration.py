"""Report completed policy calibration after independent dense-trace validation."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from zj_wam.io import resolve, sha256, write_json

BASE = resolve('artifacts/followup_v1/policy_refinement')
read = lambda p: json.loads(p.read_text())


def main():
    audit = read(BASE/'final_validation_calibration/raw_audit.json')
    assert audit['validation_only'] and audit['episodes'] == 72 and audit['local_blocks'] == 2520
    assert audit['parent_raw_audit_sha256'] == sha256(BASE/'calibration/raw_audit.json')
    assert audit['source_sha256'] == sha256(resolve('scripts/analyze_followup_policy_final.py'))
    assert max(audit['max_error'].values()) < 1e-8
    analyses = {s: read(BASE/s/'analysis.json') for s in ['development', 'calibration']}
    methods = ['controller_pusher','lewm_global','vision_contact','vision_adaptive','gaussian_action']
    names = ['Controller','LeWM','영상 접촉 모델','영상 적응 모델','Gaussian action']
    rows = []
    for method, name in zip(methods,names):
        values=[]
        for split in analyses:
            a=analyses[split]
            assert not a['incomplete_seeds'] and len(a['completed_paired_seeds']) == 12
            r=next(x for x in a['local_summary'] if x['method']==method)
            values += [f"{100*r['joint']['mean']:.1f}%", f"{100*r['risk']['mean']:.1f}%"]
            e=pd.read_csv(BASE/split/'evaluation.csv'); e=e[e.method==method]
            assert len(e)==12 and e.active.all() and (e.object_error>1).all() and not e.joint.any()
        rows.append('| '+name+' | '+' | '.join(values)+' |')
    block='''<!-- BEGIN POLICY CALIBRATION COMPLETE -->
## 정책 calibration: 국소 개선은 반복되지만 전체 궤적 보존은 실패

별도 초기 상태 12개에서 6개 방법의 **72개 실행·2,520개 국소 구간**을 완료했다. 기존 expert calibration으로 정한 수정 규칙을 그대로 적용했으며, 이 데이터에서 새 규칙을 맞추지 않았다. 독립 분석기로 100Hz 물체 궤적, polygon coverage, 실제 pusher jerk와 국소 비교를 재계산했다. 개발용 12개 상태와도 분리했다.

**다섯 수정 방법 모두 calibration 12개 실행에서 전체 물체 궤적 보존에 실패했고, 보존과 전체 jerk 10% 감소의 동시 달성률은 0/12였다.** Development에서도 각 방법이 0/12였으므로 누적 보존 문제는 두 상태 집합에서 관측되었다. 방법별로 같은 초기 상태를 공유하므로 60개의 독립 실패 사례로 세지 않는다. 0/12라는 관측만으로 모든 상태에서 불가능하다고 결론 내리지 않는다.

국소 비교는 각 방법이 실제로 도달한 현재 상태에서 수정을 하지 않은 8행동 실행과 비교한다. 전체 비교는 같은 초기 상태에서 독립적으로 실행한 nominal 정책의 **동일 시각 물체 위치·회전 궤적**과 비교한다. 따라서 국소 통과율은 원래 전체 궤적의 보존율이 아니다. 위치 2px·회전 2도 및 coverage 감소 0.02의 허용 조건을 유지했다.

| 방법 | 개발: 국소 보존＋jerk 10% 감소 | 개발: 수정 후 국소 위반율 | Calibration: 국소 보존＋jerk 10% 감소 | Calibration: 수정 후 국소 위반율 |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

Calibration에서 모든 수정 방법의 평균 전체 jerk는 nominal보다 약 10.1–14.5% 줄었지만, 원래 물체 궤적을 보존하지 못했으므로 연구의 목표를 달성한 개선으로 세지 않는다. 목표 영역 도달 성공도 물체 궤적 보존과 다른 지표다. 이 결과는 **현재 국소 승인 규칙을 반복 적용하는 방법의 한계**를 보여주며, 새로운 종료 상태 제약이나 다른 정책 결합법의 실패를 미리 뜻하지 않는다.

Calibration은 최종 test가 아니다. 별도 잠금으로 고정한 12개 초기 상태의 최종 정책 평가와, 계산 지연 동안의 실제 움직임을 포함하는 평가는 별도로 진행한다.

근거: [calibration 분석](../artifacts/followup_v1/policy_refinement/calibration/analysis.json), [원시 감사](../artifacts/followup_v1/policy_refinement/calibration/raw_audit.json), [독립 재계산](../artifacts/followup_v1/policy_refinement/final_validation_calibration/raw_audit.json).
<!-- END POLICY CALIBRATION COMPLETE -->
'''
    path=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md'); doc=path.read_text()
    begin='<!-- BEGIN POLICY CALIBRATION COMPLETE -->'; end='<!-- END POLICY CALIBRATION COMPLETE -->'
    if begin in doc:doc=doc[:doc.index(begin)]+block+doc[doc.index(end)+len(end):]
    else:
        anchor='<!-- BEGIN POLICY FIRST CORRECTION -->';assert anchor in doc;doc=doc.replace(anchor,block+'\n'+anchor,1)
    path.write_text(doc)
    paths=[BASE/s/n for s in ['development','calibration'] for n in ['analysis.json','raw_audit.json','evaluation.csv']]+[BASE/'final_validation_calibration/raw_audit.json']
    write_json(BASE/'final_validation_calibration/report_evidence.json',{'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in paths},'source_sha256':sha256(Path(__file__)),'scope':'Development and calibration only; no final test claim.'})
    print('REPORTED completed policy calibration')

if __name__ == '__main__':main()
