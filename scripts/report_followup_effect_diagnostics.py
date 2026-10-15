"""Publish complete, independently audited DINO effects or repeated-probe effects.

Smoke mode verifies report construction and writes a preview only. Full reports
require the pipeline completion record, source lock, and matching raw audit.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve, sha256, write_json

BASE = resolve('artifacts/followup_v1')
read = lambda p: json.loads(p.read_text())
PAIRS = [('pusht', 'lewm_frame'), ('pusht', 'lewm_history'),
         ('pusht', 'dino_projected_patch_mean'), ('pusht', 'dino_projected_spatial_quadrants'),
         ('ogb-cube', 'lewm_frame'), ('ogb-cube', 'dino_projected_patch_mean')]


def verify(kind, smoke):
    folder = 'dinowm_effect_readouts' if kind == 'dino' else 'probe_stability_effects'
    out = BASE / (folder + ('_smoke' if smoke else ''))
    audit = read(out / 'raw_audit.json')
    assert audit['smoke'] == smoke and audit['states'] == (4 if smoke else 402)
    for name in ['analysis', 'resolution', 'protocol']:
        assert audit[name + '_sha256'] == sha256(out / (name + '.json'))
    auditor = 'audit_retrained_dinowm_effect_readouts.py' if kind == 'dino' else 'audit_followup_probe_stability_effects.py'
    assert audit['source_sha256'] == sha256(resolve('scripts/' + auditor))
    if not smoke:
        done = read(out / 'pipeline_complete.json')
        assert done['states'] == 402 and done['raw_audit_sha256'] == sha256(out / 'raw_audit.json')
        assert done['lock_sha256'] == sha256(out / 'pipeline_lock.json')
        for name, digest in read(out / 'pipeline_lock.json')['inputs'].items():
            assert sha256(resolve(name)) == digest, name
        source = BASE / 'dinowm_interventions'
        complete = read(source / 'pipeline_complete.json')
        assert complete['states'] == 402 and complete['raw_audit_sha256'] == sha256(source / 'raw_audit.json')
    return out, audit, read(out / 'analysis.json')


def fmt(value, scale=1):
    return '—' if value is None else f'{value * scale:.3f}'


def interval(record, scale=1):
    if record['mean'] is None:
        return '—'
    lo, hi = record['ci95']
    return f"{record['mean'] * scale:+.3f} [{lo * scale:+.3f}, {hi * scale:+.3f}]"


def dino_section(out, audit, analysis):
    rows = analysis['summary']
    lines = ['## DINO-WM의 실제 효과 해상도와 미래 효과 예측', '',
             f"공통 PushT 210개·Cube 192개 상태의 **402개 상태에서 {audit['candidate_method_rows_including_inactive']:,}개 후보–readout–대조군 행과 {audit['decoded_array_checks']:,}개 decoded 배열**을 검증했다. 기존 물리 후보를 재사용한 개발·calibration 진단이며, 별도 최종 test나 smoothing 선택 성능은 아니다.", '',
             'DINO-WM은 별도로 검증한 third-party retrain이다. 원래 공개 모델의 재현 결과로 부르지 않는다. 모델 가중치와 학습된 readout은 고정하고 실제 nominal/modified endpoint와 예측 endpoint를 같은 readout으로 읽었다. Native patch mean·spatial quadrants·history와 각각의 고정 192차원 투영을 검사했으며 ridge·작은 MLP·물리 대조군의 전체 결과를 원본에 제공한다.', '',
             '아래는 development의 사전 지정 ridge 구성이다. 위치 오차는 PushT px, Cube mm이며 회전은 deg다. 각 셀은 **위치 / 회전**이다. 실제 효과 오차는 실제 이미지를 읽는 측정 경로의 오차, 예측 효과 오차는 모델의 미래 출력까지 포함한 오차다. Cube 표는 symmetry readout 진단이며, 실제 보존 판정은 full 회전 2도 기준을 유지한다. Raw 회전 readout 결과도 원본에 별도로 남겼다.', '',
             '| 환경·표현 | 실제 효과 오차 | 예측 효과 오차 | 측정 해상도 초과 효과 비율, % |',
             '| --- | --- | --- | --- |']
    selected = []
    reps = ['dino_patch_mean', 'dino_spatial_quadrants', 'dino_history_patch_mean',
            'dino_projected_patch_mean', 'dino_projected_spatial_quadrants', 'dino_projected_history_patch_mean']
    for env in ['pusht', 'ogb-cube']:
        for rep in reps:
            target = 'raw' if env == 'pusht' else 'symmetry'
            select = dict(environment=env, representation=rep, target=target, readout='ridge', split='development')
            def pick(method):
                found = [r for r in rows if all(r[k] == v for k, v in select.items()) and r['method'] == method and r['phase'] == 'all']
                assert len(found) == 1, (select, method)
                return found[0]
            real, pred = pick('real_encoded'), pick('predicted')
            scale = 1 if env == 'pusht' else 1000
            def err(r):
                return fmt(r['position_effect_error']['mean'], scale) + ' / ' + fmt(r['rotation_effect_error']['mean'], 180 / np.pi)
            support = fmt(pred['position_resolvable_fraction']['mean'], 100) + ' / ' + fmt(pred['rotation_resolvable_fraction']['mean'], 100)
            lines.append(f'| {env} / {rep} | {err(real)} | {err(pred)} | {support} |')
            selected.append((select, scale))
    lines += ['', '예측 효과 오차에서 zero-effect 대조군 오차를 뺀 paired 차이는 아래와 같다. 음수면 해당 물리 단위의 평균 오차가 더 작다. 각 수치의 대괄호는 family bootstrap 95% 구간이다. 여러 표현을 탐색한 진단 비교이므로 유리한 셀 하나만으로 모델 우위를 확정하지 않는다.', '',
              '| 환경·표현 | 위치 오차 차이 [95% 구간] | 회전 오차 차이 [95% 구간] |',
              '| --- | --- | --- |']
    for select, scale in selected:
        found = [r for r in analysis['paired_controls'] if all(r[k] == v for k, v in select.items()) and r['control'] == 'zero_effect']
        assert len(found) == 1
        r = found[0]
        lines.append(f"| {select['environment']} / {select['representation']} | {interval(r['position_difference'], scale)} | {interval(r['rotation_difference'], 180 / np.pi)} |")
    lines += ['', '해상도는 calibration의 실제 효과 측정 오차에서 상태 가중치를 같게 한 90분위수로 정했다. 그 해상도보다 큰 실제 효과에 한해서 방향 일치를 해석하며, 작은 효과도 물리 보존·위반 통계에서 제외하지 않는다. 표현마다 해상도와 방향 평가 표본이 다르므로 방향 정확도만으로 표현의 순위를 정하지 않는다.', '',
              'Zero-effect·현재 상태 유지·크기와 방향 조건을 맞춘 wrong-action을 모두 제공한다. Wrong-action 대조가 없는 후보는 지원 불가로 남겼으며 전체 사례에 그 비교가 가능하다고 가정하지 않았다. 작은 물체 변화가 많은 후보 bank에서는 zero-effect 자체가 강한 기준선이므로, 절대 pose를 읽는 능력과 작은 intervention 효과 예측을 구분한다.', '',
              '감사는 token에서 readout 추론, 물리 endpoint 차이, Cube 회전 변환, wrong-action 대응, calibration 해상도, 상태별 집계와 family 구간을 재계산했다. 이 결과가 LeWM 대비 더 유용한 smoothing으로 이어지는지는 같은 후보의 승인·선택 비교와 독립 test에서 별도로 검증해야 한다.', '',
              '근거: [분석](../artifacts/followup_v1/dinowm_effect_readouts/analysis.json), [해상도](../artifacts/followup_v1/dinowm_effect_readouts/resolution.json), [원시 감사](../artifacts/followup_v1/dinowm_effect_readouts/raw_audit.json), [고정 실행](../artifacts/followup_v1/dinowm_effect_readouts/pipeline_complete.json).']
    return '\n'.join(lines)


def stability_section(out, audit, analysis):
    repro = read(out / 'reproducibility.json')
    assert repro['states'] == audit['states'] and repro['models'] == 180
    for key, name in [('raw_audit', 'raw_audit'), ('analysis', 'analysis'), ('resolution', 'resolution'), ('catalog', 'protocol')]:
        assert repro[key + '_sha256'] == sha256(out / (name + '.json'))
    assert repro['source_sha256'] == sha256(resolve('scripts/summarize_followup_probe_effect_stability.py'))
    frame = pd.read_csv(out / 'replicate_metrics.csv')
    for r in repro['variation']:
        g = frame
        for key in ['environment', 'split', 'representation', 'target', 'fraction', 'learner', 'method', 'metric']:
            g = g[g[key] == r[key]]
        assert len(g) == r['replicates'] == 3
        values = g.value.dropna()
        assert len(values) == r['available']
        if len(values):
            np.testing.assert_allclose([values.mean(), values.min(), values.max()], [r['mean'], r['min'], r['max']], rtol=1e-10, atol=1e-12)
    lines = ['## 추가 B 완료: 실제·예측 intervention 효과의 학습 반복 민감도', '',
             f"고정된 **180개 probe를 공통 402개 상태**에 적용하고 {audit['candidate_method_rows_including_inactive']:,}개 후보–probe–대조군 행을 감사했다. 앞선 capture 반복에 더해 실제 효과의 측정 가능성과 미래 효과 예측을 같은 학습 반복에서 확인했다. World model 재학습이나 새로운 물리 실행은 없다.", '',
             '아래는 세 개 중첩 subset seed로 학습 family의 50%를 사용한 ridge의 development 평균 효과 오차 **최소–최대**다. 각 셀은 위치 / 회전이며 PushT px·Cube mm·deg 단위를 사용한다. 이 범위는 학습 표본 선택에 따른 변동으로 신뢰구간이 아니다. 개별 probe의 평가 family 구간은 원본에 별도 제공한다.', '',
             '| 환경·표현 | 실제 효과 오차 범위 | 예측 효과 오차 범위 |', '| --- | --- | --- |']
    def ranges(env, rep, method, fraction, learner, axis):
        result = []
        for metric, scale in [('position_effect_error', 1 if env == 'pusht' else 1000), ('rotation_effect_error', 180 / np.pi)]:
            found = [r for r in repro['variation'] if r['environment'] == env and r['representation'] == rep and r['split'] == 'development' and r['method'] == method and r['fraction'] == fraction and r['learner'] == learner and r['variation_axis'] == axis and r['metric'] == metric]
            assert len(found) == 1
            r = found[0]
            result.append('—' if r['available'] == 0 else f"{r['min'] * scale:.3f}–{r['max'] * scale:.3f}")
        return ' / '.join(result)
    for env, rep in PAIRS:
        lines.append(f"| {env} / {rep} | {ranges(env, rep, 'real_encoded', .5, 'ridge', 'training_subset')} | {ranges(env, rep, 'predicted', .5, 'ridge', 'training_subset')} |")
    lines += ['', '다음은 같은 전체 학습 family를 사용한 작은 MLP의 세 초기화 범위다. 표본 선택의 변동과 optimizer 초기화의 변동을 섞지 않는다.', '',
              '| 환경·표현 | 실제 효과 오차 범위 | 예측 효과 오차 범위 |', '| --- | --- | --- |']
    for env, rep in PAIRS:
        lines.append(f"| {env} / {rep} | {ranges(env, rep, 'real_encoded', 1., 'mlp64', 'mlp_initialization')} | {ranges(env, rep, 'predicted', 1., 'mlp64', 'mlp_initialization')} |")
    lines += ['', '25%·50%·100% 학습량, 실제 endpoint·예측 endpoint·zero-effect·persistence·wrong-action, label-shuffle 및 평균값 대조를 유지했다. 전체 자료의 deterministic ridge는 한 번만 사용했다. 겹치는 subset 세 개를 독립 데이터셋 세 개로 세지 않으며, test에서 유리한 seed를 고르지 않았다.', '',
              'Cube는 symmetry readout의 재현성 진단이다. 실제 물체 궤적 보존은 full SO(3) 허용 오차로 평가하며 영상 대칭 때문에 물리 기준을 완화하지 않는다. 위치·회전별 해상도와 진단 가능한 효과 비율도 probe별로 기록했다. 서로 다른 진단 표본에서 얻은 방향 정확도를 동일 표본의 비교처럼 해석하지 않는다.', '',
              'LeWM frame과 DINO projected mean의 192차원 비교에서는 같은 subset·초기화를 짝지어 평균 오차 순위가 얼마나 반복되는지 별도로 기록했다. 그 횟수는 학습 반복의 기술 통계이며 독립적인 유의성 검정이 아니다. 모델의 학습 데이터와 구조도 다르므로 차원을 맞췄다는 사실만으로 순수 아키텍처 효과를 주장하지 않는다.', '',
              '**추가 B의 capture–실제 효과–예측 효과 반복은 완료됐다.** 이 단계는 진단의 안정성을 확인한 것이며, 유용한 correction 승인이나 smoothing 성능의 우위를 확정한 것은 아니다. 표현을 실제 선택기에 사용하려면 규칙을 고정하고 새 calibration·독립 평가로 넘어간다.', '',
              '근거: [반복 분석](../artifacts/followup_v1/probe_stability_effects/reproducibility.json), [개별 반복 지표](../artifacts/followup_v1/probe_stability_effects/replicate_metrics.csv), [원시 감사](../artifacts/followup_v1/probe_stability_effects/raw_audit.json), [전체 효과 분석](../artifacts/followup_v1/probe_stability_effects/analysis.json).']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['dino', 'stability'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out, audit, analysis = verify(args.kind, args.smoke)
    body = dino_section(out, audit, analysis) if args.kind == 'dino' else stability_section(out, audit, analysis)
    tag = 'DINO EFFECT READOUTS COMPLETE' if args.kind == 'dino' else 'PROBE STABILITY EFFECTS COMPLETE'
    section = f'<!-- BEGIN {tag} -->\n{body}\n<!-- END {tag} -->'
    if args.smoke:
        (out / 'report_preview.md').write_text('SMOKE FORMAT CHECK ONLY — values are not a full research report.\n\n' + section)
        write_json(out / 'report_format_audit.json', {'smoke': True, 'source_sha256': sha256(Path(__file__)), 'raw_audit_sha256': sha256(out / 'raw_audit.json'), 'scope': 'Complete section formatting on smoke schema; no publication or full-sample claim.'})
        print('SMOKE REPORT FORMAT VERIFIED', args.kind)
        return
    doc = resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md')
    text = doc.read_text()
    begin, end = f'<!-- BEGIN {tag} -->', f'<!-- END {tag} -->'
    if begin in text:
        i, j = text.index(begin), text.index(end) + len(end)
        text = text[:i] + section + text[j:]
    else:
        i = text.index('<!-- BEGIN PROBE STABILITY CAPTURE -->')
        text = text[:i] + section + '\n\n' + text[i:]
    doc.write_text(text)
    assert section in doc.read_text()
    names = ['analysis.json', 'raw_audit.json', 'resolution.json', 'pipeline_complete.json']
    if args.kind == 'stability':
        names += ['reproducibility.json', 'replicate_metrics.csv']
    write_json(out / 'report_evidence.json', {'inputs': {n: sha256(out / n) for n in names}, 'source_sha256': sha256(Path(__file__)), 'section_sha256': hashlib.sha256(section.encode()).hexdigest(), 'unix': time.time(), 'scope': 'Completed effect diagnosis; acceptance/smoothing final comparisons remain separate.'})
    print('REPORTED', args.kind, '402 independently audited states')

if __name__ == '__main__':
    main()
