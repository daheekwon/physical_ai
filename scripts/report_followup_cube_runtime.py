"""Independently check calibration budget choices and report complete Cube costs."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
BASE=resolve('artifacts/followup_v1/cube');OUT=BASE/'runtime';PURE=BASE/'runtime_pure';read=lambda p:json.loads(p.read_text())


def report():
    raw=read(OUT/'raw_audit.json');pr=read(PURE/'raw_audit.json');assert not raw['smoke'] and raw['states']==192 and raw['outcomes']==2304 and raw['timings']==6912 and not pr['smoke']
    for name,h in pr['inputs'].items():assert sha256(OUT/name)==h,name
    manifest=read(OUT/'manifest.json');pm=read(PURE/'manifest.json');cfg=manifest['cube_config'];rcfg=manifest['config'];a=read(PURE/'analysis.json');settings=read(PURE/'calibrated_settings.json')['settings'];f=pd.read_csv(OUT/'evaluation.csv');t=pd.read_csv(OUT/'timings.csv');b=pd.read_csv(PURE/'evaluation.csv');maxerr={};verified=0
    def check(name,x,y,tol=1e-12):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);assert x.shape==y.shape;err=float(np.max(abs(x-y)));maxerr[name]=max(maxerr.get(name,0.),err);assert err<=tol,(name,err)
    for setting in settings:
        spec=pm['config']['pipelines'][setting['pipeline']];options=[]
        for profile in sorted(spec['profiles']):
            g=f[(f.split=='calibration')&(f.method==spec['method'])&(f.profile==profile)];times=t[(t.split=='calibration')&(t.method==spec['method'])&(t.profile==profile)];active=g[g.active];contact=active[active.phase!='approach'];p95=float(np.quantile(times.seconds,.95))
            valid=len(active)>0 and len(contact)>0 and float((active.object_error>1).mean())<=cfg['target_risk'] and float((contact.object_error>1).mean())<=cfg['target_risk'] and float(g.active.mean())>=cfg['minimum_active_fraction'] and p95<=setting['cap_seconds']
            if valid:options.append((profile,(float(g.joint.mean()),float(g.jerk_gain.mean()),-p95)))
        chosen=max(options,key=lambda v:v[1])[0] if options else None;assert setting['profile']==chosen and setting['enabled']==(chosen is not None);verified+=1
        delivered=b[(b.pipeline==setting['pipeline'])&(b.cap_seconds==setting['cap_seconds'])].sort_values('state_id');assert len(delivered)==96
        if chosen is None:
            assert not delivered[['active','joint','violation']].to_numpy().any();assert not delivered[['candidate','object_error','jerk_gain','norm','deadline_exceedance']].to_numpy().any()
        else:
            source=f[(f.split=='development')&(f.method==spec['method'])&(f.profile==chosen)].sort_values('state_id');np.testing.assert_array_equal(delivered.state_id,source.state_id)
            for key in ['candidate','active','joint','violation','object_error','jerk_gain','norm']:check('delivered_'+key,delivered[key],source[key])
            times=t[(t.split=='development')&(t.method==spec['method'])&(t.profile==chosen)].copy();times['miss']=times.seconds>setting['cap_seconds'];expected=times.groupby('state_id').miss.mean();check('deadline_fraction',delivered.deadline_exceedance,delivered.state_id.map(expected))
        if setting['pipeline'].endswith('_no_wm'):
            measured=t[(t.method==spec['method'])&t.profile.isin(spec['profiles'])];assert not measured[['encoder_calls','jacobian_calls','finite_prediction_calls']].to_numpy().any()
    assert verified==35
    budgetrows=[];pipelines=list(pm['config']['pipelines']);labels={'controller_no_wm':'제어기','contact_no_wm':'접촉 proxy','lewm_verifier_only':'LeWM verifier만','lewm_generator_and_verifier':'LeWM 생성＋검증','lewm_best_calibrated':'LeWM calibration 선택'}
    for cap in rcfg['budgets_seconds']:
        cells=[]
        for pipeline in pipelines:
            r=next(x for x in a['summary'] if x['pipeline']==pipeline and x['cap_seconds']==cap)
            cells.append(f"{100*r['joint']['mean']:.1f} / {100*r['risk']['mean']:.1f}" if r['enabled'] and r['risk']['mean'] is not None else '미적용')
        budgetrows.append(f"| {cap*1000:g} | "+' | '.join(cells)+' |')
    profiles=[]
    for r in read(OUT/'analysis.json')['summary']:
        if r['split']!='development':continue
        profiles.append(f"| {r['profile']} | {r['method']} | {1000*r['p50_seconds']:.1f} | {1000*r['p95_seconds']:.1f} | {100*r['joint']['mean']:.1f} | "+(f"{100*r['risk']['mean']:.1f}" if r['risk']['mean'] is not None else '미정의')+' |')
    # Independently identify held-out feasibility rather than equating calibration success with validity.
    limitations=[]
    for (pipeline,cap),g in b.groupby(['pipeline','cap_seconds']):
        active=g[g.active];contact=active[active.phase!='approach'];risk=float(active.violation.mean()) if len(active) else None;cr=float(contact.violation.mean()) if len(contact) else None
        if len(active) and (risk>cfg['target_risk'] or (cr is not None and cr>cfg['target_risk'])):limitations.append({'pipeline':pipeline,'cap_seconds':cap,'development_risk':risk,'development_contact_risk':cr})
    supplemental={'calibration_choices_reproduced':verified,'max_error':maxerr,'development_risk_exceedances':limitations,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in [OUT/'raw_audit.json',OUT/'analysis.json',OUT/'evaluation.csv',OUT/'timings.csv',PURE/'manifest.json',PURE/'calibrated_settings.json',PURE/'analysis.json',PURE/'evaluation.csv',PURE/'raw_audit.json']},'scope':'Recomputed all35 pure-pipeline profile choices from raw calibration rows and timing repetitions; delivered development actions/metrics and model-free call counts checked. No new timings/physics.'};write_json(PURE/'independent_budget_audit.json',supplemental)
    paired=[]
    for r in a['paired']:
        if r['pipeline'] not in ['lewm_verifier_only','lewm_generator_and_verifier'] or r['cap_seconds'] not in [1.,2.,5.]:continue
        v=r['U_difference'];paired.append(f"| {r['cap_seconds']*1000:g} | {labels[r['pipeline']]} − 제어기 | {100*v['mean']:.2f} [{100*v['ci95'][0]:.2f}, {100*v['ci95'][1]:.2f}] |")
    lookup={(r['pipeline'],r['cap_seconds']):r for r in a['summary']}
    two_ctrl=lookup[('controller_no_wm',2.)];two_wm=lookup[('lewm_verifier_only',2.)];two_contact=lookup[('contact_no_wm',2.)]
    assert all(not r['enabled'] for r in a['summary'] if r['cap_seconds']<=.25)
    block=f'''<!-- BEGIN CUBE RUNTIME COMPLETE -->
## Cube 계산 비용 비교 완료: 생성기와 verifier 비용을 구분

**이번 구현에서는 250ms 이하 budget에서 조건을 만족한 구성이 없었다.** 500ms·1초에서는 LeWM verifier를 추가해도 제어기와 U·위반율이 같았다. 2초에서는 제어기 U {100*two_ctrl['joint']['mean']:.2f}%·위반 {100*two_ctrl['risk']['mean']:.2f}%, LeWM verifier U {100*two_wm['joint']['mean']:.2f}%·위반 {100*two_wm['risk']['mean']:.2f}%, 접촉 proxy U {100*two_contact['joint']['mean']:.2f}%·위반 {100*two_contact['risk']['mean']:.2f}%였다. LeWM의 U 증가를 확인한 결과는 아니지만, 이 비교에서 제어기의 위반율은 목표 10%를 넘고 LeWM의 관측 위반율은 더 낮았다. U만으로 제어기가 낫다고 해석하지 않는다. 위험 차이의 일반화나 통계적 우위는 별도 검증이 필요하다.

Calibration 24개 family·96개 상태와 development 24개 family·96개 상태에서 **12개 구성 × 3회**, 총 6,912회 시간을 새로 측정했다. 대응하는 선택 결과 2,304개를 원래 물리 실행과 연결해 감사했다. 제어기 응답 fit, 다항식 후보 생성·최적화, 필요한 encoder·Jacobian·유한 예측·선택을 매번 다시 계산했다. Checkpoint 초기 로딩·저장 관측의 disk I/O·영상 획득은 제외했다. 물리 실행은 같은 action임을 확인한 기존 dense trace를 재사용했다.

완전히 모델 없는 제어기/접촉 경로, 같은 비-WM 후보에 LeWM verifier만 추가한 경로, LeWM으로 방향을 만들고 검증하는 경로를 구분했다. **제어기와 접촉 경로의 모든 시간 측정에서 encoder·Jacobian·predictor 호출이 0임을 확인했다.** 비-WM full bank와 WM full bank의 기존 선택 재현은 {raw['full_parent_choice_reproductions']:,}개다.

아래는 development에서의 **U / 선택 후 위반율, %**다. U는 2mm·full 회전 2도의 500Hz 물체 궤적 보존과 실제 EEF jerk 10% 감소를 함께 달성한 상태 비율이다. Budget별 profile은 calibration의 p95, 전체/접촉 위반율 ≤10%, 적용 비율 ≥25%로 고정했다. ‘미적용’은 그 조건을 만족한 profile이 없어 nominal로 fallback한 경우다. 보존만으로 개선 성공이라고 세지 않는다.

| 계산 budget, ms | 제어기, WM 없음 | 접촉 proxy, WM 없음 | 같은 비-WM 후보＋LeWM verifier | LeWM 생성＋검증 | Calibration 선택 LeWM |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(budgetrows)}

같은 budget의 제어기 대비 LeWM 경로의 U 차이 일부를 아래에 제시한다. 전체 7개 budget·5개 경로, 적용 비율·jerk 감소·deadline 초과·family 신뢰구간은 원본 분석에 있다. 구간은 24개 development family의 bootstrap이며, 방법 동등성이나 낮은 위험의 확률적 보증을 뜻하지 않는다.

| Budget, ms | 비교 | U 차이, %p [95% 구간] |
| --- | --- | --- |
{chr(10).join(paired)}

개별 profile의 원시 비용과 성능은 다음과 같다. 이는 각 profile을 무조건 적용한 진단이며, 위의 calibration한 budget 선택과 구분한다.

| Profile | Verifier | p50, ms | p95, ms | U, % | 위반율, % |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(profiles)}

Calibration에서 조건을 통과했지만 development에서 전체 또는 접촉 위반율 10%를 넘은 budget–경로 조합은 {len(limitations)}개다. [독립 budget 감사](../artifacts/followup_v1/cube/runtime_pure/independent_budget_audit.json)에 해당 조합을 남겼다. **Calibration 통과를 실제 배포 조건의 보장으로 해석하지 않는다.** p95가 budget 이내라도 일부 호출은 더 느릴 수 있으며 deadline 초과율을 별도로 보고했다.

시간은 공유 host의 해당 구현·장치에서 측정했다. 측정 기간 중 CPU 기반 DINO 수집·분석 등이 함께 실행됐으므로 독점 host의 성능 하한으로 해석하지 않는다. 구성 순서는 상태마다 고정 난수로 섞고 각 구성을 세 번 측정했지만 모든 외부 부하를 제거한 실험은 아니다. 이 결과는 새 물리 실행이나 지연을 주입한 closed loop가 아니다. Cube의 독립 최종 비용 평가와 지연·반복·조건 적응은 별도로 남아 있다.

근거: [전체 비용 분석](../artifacts/followup_v1/cube/runtime/analysis.json), [원시 선택·물리 정합성 감사](../artifacts/followup_v1/cube/runtime/raw_audit.json), [순수 경로 비교](../artifacts/followup_v1/cube/runtime_pure/analysis.json), [calibration 선택 재현](../artifacts/followup_v1/cube/runtime_pure/independent_budget_audit.json).
<!-- END CUBE RUNTIME COMPLETE -->
'''
    doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();start='<!-- BEGIN CUBE RUNTIME COMPLETE -->';end='<!-- END CUBE RUNTIME COMPLETE -->'
    if start in s:i=s.index(start);j=s.index(end,i)+len(end);s=s[:i]+block.rstrip()+s[j:]
    else:anchor='<!-- END CUBE RUNTIME IMPLEMENTATION -->';i=s.index(anchor)+len(anchor);s=s[:i]+'\n\n'+block+s[i:]
    doc.write_text(s);write_json(PURE/'report_evidence.json',{'source_sha256':sha256(Path(__file__)),'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in [OUT/'raw_audit.json',OUT/'analysis.json',PURE/'analysis.json',PURE/'independent_budget_audit.json']},'document':str(doc.relative_to(resolve('.'))),'section_sha256':hashlib.sha256(block.rstrip().encode()).hexdigest()});print('REPORTED full Cube runtime',verified,'budget settings',len(limitations),'development exceedances',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-pid',type=int);args=p.parse_args();start=time.monotonic()
    while not (PURE/'raw_audit.json').exists():
        assert args.dependency_pid,'Full runtime audit incomplete';os.kill(args.dependency_pid,0)
        if time.monotonic()-start>7200:raise RuntimeError('Observation window elapsed; inspect live queue')
        time.sleep(5)
    report()
if __name__=='__main__':main()
