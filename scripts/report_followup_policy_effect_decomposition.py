"""Publish complete four-arm first-correction diagnosis after independent audit."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from zj_wam.io import resolve,write_json,sha256

OUT=resolve('artifacts/followup_v1/policy_effect_decomposition');read=lambda p:json.loads(p.read_text())

def report():
    a=read(OUT/'analysis.json');audit=read(OUT/'raw_audit.json');manifest=read(OUT/'manifest.json');cfg=manifest['config']
    assert not a['partial'] and not audit['partial'] and a['new_replan_branches']==20
    assert audit['arm_rows']==audit['raw_trace_checks_including_reused']==80 and audit['first_replan_prediction_checks']==20
    assert audit['manifest_sha256']==sha256(OUT/'manifest.json')
    for path,h in manifest['input_sha256'].items():assert sha256(resolve(path))==h,path
    f=pd.read_csv(OUT/'evaluation.csv');assert len(f)==80 and not f.duplicated(['seed','method','arm']).any()
    arms=['a_nominal','b_first_fixed','c_first_replan','d_repeated']
    assert set(f.arm)==set(arms) and set(f.seed)==set(cfg['development_seeds']) and set(f.method)==set(cfg['methods'])
    names={'controller_pusher':'제어기','vision_contact':'고정 접촉','vision_adaptive':'온라인 접촉','lewm_global':'LeWM','gaussian_action':'Gaussian'}
    rows=[]
    for method in cfg['methods']:
        values=[]
        for arm in arms[1:]:
            g=f[(f.method==method)&(f.arm==arm)];assert len(g)==4;values.append(f'{int(g.preserved.sum())}/4')
        b=f[(f.method==method)&(f.arm=='b_first_fixed')];c=f[(f.method==method)&(f.arm=='c_first_replan')];delta=100*(c.jerk_gain.mean()-b.jerk_gain.mean())
        rows.append(f"| {names[method]} | {' | '.join(values)} | {delta:+.1f}%p |")
    totals={arm:int(f[f.arm==arm].preserved.sum()) for arm in arms[1:]}
    b=f[f.arm=='b_first_fixed'];c=f[f.arm=='c_first_replan'];joined=c.merge(b,on=['seed','method'],suffixes=('_c','_b'),validate='one_to_one');lost=int((joined.preserved_b & ~joined.preserved_c).sum());recovered=int((~joined.preserved_b & joined.preserved_c).sum())
    block=f'''<!-- BEGIN POLICY FIRST CORRECTION -->
## 첫 correction의 잔류 효과와 이후 정책 반응

개발 seed 4개·다섯 수정 방법에서 네 조건을 모두 완료했다. (A) 수정 없는 원래 실행, (B) 첫 correction만 적용하고 원래의 후속 상대 명령을 고정 재생, (C) 같은 첫 correction 뒤 자기 관측으로 정책을 재계획하되 추가 수정 없음, (D) 계속 재계획·수정하는 실행을 비교했다. 새 물리 실행은 A 4개·B 20개·C 20개이며 D는 기존 검증 결과를 재사용했다. A가 방법마다 반복되는 분석표의 80행을 독립 episode 80개로 세지 않는다.

모든 비교에서 correction 이전 action·물리 trace와 첫 correction chunk까지의 실행이 정확히 같았다. C 20개에서는 이후 명령이 자기 관측의 정책 출력으로부터 수정 없이 변환됐는지 검사했고, 첫 재계획의 정책 출력도 기존 D 실행과 일치했다. 전체 300개 행동(30초)의 100Hz 물체 pose·실행 jerk와 독립 polygon coverage를 감사했다.

**전체 궤적 보존은 B {totals['b_first_fixed']}/20, C {totals['c_first_replan']}/20, D {totals['d_repeated']}/20이었다.** 아래 표의 각 셀은 네 seed 중 보존한 수다. Jerk 차이는 C와 B의 전체 jerk 감소율 평균 차이이며, 보존 실패를 상쇄하는 성공 지표로 사용하지 않는다.

| 첫 correction 방법 | B 원래 후속 명령 고정 | C 이후 정책 재계획 | D 매번 재계획·수정 | C−B jerk 감소율 차이 |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

B에서는 정책의 후속 반응과 추가 correction이 없어도 {20-totals['b_first_fixed']}/20에서 전체 궤적이 이탈했다. 같은 첫 correction 이후 정책 재계획을 허용한 C는 B에서 보존하던 {lost}개를 보존하지 못했고, B에서 실패하던 {recovered}개는 보존했다. 이는 네 개발 family에서 관측한 분기 결과이며 모집단의 방법 순위나 독립적인 기여도 추정은 아니다.

B는 원래의 **상대 행동 명령**을 고정한 진단이다. 첫 수정 때문에 pusher의 상태가 달라지면 같은 후속 상대 명령도 다른 실제 pusher 경로를 만들 수 있다. 따라서 B의 장기 이탈을 물체의 수동 관성만으로 설명하지 않는다. B−C는 그 후 정책의 반응, C−D는 추가 correction을 허용한 실행 방식의 영향을 진단하지만, 비선형 상호작용 때문에 이를 독립적 효과처럼 더하지 않는다. Task 성공 여부는 물체 궤적 보존과 별도로 원본 표에 제공한다.

이번 결과는 짧은 수정 구간의 보존이 이후 원래 궤적으로의 복귀를 보장하지 않음을 보여준다. 긴 suffix, 종료 상태·속도, 누적 기준 궤적에 대한 허용량과 정책 반응을 함께 다뤄야 한다. 이 분기 진단에는 실시간 추론 지연을 주입하지 않았으며 지연 실험과 구분한다.

근거: [사전 분기 프로토콜](../artifacts/followup_v1/policy_effect_decomposition/manifest.json), [전체 분석](../artifacts/followup_v1/policy_effect_decomposition/analysis.json), [80행 원시 감사](../artifacts/followup_v1/policy_effect_decomposition/raw_audit.json), [개별 분기 결과](../artifacts/followup_v1/policy_effect_decomposition/evaluation.csv).
<!-- END POLICY FIRST CORRECTION -->
'''
    p=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');doc=p.read_text();start='<!-- BEGIN POLICY FIRST CORRECTION -->';end='<!-- END POLICY FIRST CORRECTION -->';assert start in doc and end in doc;doc=doc[:doc.index(start)]+block+doc[doc.index(end)+len(end):];p.write_text(doc)
    write_json(OUT/'report_evidence.json',{'inputs':{name:sha256(OUT/name) for name in ['manifest.json','analysis.json','raw_audit.json','evaluation.csv']},'source_sha256':sha256(Path(__file__)),'counts':totals,'scope':'Full four-development-family diagnosis only; no finaltest retuning, no population claim.'})
    print('REPORTED complete four-arm policy diagnosis',totals,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-pid',type=int);args=p.parse_args()
    if args.dependency_pid:
        start=time.monotonic();print('WAITING for complete20C branches and80arm raw audit',flush=True)
        while not (OUT/'raw_audit.json').exists() or not (OUT/'analysis.json').exists():
            try:os.kill(args.dependency_pid,0)
            except ProcessLookupError:raise RuntimeError('Policy decomposition parent ended before full audit; inspect logs')
            if time.monotonic()-start>21600:raise RuntimeError('Observation window elapsed; revalidate parent')
            time.sleep(10)
    report()
if __name__=='__main__':main()
