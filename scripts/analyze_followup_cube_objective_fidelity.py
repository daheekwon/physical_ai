"""Audit expected versus realized local jerk on completed repeated Cube trials."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.analyze_followup_response_selection import mean_ci
BASE=resolve('artifacts/followup_v1');SRC=BASE/'cube_policy_refinement';OUT=BASE/'cube_objective_fidelity';read=lambda p:json.loads(p.read_text())

def energy(p):return float(np.sum(np.diff(p,n=3,axis=0)**2)/.002**5)

def main():
    OUT.mkdir(exist_ok=True);done=read(SRC/'pipeline_complete.json');assert done['raw_audit_sha256']==sha256(SRC/'raw_audit.json');cfg=read(resolve('configs/followup_cube_policy_refinement.json'));rows=[];inputs={str(p.relative_to(resolve('.'))):sha256(p) for p in [Path(__file__),SRC/'raw_audit.json',SRC/'pipeline_complete.json']};errors={'expected_relative':0.,'actual_gain':0.}
    for seed in cfg['development_seeds']:
        for method in [m for m in cfg['methods'] if m!='nominal']:
            path=SRC/f'{seed}_{method}.npz';meta=read(path.with_suffix('.json'));assert sha256(path)==meta['data_sha256'];inputs[str(path.relative_to(resolve('.')))]=sha256(path);inputs[str(path.with_suffix('.json').relative_to(resolve('.')))]=sha256(path.with_suffix('.json'))
            with np.load(path) as a:
                for row in meta['local_rows']:
                    bi=row['block'];selected=int(a[f'block_{bi:03d}_selected']);cost=a[f'bank_{bi:03d}_costs'];pred=a[f'bank_{bi:03d}_predicted_robot'];prefix=a[f'block_{bi:03d}_past_points'][-75:,:3];expected=[energy(np.r_[prefix,pred[i,:,:3]]) for i in [0,selected]];np.testing.assert_allclose(expected,cost[[0,selected]],rtol=1e-8,atol=1e-2);errors['expected_relative']=max(errors['expected_relative'],float(np.max(abs(np.array(expected)-cost[[0,selected]])/np.maximum(1,cost[[0,selected]]))));physical_prefix=a[f'local_{bi:03d}_prefix_eef_position'];actual=[energy(np.r_[physical_prefix,a[f'local_{bi:03d}_{tag}_eef_position']]) for tag in ['nominal','modified']];gain=(actual[0]-actual[1])/max(actual[0],1e-12);np.testing.assert_allclose(gain,row['jerk_gain'],atol=1e-8);errors['actual_gain']=max(errors['actual_gain'],abs(gain-row['jerk_gain']));pg=(expected[0]-expected[1])/max(expected[0],1e-12);active=bool(selected);eg=pg-gain
                    rows.append({'seed':seed,'initial_pose_family':seed,'method':method,'block':bi,'step':row['step'],'selected':selected,'active':active,'expected_nominal_jerk':expected[0],'expected_modified_jerk':expected[1],'actual_nominal_jerk':actual[0],'actual_modified_jerk':actual[1],'predicted_gain':pg,'actual_gain':gain,'gain_overprediction':eg,'predicted_useful':active and pg>=.1,'actual_useful':active and gain>=.1,'actual_worse':active and gain<0.,'object_preserved':row['object_error']<=1,'nominal_relative_cost_error':(expected[0]-actual[0])/max(actual[0],1e-12),'nominal_abs_relative_cost_error':abs(expected[0]-actual[0])/max(actual[0],1e-12),'joint':active and gain>=.1 and row['object_error']<=1})
            print('OBJECTIVE',seed,method,flush=True)
    f=pd.DataFrame(rows);assert len(f)==12*3*57;f.to_csv(OUT/'evaluation.csv',index=False);summary=[]
    for method,g in f.groupby('method'):
        active=g[g.active];wanted=active[active.predicted_useful];summary.append({'method':method,'blocks':len(g),'active_blocks':len(active),'predicted_useful_blocks':len(wanted),**{k:mean_ci(active,k) for k in ['predicted_gain','actual_gain','gain_overprediction','actual_worse','object_preserved','nominal_abs_relative_cost_error']},'actual_useful_given_predicted_useful':mean_ci(wanted,'actual_useful'),'joint_given_predicted_useful':mean_ci(wanted,'joint'),'median_nominal_abs_relative_cost_error':float(active.nominal_abs_relative_cost_error.median())})
    analysis={'blocks':len(f),'summary':summary,'scope':'Posthoc selected local H5+3 objective prediction fidelity on same audited Cube development episodes; branch outcomes not available to executed optimizer. Local expected-versus-actual cost diagnosis, not independent causal identification or new selector.'};write_json(OUT/'analysis.json',analysis);write_json(OUT/'raw_audit.json',{'inputs':inputs,'blocks':len(f),'max_errors':errors,'scope':'Recomputed selected and nominal expected translation jerk from saved predicted robot traces and physical-prefix inputs; actual jerk from both dense branches; all gains reproduce original independently audited episode records.','analysis_sha256':sha256(OUT/'analysis.json')})
    lines=['<!-- BEGIN CUBE OBJECTIVE FIDELITY -->','## Cube에서 예상 jerk 감소가 실제 실행으로 이어지는가','',f"반복 실험의 세 수정 방법에서 **{len(f):,}개 국소 H5＋3 비교**를 재검사했다. 각 선택은 실행 전에 확정됐으며 이 분석의 실제 미래 trace는 선택 입력이 아니다. 아래 평균은 활성 correction에 한정하고, 불확실성은 12개 초기 상태 family 단위로 계산했다.",'','| 방법 | 예상 jerk 감소, % | 실제 jerk 감소, % | 실제 jerk 증가, % | 예상 10% 이상 후보의 실제 10% 달성, % |','| --- | --- | --- | --- | --- |']
    for r in summary:
        conditional=r['actual_useful_given_predicted_useful']['mean'];ct='—' if conditional is None else f'{100*conditional:.2f}';lines.append(f"| {r['method']} | {100*r['predicted_gain']['mean']:.2f} | {100*r['actual_gain']['mean']:.2f} | {100*r['actual_worse']['mean']:.2f} | {ct} |")
    lines+=['','예상 감소와 실제 감소의 차이는 현재 past-PD surrogate와 후보 분포에서의 목적함수 충실도 문제를 보여준다. 이 차이를 LeWM의 물체 예측 오류로 바로 돌릴 수 없다. 물체 verifier를 통과해도 실행 jerk를 잘못 예측하면 유용한 수정을 선택하지 못할 수 있다.','','따라서 정책용 재보정에서는 물체 위험 threshold만 조절하는 비교와 함께, 현재까지의 proprioception으로 실행 응답·jerk 예측을 보완하는 비교가 필요하다. 원래 후보 공간의 물리적 개선 가능성은 별도 유한 oracle로 확인해야 한다. 이번 국소 진단에서 잘 나온 조건을 기존 최종 평가에 사후 적용하지 않는다.','','근거: [예상·실제 비용 및 family 구간](../artifacts/followup_v1/cube_objective_fidelity/analysis.json), [2,052개 국소 기록](../artifacts/followup_v1/cube_objective_fidelity/evaluation.csv), [원시 재계산](../artifacts/followup_v1/cube_objective_fidelity/raw_audit.json).','<!-- END CUBE OBJECTIVE FIDELITY -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:i=s.index('<!-- END CUBE JERK PARTITION -->')+len('<!-- END CUBE JERK PARTITION -->');s=s[:i]+'\n\n'+section+s[i:]
    doc.write_text(s);write_json(OUT/'completion_lock.json',{'inputs':inputs,'source_sha256':sha256(Path(__file__)),'analysis_sha256':sha256(OUT/'analysis.json'),'raw_audit_sha256':sha256(OUT/'raw_audit.json'),'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'unix':time.time()});print(json.dumps(analysis,indent=2),flush=True)

if __name__=='__main__':main()
