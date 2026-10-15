"""Final policy global/local raw audit and prespecified episode-level inference."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,pymunk
from scipy.stats import beta
from shapely.geometry import Polygon
from shapely.ops import unary_union
from zj_wam.policy_refinement import make_scene
from zj_wam.io import resolve,write_json,sha256
from scripts.analyze_followup_response_selection import mean_ci

BASE=resolve('artifacts/followup_v1');ROOT=BASE/'policy_refinement'


def read(p):return json.loads(p.read_text())


def binomial_interval(successes,n):
    if n==0:return {'count':0,'n':0,'mean':None,'ci95':None}
    assert 0<=successes<=n
    return {'count':int(successes),'n':int(n),'mean':successes/n,
        'ci95':[float(beta.ppf(.025,successes,n-successes+1)) if successes else 0.,float(beta.ppf(.975,successes+1,n-successes)) if successes<n else 1.]}


def energy(q):return float(np.sum((np.diff(q,n=3,axis=0)/.01**3)**2)*.01)


class Coverage:
    def __init__(self,seed):
        env=make_scene(seed);self.vertices=[list(s.get_vertices()) for s in env.block.shapes];self.body=pymunk.Body(body_type=pymunk.Body.KINEMATIC);self.body.center_of_gravity=env.block.center_of_gravity
        self.target=self.polygon(env._get_goal_pose_body(env.goal_pose));env.close()
    def polygon(self,body):return unary_union([Polygon([body.local_to_world(v) for v in vertices]) for vertices in self.vertices])
    def __call__(self,state):
        self.body.angle=float(state[4]);self.body.position=state[2:4].tolist();return self.polygon(self.body).intersection(self.target).area/self.target.area


def report(result,dest):
    primary=result['paired'][0];difference=primary['joint_difference'];lookup={r['method']:r for r in result['summary']};rows=[]
    for method in result['methods']:
        r=lookup[method];risk=r['global_risk_exact']['mean'];risk_text='미정의' if risk is None else f'{100*risk:.1f}%'
        rows.append(f"| {method} | {100*r['joint_exact']['mean']:.1f}% | {risk_text} | {100*r['any_success_exact']['mean']:.1f}% | {100*r['jerk_gain']['mean']:+.1f}% |")
    verdict='선언한 실질적 개선 조건을 충족했다.' if result['primary_decision']['supported_practical_advantage'] else '선언한 조건에서 LeWM의 실질적 추가 가치를 입증하지 못했다. 동등성이나 모든 환경에서의 무용성을 뜻하지 않는다.'
    lo,hi=difference['ci95']
    block=f'''<!-- BEGIN POLICY FINAL TEST -->
## Frozen policy의 독립 최종 평가

사전 예약한 12개 초기 상태 seed에서 6방법·72 episode를 완료했다. 원래 expert H=5에서 정한 선택 규칙을 그대로 옮겼으며 정책 test에서 threshold·모델·후보 설정을 다시 맞추지 않았다. 각 episode는 성공 여부와 무관하게 300 action을 실행했다. 수정 없는 정책과 수정 있는 정책은 같은 초기 상태와 정책 호출 난수 규칙을 공유하고, 각각 자기 관측으로 이후 행동을 생성한다.

주 비교의 LeWM−controller **전체 episode 동시 달성률 U 차이는 {100*difference['mean']:+.2f}%p**, paired family 95% 구간 [{100*lo:+.2f},{100*hi:+.2f}]%p였다. {verdict} 실질적 개선 폭은 U 10%p, 경험적 global 위반율 목표는 10% 이하, 최소 수정 적용 비율은 25%로 사전 지정했다.

| 방법 | 전체 episode U | 수정 episode 중 global 위반율 | 한 번 이상 task 성공 | 평균 전체 jerk 감소율 |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

U는 원래 정책 대비 시간 정렬된 물체 궤적 보존과 실제 jerk 10% 감소를 함께 요구한다. 물체 위치 2px·회전 2도 및 제어 시점 coverage 감소 0.02 기준을 유지했다. Task 성공률과 궤적 보존은 다른 지표다. 국소 chunk 평가 2,520개도 같은 raw trace에서 별도로 검증했으며, 국소 성공을 전체 episode 보존으로 바꾸어 보고하지 않는다.

12개 family의 정밀도에는 한계가 있다. Episode 지표에는 정확 이항 95% 구간을 함께 제공한다. 예를 들어 0/12 사건의 상한은 약 26.5%이므로, 관측 실패가 없더라도 모집단 실패율 10% 이하를 인증할 수 없다. Paired bootstrap의 폭이 0이어도 모집단 동등성의 근거로 해석하지 않는다. 보조 방법 비교는 탐색적이다.

검증은 100Hz 물체 위치·회전과 실제 pusher jerk, 원시 pose에서 독립적으로 재계산한 goal coverage, 수정 이전 action·물리 prefix, 2,520개 국소 counterfactual을 포함했다. 정책 최초 correction 이후의 첫 위반 시점도 별도 제공한다. 이 실험은 지연을 실제 제어에 삽입한 평가가 아니며, 새로운 policy calibration으로 배포 가능한 반복 보존 규칙을 찾은 결과도 아니다.

근거: [정책 별도 사전 잠금](../artifacts/followup_v1/policy_refinement/final_protocol.json), [최종 분석](../artifacts/followup_v1/policy_refinement/final_analysis/analysis.json), [독립 원시 감사](../artifacts/followup_v1/policy_refinement/final_analysis/raw_audit.json), [국소·누적 평가](../artifacts/followup_v1/policy_refinement/test/analysis.json).
<!-- END POLICY FINAL TEST -->
'''
    p=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=p.read_text();start='<!-- BEGIN POLICY FINAL TEST -->';end='<!-- END POLICY FINAL TEST -->'
    if start in s:s=s[:s.index(start)]+block+s[s.index(end)+len(end):]
    else:
        index=s.index('\n\n',s.index('\n\n')+2);s=s[:index]+'\n\n'+block+s[index:]
    p.write_text(s);write_json(dest/'report_evidence.json',{'analysis_sha256':sha256(dest/'analysis.json'),'raw_audit_sha256':sha256(dest/'raw_audit.json'),'report_sha256':sha256(p),'scope':'Policy final only; no claim allF0-F11complete'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate-split',choices=['development','calibration']);a=parser.parse_args();split=a.validate_split or 'test';final=a.validate_split is None
    if final:
        from scripts.run_followup_policy_final import verify,LOCK
        lock=verify()
    cfg=read(resolve('configs/followup_policy_refinement.json'));maincfg=read(resolve('configs/followup_v1.json'));fcfg=read(resolve('configs/followup_policy_final.json'));tol=maincfg['tolerances'];folder=ROOT/split;rawaudit=read(folder/'raw_audit.json');parent=pd.read_csv(folder/'evaluation.csv')
    assert rawaudit['state_method_rows']==72 and rawaudit['local_blocks']==2520 and len(rawaudit['paired_seeds'])==12
    rows=[];errors={};locals_checked=0
    def check(name,x,y,atol=1e-8):
        e=float(np.max(np.abs(np.asarray(x)-np.asarray(y))));errors[name]=max(errors.get(name,0.),e);assert e<=atol,(name,e)
    for seed in cfg[split+'_seeds']:
        with np.load(folder/f'{seed}_nominal.npz') as a:ref=a['physics'];refcov=a['coverage'];refaction=a['actions'];initial=a['initial_state']
        if final:check('locked_initial_state',initial,lock['test_initial_states'][str(seed)],0.)
        covfun=Coverage(seed);j0=energy(np.vstack([initial[:2],ref[:,:2]]))
        for method in cfg['methods']:
            path=folder/f'{seed}_{method}.json';meta=read(path);local=pd.read_csv(path.with_suffix('.csv'));active=[x for x in meta['decisions'] if x['active']]
            with np.load(path.with_suffix('.npz')) as a:
                physics=a['physics'];actions=a['actions'];cov=a['coverage'];assert len(actions)==300 and len(physics)==3000
                check('initial',a['initial_state'],initial,0.);check('dense_control',a['states'][1:],physics[9::10],0.)
                recomputed=np.array([covfun(x) for x in physics[9::10]]);check('independent_coverage',recomputed,cov,1e-10)
                first=active[0]['control_step'] if active else 300
                check('precorrection_physics',physics[:first*10],ref[:first*10],0.);check('precorrection_actions',actions[:first],refaction[:first],0.)
                for i,r in enumerate(local.itertuples()):
                    t=r.control_step;truth=a['local_nominal_physics'][i];actual=physics[t*10:(t+8)*10];d=actual[:,2:5]-truth[:,2:5]
                    pos=np.linalg.norm(d[:,:2],axis=1).max();rot=np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2]))).max();ct=np.array([covfun(x) for x in truth[[49,79]]]);check('independent_local_coverage',ct,a['local_nominal_coverage'][i,[4,7]],1e-10)
                    E=max(pos/tol['position'],rot/tol['rotation'],np.maximum(0,ct-cov[[t+4,t+7]]).max()/tol['coverage_loss']);check('local_E',E,r.object_error)
                    prefix=physics[t*10-30:t*10,:2];baseline=energy(np.r_[prefix,truth[:,:2]]);cost=energy(np.r_[prefix,actual[:,:2]]);gain=(baseline-cost)/baseline if baseline>1e-8 else 0.;check('local_gain',gain,r.jerk_gain);locals_checked+=1
                d=physics[:,2:5]-ref[:,2:5];pos=np.linalg.norm(d[:,:2],axis=1);angle=np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2])));loss=np.maximum(0,refcov-cov)
                dense=np.maximum(pos/tol['position'],angle/tol['rotation']);cover=loss/tol['coverage_loss'];E=float(max(dense.max(),cover.max()));cost=energy(np.vstack([initial[:2],physics[:,:2]]));gain=(j0-cost)/j0 if j0>1e-8 else 0.
                old=parent[(parent.seed==seed)&(parent.method==method)].iloc[0];check('parent_global_E',E,old.object_error);check('parent_global_gain',gain,old.jerk_gain)
                hits=np.flatnonzero(dense>1);chits=np.flatnonzero(cover>1);times=[]
                if len(hits):times.append((hits[0]+1)*.01)
                if len(chits):times.append((chits[0]+1)*.1)
                crossing=min(times) if times else None
                if crossing is not None:assert active and crossing>first*.1
                row={'seed':seed,'initial_pose_family':seed,'method':method,'active':bool(active),'preserved':E<=1,'violation':E>1,'object_error':E,'jerk_gain':gain,'joint':bool(active and E<=1 and gain>=.1),'any_success':bool((cov>.95).any()),'final_success':bool(cov[-1]>.95),'first_correction_seconds':first*.1 if active else None,'first_violation_seconds':crossing,'seconds_after_first_correction':crossing-first*.1 if crossing is not None else None}
                assert row['joint']==old.joint and row['any_success']==old.any_success and row['active']==old.active;rows.append(row)
    f=pd.DataFrame(rows);assert len(f)==72 and locals_checked==2520;summary=[];paired=[]
    for method,g in f.groupby('method'):
        active=g[g.active];summary.append({'method':method,'episodes':len(g),'joint_exact':binomial_interval(int(g.joint.sum()),len(g)),'global_risk_exact':binomial_interval(int(active.violation.sum()),len(active)),'any_success_exact':binomial_interval(int(g.any_success.sum()),len(g)),'preserved_exact':binomial_interval(int(g.preserved.sum()),len(g)),**{k:mean_ci(g,k) for k in ['joint','active','jerk_gain','any_success','final_success']}})
    for later,earlier in [('lewm_global','controller_pusher'),('vision_contact','controller_pusher'),('vision_adaptive','vision_contact'),('gaussian_action','controller_pusher')]:
        x=f[f.method==later].merge(f[f.method==earlier],on=['seed','initial_pose_family'],suffixes=('','_base'),validate='one_to_one');assert len(x)==12
        for k in ['joint','jerk_gain','any_success']:x[k+'_difference']=x[k].astype(float)-x[k+'_base'].astype(float)
        paired.append({'method':later,'baseline':earlier,**{k:mean_ci(x,k) for k in ['joint_difference','jerk_gain_difference','any_success_difference']}})
    wm=next(r for r in summary if r['method']=='lewm_global');risk=wm['global_risk_exact']['mean'];qualified=risk is not None and risk<=.1 and wm['active']['mean']>=.25;lo=paired[0]['joint_difference']['ci95'][0]
    result={'split':split,'validation_only':not final,'methods':cfg['methods'],'summary':summary,'paired':paired,'primary_decision':{'empirical_qualified':qualified,'supported_practical_advantage':bool(qualified and lo>=fcfg['minimum_practical_U_difference'])},'scope':fcfg['scope'],'precision':fcfg['sample_precision']}
    dest=ROOT/('final_analysis' if final else 'final_validation_'+split);dest.mkdir(exist_ok=True);f.to_csv(dest/'evaluation.csv',index=False)
    evidence={'split':split,'validation_only':not final,'episodes':72,'families':12,'local_blocks':locals_checked,'max_error':errors,'source_sha256':sha256(Path(__file__)),'parent_raw_audit_sha256':sha256(folder/'raw_audit.json')}
    if final:verify();result['lock_sha256']=sha256(LOCK);evidence['lock_sha256']=sha256(LOCK)
    write_json(dest/'raw_audit.json',evidence);write_json(dest/'analysis.json',result)
    if final:report(result,dest)
    print('AUDITED',split,'72episodes2520localblocks',flush=True)
if __name__=='__main__':main()
