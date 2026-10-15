"""Transfer nominal-only verifier rules to shifted banks with independent checks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,sha256,write_json
from scripts.output_replacement_core import physical,selections
from scripts.audit_followup_output_replacement import selected
from scripts.analyze_followup_response_selection import mean_ci
from scripts.audit_retrained_dinowm_readouts import cluster
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def main(smoke):
    out=BASE/('cube_shift_candidate_bank_smoke' if smoke else 'cube_shift_candidate_bank');audit=read(out/'raw_audit.json');manifest=read(out/'manifest.json');cfg=read(resolve('configs/followup_cube_policy_selection.json'));assert audit['contexts']==(28 if smoke else 336) and audit['source_sha256']==sha256(resolve('scripts/audit_followup_cube_shift_candidate_bank.py'))
    for p,h in {**manifest['inputs'],**audit['inputs']}.items():assert sha256(resolve(p))==h,p
    if smoke:
        old=read(BASE/'cube/proposal_ablation/analysis.json')['settings']['non_wm_proposals']['rules'];rules={m:{'enabled':not r.get('disabled',False),'tau':r.get('tau')} for m,r in old.items()};rulefile=BASE/'cube/proposal_ablation/analysis.json'
    else:
        selection=BASE/'cube_policy_selection';sa=read(selection/'raw_audit.json');assert sa['calibration_trials']>0 and read(selection/'pipeline_complete.json')['selection_audit_sha256']==sha256(selection/'raw_audit.json');rulefile=selection/'rules.json';rules={x['method']:x['rule'] for x in read(rulefile)['rules']}
        for p,h in read(selection/'protocol.json')['inputs'].items():assert sha256(resolve(p))==h,p
    rows=[];frames=[];inputs={};errors={}
    def check(name,x,y,tol=1e-8):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);np.testing.assert_allclose(x,y,atol=tol,rtol=1e-10,equal_nan=True,err_msg=name);errors[name]=max(errors.get(name,0.),float(abs(x-y).max()) if x.size else 0.)
    for source in manifest['sources']:
        cp=resolve(source);cm=read(cp);p=out/cm['condition']/cp.name;a=dict(np.load(p.with_suffix('.npz')));truth=physical('cube',a);cost=a['costs'].astype(float);n=len(cost);frame=pd.DataFrame({'state_id':cp.stem,'initial_pose_family':cm['seed'],'phase':cm['phase'],'candidate':np.arange(n),'norm':a['norm'],'predicted_cost':cost,'predicted_gain':(cost[0]-cost)/max(cost[0],1e-12),'dense_full':truth['dense_full'],'jerk_gain':truth['jerk_gain'],**{m:a[m+'_scores'].astype(float) for m in cfg['methods']}})
        original=pd.read_csv(p.with_suffix('.csv'));check('physical_metric',frame.dense_full,original.object_error,2e-5);check('physical_jerk',frame.jerk_gain,original.jerk_gain);frame['condition']=cm['condition'];frames.append(frame)
        for q in [p,p.with_suffix('.npz'),p.with_suffix('.csv')]:inputs[str(q.relative_to(resolve('.')))]=sha256(q)
    f=pd.concat(frames,ignore_index=True)
    for condition,g in f.groupby('condition'):
        for method in cfg['methods']:
            chosen=selections(g,method,rules[method],cfg);independent=selected(g,method,rules[method],cfg)
            for col in chosen:
                if col in ['state_id','phase']:np.testing.assert_array_equal(chosen[col],independent[col])
                else:check('selection_'+col,chosen[col],independent[col])
            chosen['condition']=condition;chosen['method']=method;rows.append(chosen)
    choices=pd.concat(rows,ignore_index=True);summary=[];paired=[]
    def interval(g,col):
        answer=mean_ci(g,col)
        if len(g):
            avg,ci=cluster(g[col].to_numpy(),g.initial_pose_family.to_numpy());check('independent_mean',avg,answer['mean']);check('independent_ci',ci,answer['ci95'])
        return answer
    for (condition,method),g in choices.groupby(['condition','method']):
        summary.append({'condition':condition,'method':method,'states':len(g),**{col:interval(g,col) for col in ['active','joint','jerk_gain','predicted_gain','candidate_coverage']},'selected_risk':interval(g[g.active],'violation')})
        if condition!='sham_forward':
            sham=choices[(choices.condition=='sham_forward')&(choices.method==method)];v=g.merge(sham,on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
            for col in ['joint','active','jerk_gain']:v['difference']=v[col].astype(float)-v[col+'_base'].astype(float);paired.append({'type':'condition_vs_sham','condition':condition,'method':method,'metric':col,**interval(v,'difference')})
    for condition,g in choices.groupby('condition'):
        for method,baseline in [('lewm_global','controller_displacement'),('proprio_contact','controller_displacement'),('lewm_global','proprio_contact')]:
            v=g[g.method==method].merge(g[g.method==baseline],on=['state_id','initial_pose_family'],suffixes=('','_base'),validate='one_to_one')
            for col in ['joint','active','jerk_gain']:v['difference']=v[col].astype(float)-v[col+'_base'].astype(float);paired.append({'type':'verifier_contrast','condition':condition,'method':method,'baseline':baseline,'metric':col,**interval(v,'difference')})
    assert len(choices)==3*audit['contexts'];f.to_csv(out/'scored_candidates.csv',index=False);choices.to_csv(out/'selected.csv',index=False);write_json(out/'analysis.json',{'smoke':smoke,'contexts':audit['contexts'],'candidates':len(f),'selections':len(choices),'rules':rules,'rule_source':str(rulefile.relative_to(resolve('.'))),'rule_sha256':sha256(rulefile),'summary':summary,'paired':paired,'scope':'Single H5+3 correction after shifted warmup with causal current-observation nominal commands; all verifiers unchanged nominal-only thresholds (smoke uses expert rules). All arms use refreshed robot/gripper command compiler. No four-arm allowance/compute adaptation, condition retuning, repeatedglobal efficacy or newfinal claim. Originalnominal versus sham are distinct; materialeffects compare same-forward sham. Previously inspected developmentfamilies only.'});write_json(out/'selection_audit.json',{'smoke':smoke,'contexts':audit['contexts'],'candidates':len(f),'selections':len(choices),'max_errors':errors,'inputs':inputs,'analysis_sha256':sha256(out/'analysis.json'),'rule_sha256':sha256(rulefile),'bank_raw_audit_sha256':sha256(out/'raw_audit.json'),'source_sha256':sha256(Path(__file__)),'helpers':{p:sha256(resolve(p)) for p in ['scripts/output_replacement_core.py','scripts/audit_followup_output_replacement.py','scripts/analyze_followup_response_selection.py','scripts/audit_retrained_dinowm_readouts.py']},'scope':'Every deterministic choice crosschecked by independent DataFrame sorting and all family intervals by independent bootstrap helper. Raw candidate metric crosscheck; no shifted-data threshold search.'});print('ANALYZED SHIFT BANK',audit['contexts'],len(choices),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.smoke)
