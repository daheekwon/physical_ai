"""Post-preflight paired sham diagnosis; preserves preregistered conservative results."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from zj_wam.io import resolve,sha256,write_json
from scripts.audit_followup_dinowm_policy_final import interval
BASE=resolve('artifacts/followup_v1');OUT=BASE/'cube_shift_preflight';read=lambda p:json.loads(p.read_text())
def main():
    audit=read(OUT/'raw_audit.json');done=read(OUT/'pipeline_complete.json');assert done['raw_audit_sha256']==sha256(OUT/'raw_audit.json') and audit['states']==48;cfg=read(OUT/'manifest.json')['config'];files=[OUT/'manifest.json',OUT/'raw_audit.json',OUT/'pipeline_complete.json'];rows=[];angle_error=0.;fields=['qpos','qvel','object_pose','object_velocity','eef_position','eef_quaternion','robot_object_contacts','time']
    for p in sorted(OUT.glob('*_t*.json')):
        meta=read(p);a=dict(np.load(p.with_suffix('.npz')));assert meta['data_sha256']==sha256(p.with_suffix('.npz'));files.extend([p,p.with_suffix('.npz')]);base=a['sham_forward_r0_object_pose'];q0=Rotation.from_quat(base[:,[4,5,6,3]]);qb=base[:,3:7]/np.linalg.norm(base[:,3:7],axis=1,keepdims=True)
        for condition in cfg['conditions'][2:]:
            name=condition['name'];pose=a[name+'_r0_object_pose'];pos=float(np.linalg.norm(pose[:,:3]-base[:,:3],axis=1).max());rot=float((Rotation.from_quat(pose[:,[4,5,6,3]])*q0.inv()).magnitude().max());q=pose[:,3:7]/np.linalg.norm(pose[:,3:7],axis=1,keepdims=True);alt=float((2*np.arccos(np.clip(abs((q*qb).sum(1)),0,1))).max());np.testing.assert_allclose(rot,alt,atol=1e-7,rtol=0);angle_error=max(angle_error,abs(rot-alt));effective=pos>cfg['minimum_position_effect_m'] or rot>cfg['minimum_rotation_effect_rad'];assert effective==(pos>cfg['minimum_position_effect_m'] or alt>cfg['minimum_rotation_effect_rad']);exact=all(np.array_equal(a[name+'_r0_'+k],a['sham_forward_r0_'+k]) for k in fields)
            if name=='mass_double_friction_half':assert all(np.array_equal(a[name+'_r0_'+k],a['mass_double_r0_'+k]) for k in fields)
            rows.append({'state_id':p.stem,'initial_pose_family':meta['seed'],'phase':meta['phase'],'condition':name,'position_effect_m':pos,'rotation_effect_rad':rot,'effective':bool(effective),'all_dynamic_fields_exact_sham':exact})
    f=pd.DataFrame(rows);summary=[]
    for (name,phase),g in f.assign(phase='all').groupby(['condition','phase']):summary.append({'condition':name,'phase':phase,'states':len(g),'effective_count':int(g.effective.sum()),'exact_count':int(g.all_dynamic_fields_exact_sham.sum()),'max_position_m':float(g.position_effect_m.max()),'max_rotation_rad':float(g.rotation_effect_rad.max()),**{k:interval(g,k) for k in ['effective','position_effect_m','rotation_effect_rad']}})
    for (name,phase),g in f.groupby(['condition','phase']):summary.append({'condition':name,'phase':phase,'states':len(g),'effective_count':int(g.effective.sum()),'exact_count':int(g.all_dynamic_fields_exact_sham.sum()),**{k:interval(g,k) for k in ['effective','position_effect_m','rotation_effect_rad']}})
    assert len(f)==240;f.to_csv(OUT/'sham_comparison.csv',index=False);write_json(OUT/'sham_comparison.json',{'states':48,'families':12,'rows':len(f),'summary':summary,'rotation_check_max_error':angle_error,'inputs':{str(p.relative_to(resolve('.'))):sha256(p) for p in files},'source_sha256':sha256(Path(__file__)),'scope':'Post-preflight explanatory comparison to same-forward sham, no new execution or tuning. Full time-aligned traces compared directly; does not subtract scalar errors or replace original conservative10xsham condition definition. Direct effect floor remains original1um/10urad.'})
    lines=['<!-- BEGIN CUBE SHIFT SHAM COMPARISON -->','### Cube 조건 변화의 추가 대조: 같은 forward 절차끼리 비교','',
    '앞의 사전 지정 검사는 원래 nominal과의 차이가 sham 오차보다 충분히 큰지 확인했다. Sham 자체의 최대 변화가 약0.348mm·1.315도여서, 물성 변경의 효과를 더 분명히 보기 위해 **변경 조건과 같은 forward를 적용한 sham의 시간별 궤적을 직접 비교**했다. 최대 오차끼리 빼는 계산이 아니다. 아래는 완료 후 추가한 설명용 분석이며 원래 효과 판정이나 실행 프로토콜을 바꾸지 않았다.','',
    '| 물성 변경 | Sham 대비 효과 검출 | 기록된 물리량 전체가 sham과 동일 | 최대 위치 차이, mm | 최대 회전 차이, deg |','| --- | --- | --- | --- | --- |']
    for name in [x['name'] for x in cfg['conditions'][2:]]:
        x=next(z for z in summary if z['condition']==name and z['phase']=='all');lines.append(f"| {name} | {x['effective_count']}/48 | {x['exact_count']}/48 | {1000*x['max_position_m']:.4f} | {np.rad2deg(x['max_rotation_rad']):.4f} |")
    lines+=['',
    '**객체 마찰0.5배는48개 상태 모두에서 qpos·qvel·물체 pose/속도·EEF pose·접촉 기록·시각이 sham과 정확히 같았다.** 질량2배＋마찰0.5배도 모든 상태에서 질량2배 단독과 정확히 같았다. 따라서 이 실행 범위에서는 마찰 감소를 독립적인 유효 변화나 결합 변화에 대한 적응 근거로 사용할 수 없다. 접촉 상대 재질과 action window가 다른 경우까지 일반화하지 않는다.','',
    '질량0.5배/2배는 각각25/48개 상태, 마찰2배는4/48개 상태에서 원래 수치 하한1μm 또는10μrad를 넘는 직접 효과가 있었다. 이 비율은2mm·2도 보존 위반율이 아니다. 향후 적응 비교에서는 같은 조건·같은 초기화 절차의 nominal을 기준으로 삼아, 물성 변경 효과와 초기화 절차의 효과를 섞지 않아야 한다.','',
    '모든 물리 trace는 앞선672개 분기 독립 감사의 결과를 사용했다. 회전 오차는 상대 회전과 quaternion 내적 두 계산으로 확인했고, 단계별 효과와 family 구간도 기록했다. 이 분석 자체는 adaptive refinement의 성공 근거가 아니다.','',
    '근거: [같은 절차의 sham 비교](../artifacts/followup_v1/cube_shift_preflight/sham_comparison.json), [모든 상태별 결과](../artifacts/followup_v1/cube_shift_preflight/sham_comparison.csv).','<!-- END CUBE SHIFT SHAM COMPARISON -->'];section='\n'.join(lines);doc=resolve('docs/ZJ_WAM_FOLLOWUP_RESULTS.md');s=doc.read_text();begin=lines[0];end=lines[-1]
    if begin in s:i=s.index(begin);j=s.index(end)+len(end);s=s[:i]+section+s[j:]
    else:anchor='<!-- END CUBE SHIFT VALIDITY -->';i=s.index(anchor)+len(anchor);s=s[:i]+'\n\n'+section+s[i:]
    doc.write_text(s);write_json(OUT/'sham_report_evidence.json',{'inputs':{p:sha256(OUT/p) for p in ['sham_comparison.json','sham_comparison.csv','raw_audit.json']},'section_sha256':hashlib.sha256(section.encode()).hexdigest(),'source_sha256':sha256(Path(__file__)),'unix':time.time()});print('VERIFIED SHAM COMPARISON',len(f),angle_error,flush=True)
if __name__=='__main__':main()
