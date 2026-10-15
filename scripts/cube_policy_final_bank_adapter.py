"""Execution adapter for frozen independent Cube policy methods."""
import json
import numpy as np
from zj_wam.io import resolve,sha256
from scripts.cube_policy_final_core import phase_roots
from scripts.cube_policy_calibration_bank_core import root_commands
BASE=resolve('artifacts/followup_v1');read=lambda p:json.loads(p.read_text())
def source_dir(seed):
    assert seed==340000 or 340200<=seed<340264
    return BASE/('cube_policy_refinement' if seed==340000 else 'cube_policy_final_nominal')
def split(seed):return 'smoke' if seed==340000 else 'test'
def settings(smoke):
    protocol=read(BASE/'cube_policy_final/final_protocol.json')
    for p,h in protocol['inputs'].items():assert sha256(resolve(p))==h,p
    cfg={**read(resolve('configs/followup_cube_policy_calibration_bank.json')),**protocol['config']};cfg['seeds']=protocol['test_seeds'];cfg['smoke_seeds']=[cfg['smoke_seed']]
    if not smoke:
        d=BASE/'cube_policy_final_nominal';assert read(d/'pipeline_complete.json')['raw_audit_sha256']==sha256(d/'raw_audit.json')
    return cfg
def states(seed,cfg):
    with np.load(source_dir(seed)/f'{seed}_nominal.npz') as a:return phase_roots(a,cfg)
def rules():return {x['method']:x['rule'] for x in read(BASE/'cube_policy_final/final_protocol.json')['rules']}
def choose(bank,rule):
    cfg=read(resolve('configs/followup_cube_policy_selection.json'));cost=bank['costs'];norm=bank['norm'];gain=(cost[0]-cost)/max(cost[0],1e-12);ok=(norm>cfg['minimum_norm'])&(gain>cfg['minimum_predicted_gain'])
    ok=ok&(bank['scores']<=rule['tau']) if rule['enabled'] else np.zeros(len(norm),bool)
    ids=np.flatnonzero(ok);return int(ids[np.lexsort((ids,norm[ids],cost[ids]))[0]]) if len(ids) else 0
