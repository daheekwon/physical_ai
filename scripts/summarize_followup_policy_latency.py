"""Supplement the immutable latency audit with normalized applied-action counts."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve, sha256, write_json
from zj_wam.adapters import ExpertDataset
from scripts.analyze_followup_response_selection import mean_ci

BASE = resolve('artifacts/followup_v1')

def read(path):
    return json.loads(path.read_text())

def summarize(smoke=False):
    out = BASE / ('policy_latency_smoke' if smoke else 'policy_latency')
    manifest = read(out / 'manifest.json')
    audit = read(out / 'raw_audit.json')
    assert audit['smoke'] == smoke
    assert audit['manifest_sha256'] == sha256(out / 'manifest.json')
    for path, h in manifest['input_sha256'].items():
        assert sha256(resolve(path)) == h, path
    cfg = read(resolve('configs/followup_v1.json'))
    scale = ExpertDataset(cfg['dataset']).scaler.scale_
    frame = pd.read_csv(out / 'evaluation.csv')
    calls = []
    episode_active = {}
    for (seed, method, profile), group in frame.groupby(['seed', 'method', 'profile']):
        path = out / f'{seed}_{method}_{profile}.json'
        meta = read(path)
        assert meta['manifest_sha256'] == sha256(out / 'manifest.json')
        with np.load(path.with_suffix('.npz')) as a:
            active = False
            for k, d in enumerate(meta['calls']):
                count = d['delivered_count']
                request_norm = float(np.linalg.norm((a['planned_chunks'][k, :min(8, meta['steps']-d['request_step'])] - a['nominal_chunks'][k, :min(8, meta['steps']-d['request_step'])]) / scale))
                assert (request_norm > 1e-6) == d['active'], (seed, method, profile, k, request_norm)
                actual_norm = 0.
                if count:
                    t = d['dispatch_step']
                    actual_norm = float(np.linalg.norm((a['actions'][t:t+count] - a['nominal_chunks'][k, :count]) / scale))
                applied = actual_norm > 1e-6
                active |= applied
                calls.append({'seed':seed, 'method':method, 'profile':profile, 'call_index':k, 'requested_active':d['active'], 'delivered_active':applied, 'delivered_norm':actual_norm, 'delivered_actions':count})
        episode_active[(seed, method, profile)] = active
    old_active = frame.active.to_numpy().copy()
    frame['active'] = [episode_active[(r.seed,r.method,r.profile)] for r in frame.itertuples()]
    frame['joint'] = frame.active & frame.preserved & (frame.jerk_gain >= cfg['meaningful_jerk_reduction'])
    summary = []
    for (method, profile, reference), g in frame.groupby(['method','profile','reference']):
        selected = g[g.active].copy()
        selected['violation'] = ~selected.preserved
        summary.append({'method':method,'profile':profile,'reference':reference,'episodes':len(g), **{k:mean_ci(g,k) for k in ['joint','preserved','jerk_gain','any_success','final_success','hold_fraction','policy_calls','active']},'active_risk':mean_ci(selected,'violation')})
    paired = []
    for (profile, reference), g in frame.groupby(['profile','reference']):
        baseline = g[g.method=='controller_pusher']
        if not len(baseline):
            continue
        for method in manifest['config']['methods']:
            if method in ['nominal','controller_pusher']:
                continue
            match = g[g.method==method].merge(baseline,on=['seed','initial_pose_family'],suffixes=('','_base'))
            if not len(match):
                continue
            for key in ['joint','jerk_gain','any_success','hold_fraction']:
                match[key+'_difference'] = match[key].astype(float)-match[key+'_base'].astype(float)
            paired.append({'method':method,'baseline':'controller_pusher','profile':profile,'reference':reference, **{key:mean_ci(match,key+'_difference') for key in ['joint','jerk_gain','any_success','hold_fraction']}})
    frame.to_csv(out/'normalized_evaluation.csv',index=False)
    pd.DataFrame(calls).to_csv(out/'delivered_corrections.csv',index=False)
    write_json(out/'normalized_analysis.json',{'smoke':smoke,'summary':summary,'paired':paired,'old_vs_normalized_active_rows_changed':int(np.sum(old_active!=frame.active.to_numpy())), 'inputs':{name:sha256(out/name) for name in ['manifest.json','raw_audit.json','evaluation.csv']},'analysis_source_sha256':sha256(Path(__file__)), 'scope':'Original physical audit retained. Applied correction uses original normalized action norm >1e-6 on delivered commands only; pending or undelivered correction is not active. Four development families are diagnostic; smoke is implementation verification only. No performance-based retuning.'})
    print('NORMALIZED AUDIT',audit['episodes'],'episodes;',len(calls),'calls; active classification changes',int(np.sum(old_active!=frame.active.to_numpy())),flush=True)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    summarize(parser.parse_args().smoke)
