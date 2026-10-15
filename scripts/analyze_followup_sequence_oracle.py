"""Recompute all stored candidate outcomes and replay selected witnesses."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from zj_wam.io import resolve, sha256, write_json
from scripts.analyze_followup_policy_final import Coverage, binomial_interval
from scripts.run_followup_policy_terminal import replay

read = lambda p: json.loads(p.read_text())


def audit_one(args):
    out, seed = args
    out = Path(out); manifest = read(out/'manifest.json'); cfg = manifest['config']
    path = out/f'{seed}.json'; meta = read(path); frame = pd.read_csv(path.with_suffix('.csv'))
    assert meta['manifest_sha256'] == sha256(out/'manifest.json')
    for ext, h in meta['files'].items(): assert sha256(path.with_suffix(ext)) == h
    errors = {}
    def check(name, a, b, tolerance=1e-9):
        error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        errors[name] = max(errors.get(name, 0.), error)
        assert error <= tolerance, (seed, name, error)
    def energy(q):
        # Explicit third finite difference, independent of the runner helper.
        jerk = (q[3:]-3*q[2:-1]+3*q[1:-2]-q[:-3])/.01**3
        return float(np.sum(jerk**2)*.01)
    with np.load(path.with_suffix('.npz')) as z:
        coef, physics, cov = z['coefficients'], z['physics'], z['coverage']
        basis, scale, offsets, nominal, initial = [z[k] for k in ['basis','scale','offsets','nominal','initial']]
    count = len(frame)
    assert len(coef) == len(physics) == len(cov) == count == meta['candidates']
    assert physics.shape[1:] == (3000, 5) and cov.shape[1:] == (300,)
    assert list(offsets) == list(range(16,293,8)) and coef.shape[1:] == (35,4)
    check('orthonormal_basis', basis.T@basis, np.eye(4))
    check('basis_endpoint', basis[[0,1,-2,-1]], np.zeros((4,4)))
    with np.load(resolve(cfg['source'])/f'{seed}_nominal.npz') as z:
        check('original_nominal_trace', physics[0], z['physics'][:,:5], 0)
        check('original_nominal_coverage', cov[0], z['coverage'], 0)
        check('original_nominal_actions', nominal, z['actions'], 0)
        check('original_initial', initial, z['initial_state'], 0)
    reference = physics[0]; reference_energy = energy(np.vstack([initial[:2], reference[:,:2]]))
    coverage = Coverage(seed)
    selected = {0, meta['greedy_index'], meta['best_index'], int(frame.object_error.idxmax())}
    independently = []
    for row in frame.itertuples():
        i = row.index; trace = physics[i]; coefficients = coef[i]
        check('unmodified_prefix_physics', trace[:160], reference[:160], 0)
        assert np.linalg.norm(coefficients, axis=1).max() <= cfg['relative_radius']*np.sqrt(10)+1e-9
        action = nominal.copy()
        for j, offset in enumerate(offsets):
            delta = (basis@coefficients[j]).reshape(5,2)
            action[offset:offset+5] += delta*scale
            check('polynomial_projection', delta.reshape(-1), basis@(basis.T@delta.reshape(-1)))
        assert np.max(np.abs(action)) <= 1+1e-12
        untouched = np.ones(300,dtype=bool)
        for offset in offsets: untouched[offset:offset+5] = False
        check('unmodified_suffix_and_gaps', action[untouched], nominal[untouched], 0)
        recov = np.asarray([coverage(x) for x in trace[9::10]])
        check('independent_polygon_coverage', recov, cov[i], 1e-10)
        p = np.linalg.norm(trace[:,2:4]-reference[:,2:4], axis=1).max()
        angle = trace[:,4]-reference[:,4]
        r = np.abs(np.arctan2(np.sin(angle),np.cos(angle))).max()
        loss = np.maximum(0., cov[0]-recov).max()
        error = max(p/cfg['position_tolerance'], r/cfg['rotation_tolerance'], loss/cfg['coverage_tolerance'])
        gain = 1-energy(np.vstack([initial[:2],trace[:,:2]]))/reference_energy
        norm = np.linalg.norm(coefficients)
        for name, a, b in [('position',p,row.position_max),('rotation',r,row.rotation_max),
                           ('coverage',loss,row.coverage_loss),('error',error,row.object_error),
                           ('jerk_gain',gain,row.jerk_gain),('norm',norm,row.norm)]:
            check(name,a,b,1e-8)
        useful = bool(norm>1e-6 and error<=1 and gain>=cfg['jerk_gain_threshold'])
        assert useful == row.useful
        independently.append(dict(index=i, object_error=float(error), jerk_gain=float(gain), useful=useful,
                                  active_blocks=int((np.linalg.norm(coefficients,axis=1)>1e-6).sum())))
        if i in selected:
            fresh_initial, fresh_trace, fresh_cov = replay(seed, action)
            check('fresh_initial',fresh_initial,initial,0)
            check('fresh_physics',fresh_trace[:,:5],trace,0)
            check('fresh_coverage',fresh_cov,cov[i],0)
    def order(x): return (0,-x['jerk_gain']) if x['object_error']<=1 else (1,x['object_error'],-x['jerk_gain'])
    best = min(independently,key=order)
    check('best_selected_gain', best['jerk_gain'], independently[meta['best_index']]['jerk_gain'])
    assert best['useful'] == independently[meta['best_index']]['useful']
    greedy = independently[meta['greedy_index']]
    unconstrained = max(independently, key=lambda x:x['jerk_gain'])
    result = dict(seed=seed, candidates=count, fresh_replays=len(selected), max_error=errors,
                  greedy=greedy,best=best,best_ignoring_object_constraint=unconstrained,
                  useful_candidates=sum(r['useful'] for r in independently),
                  manifest_sha256=sha256(out/'manifest.json'), source_sha256=sha256(Path(__file__)))
    write_json(out/f'{seed}_audit.json',result)
    print('AUDITED',seed,count,'candidates',flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();out=resolve('artifacts/followup_v1/sequence_oracle'+('_smoke' if args.smoke else ''))
    manifest=read(out/'manifest.json');done=read(out/'execution_complete.json')
    assert done['manifest_sha256']==sha256(out/'manifest.json')
    for path,h in manifest['inputs'].items(): assert sha256(resolve(path))==h,path
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(audit_one,[(str(out),s) for s in done['seeds']]))
    summary={}
    for method in ['greedy','best']:
        rows=[r[method] for r in results];gains=[r['jerk_gain'] for r in rows]
        summary[method]=dict(useful=binomial_interval(sum(r['useful'] for r in rows),len(rows)),
                            preserved=sum(r['object_error']<=1 for r in rows),
                            active=sum(r['active_blocks']>0 for r in rows),
                            gain_mean=float(np.mean(gains)),gain_median=float(np.median(gains)),
                            gain_min=float(np.min(gains)),gain_max=float(np.max(gains)))
    table=pd.DataFrame([dict(seed=r['seed'],candidates=r['candidates'],
            greedy_gain=r['greedy']['jerk_gain'],best_gain=r['best']['jerk_gain'],
            best_error=r['best']['object_error'],best_useful=r['best']['useful'],
            active_blocks=r['best']['active_blocks'],
            maximum_searched_gain=r['best_ignoring_object_constraint']['jerk_gain']) for r in results])
    table.to_csv(out/'state_metrics.csv',index=False)
    write_json(out/'analysis.json',dict(summary=summary,states=results,scope=manifest['config']['scope']))
    write_json(out/'raw_audit.json',dict(states=len(results),candidates=sum(r['candidates'] for r in results),
        independent_coverage_samples=300*sum(r['candidates'] for r in results),
        fresh_replays=sum(r['fresh_replays'] for r in results),
        manifest_sha256=sha256(out/'manifest.json'),analysis_sha256=sha256(out/'analysis.json'),
        source_sha256=sha256(Path(__file__))))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
