"""Repair only score serialization; preserve failed exports and rerun frozen audit."""
import csv, io, json, os, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from zj_wam.io import resolve, sha256, write_json
BASE = resolve('artifacts/followup_v1')
OUT = BASE / 'dinowm_policy_bank'
ARCHIVE = BASE / 'policy_bank_csv_export_failure_v0'
read = lambda p: json.loads(p.read_text())

def main():
    for pid in [3631115, 3639823]:
        try: os.kill(pid, 0)
        except ProcessLookupError: pass
        else: raise RuntimeError(('Previous pipeline still live', pid))
    assert not (OUT / 'pipeline_complete.json').exists()
    assert read(OUT / 'status.json')['status'] == 'failed'
    lock = read(OUT / 'pipeline_lock.json')
    manifest = read(OUT / 'manifest.json')
    done = read(OUT / 'execution_complete.json')
    assert done['states'] == 108 and done['candidates'] == 5940
    for p, h in {**lock['inputs'], **manifest['inputs']}.items():
        assert sha256(resolve(p)) == h, p
    ARCHIVE.mkdir(exist_ok=False)
    for name in ['manifest.json', 'pipeline_lock.json', 'execution_complete.json', 'status.json', 'audit.log']:
        shutil.copy2(OUT / name, ARCHIVE / name)
    for name in ['run_followup_dinowm_policy_bank.py', 'audit_followup_dinowm_policy_bank.py', 'run_followup_dinowm_policy_bank_pipeline.py']:
        shutil.copy2(resolve('scripts') / name, ARCHIVE / name)
    selection = BASE / 'dinowm_policy_selection'
    for name in ['pipeline_lock.json', 'status.json']:
        shutil.copy2(selection / name, ARCHIVE / ('selection_' + name))
    shutil.copy2('/tmp/zj_policy_selection_pipeline.log', ARCHIVE / 'selection_failure.log')
    records = []
    for spec in manifest['states']:
        p = OUT / spec['split'] / f"{spec['seed']}_t{spec['step']:03d}.csv"
        m = read(p.with_suffix('.json'))
        raw = p.with_suffix('.npz')
        assert sha256(raw) == m['data_sha256']
        dest = ARCHIVE / p.relative_to(OUT)
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(p, dest)
        reader = csv.DictReader(io.StringIO(p.read_text()))
        fields = reader.fieldnames
        rows = list(reader)
        before = [row.copy() for row in rows]
        errors = {}
        with np.load(raw) as a:
            assert len(rows) == len(a['actions'])
            for method in manifest['config']['methods']:
                values = a['score_' + method]
                old = np.array([float(r[method]) for r in rows])
                np.testing.assert_array_equal(old.astype(values.dtype), values)
                finite = np.isfinite(values)
                errors[method] = float(np.max(np.abs(old[finite] - values[finite].astype(float)))) if finite.any() else 0.
                for row, value in zip(rows, values): row[method] = repr(float(value))
        for a, b in zip(before, rows):
            assert all(a[k] == b[k] for k in fields if k not in manifest['config']['methods'])
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
        records.append({'path': str(p.relative_to(resolve('.'))), 'original_sha256': sha256(dest), 'repaired_sha256': sha256(p), 'raw_sha256': sha256(raw), 'original_numeric_errors': errors})
    evidence = {'reason': 'Float32 shortest-decimal CSV export reread as float64; one LeWM score error 1.00128174e-7 exceeds unchanged 1e-7 audit tolerance.', 'scope': 'Score columns reserialized as exact float64 promotion of original raw values. Non-score cells unchanged; no model/physics recomputation, no threshold change, no source or original protocol changes.', 'records': records, 'source_sha256': sha256(Path(__file__)), 'unix': time.time()}
    write_json(ARCHIVE / 'recovery.json', evidence)
    write_json(OUT / 'export_recovery.json', {'archive': str(ARCHIVE.relative_to(resolve('.'))), 'evidence_sha256': sha256(ARCHIVE / 'recovery.json'), 'source_sha256': sha256(Path(__file__)), 'records': len(records)})
    write_json(OUT / 'status.json', {'pid': os.getpid(), 'stage': 'audit_after_export_recovery', 'status': 'running', 'updated_unix': time.time()})
    with (OUT / 'audit_after_export_recovery.log').open('w') as log:
        result = subprocess.run([sys.executable, str(resolve('scripts/audit_followup_dinowm_policy_bank.py'))], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(OUT / 'status.json', {'pid': os.getpid(), 'stage': 'audit_after_export_recovery', 'status': 'failed', 'returncode': result.returncode})
        raise RuntimeError(result.returncode)
    a = read(OUT / 'raw_audit.json')
    assert a['states'] == 108 and a['candidates'] == 5940
    for p, h in {**lock['inputs'], **manifest['inputs'], **a['inputs'], **a['audit_helpers']}.items(): assert sha256(resolve(p)) == h, p
    write_json(OUT / 'pipeline_complete.json', {'states': 108, 'candidates': 5940, 'raw_audit_sha256': sha256(OUT / 'raw_audit.json'), 'pipeline_lock_sha256': sha256(OUT / 'pipeline_lock.json'), 'export_recovery_sha256': sha256(OUT / 'export_recovery.json'), 'unix': time.time(), 'scope': 'Full raw collection audited with original frozen auditor after exact score CSV export recovery; selection pending.'})
    write_json(OUT / 'status.json', {'status': 'complete', 'updated_unix': time.time()})
    print('RECOVERED AND AUDITED POLICY BANK108', flush=True)
    with (selection / 'recovered_pipeline.log').open('w') as log:
        result = subprocess.run([sys.executable, str(resolve('scripts/run_followup_dinowm_policy_selection_pipeline.py')), '--bank-pid', '3631115'], stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.returncode
    print('POLICY SELECTION COMPLETE AFTER AUDITED RECOVERY', flush=True)

if __name__ == '__main__': main()
