"""Resume interrupted, already locked experiments in a serial GPU queue.

This changes scheduling only. Each original runner verifies its own protocol
and skips its completed units; original analyzers remain required.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/followup_v1/protocol/resume_queue'
OUT.mkdir(exist_ok=True)


def save(value):
    p = OUT / 'status.json'
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(p)


def main():
    lock = (OUT / 'process.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stages = [
        ('cube_runtime', 'run_followup_cube_runtime.py', []),
        ('cube_runtime_audit', 'analyze_followup_cube_runtime.py', []),
        ('cube_pure_budget', 'analyze_followup_cube_runtime_pure.py', []),
        ('policy_final', 'run_followup_policy_final.py', ['run']),
        ('latency_development', 'run_followup_policy_latency_pipeline.py', ['--dependency-pid', str(os.getpid())]),
        ('latency_final', 'run_followup_policy_latency_final.py', ['continue-when-ready', '--dependency-pid', str(os.getpid())]),
        ('latency_final_report', 'report_followup_policy_latency_final.py', []),
    ]
    state = {'pid': os.getpid(), 'started_unix': time.time(), 'completed': [],
             'scope': 'Resume existing protocols after prior processes were confirmed absent; serial scheduling avoids contention among these jobs.'}
    for name, script, args in stages:
        state.update(status='running', stage=name, updated_unix=time.time())
        save(state)
        print('START', name, flush=True)
        with (OUT / (name + '.log')).open('a') as log:
            child = subprocess.Popen([sys.executable, str(ROOT / 'scripts' / script), *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save(state)
            rc = child.wait()
        state.update(returncode=rc, updated_unix=time.time())
        if rc:
            state['status'] = 'failed'
            save(state)
            raise SystemExit(rc)
        state['completed'].append(name)
        print('COMPLETE', name, flush=True)
    state.update(status='complete', updated_unix=time.time())
    save(state)


if __name__ == '__main__':
    main()
