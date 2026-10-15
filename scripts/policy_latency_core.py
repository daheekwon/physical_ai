"""Explicit simulated wait-clock and target-hold semantics for latency experiments."""
import math
import numpy as np


def delay_ticks(seconds,dt=.1):
    if not np.isfinite(seconds) or seconds<0 or dt<=0:raise ValueError('Invalid delay')
    return int(math.ceil(max(0.,seconds/dt-1e-12)))


def holding_action(absolute_target,current_position):
    return np.clip((np.asarray(absolute_target)-np.asarray(current_position))/100.,-1.,1.)


def draw_delay(samples,seed,call_index,method,methods,profile,eligible,base_seed):
    if profile=='zero':return {'acquisition':0.,'policy':0.,'refinement':0.,'total':0.,'ticks':0}
    rng=np.random.default_rng(np.random.SeedSequence([base_seed,seed,call_index]))
    acquisition=float(samples['acquisition'][rng.integers(len(samples['acquisition']))]);policy=float(samples['policy'][rng.integers(len(samples['policy']))]);refinement=0.
    if profile=='full' and eligible and method!='nominal':
        other=np.random.default_rng(np.random.SeedSequence([base_seed,seed,call_index,methods.index(method)+1]));values=samples['refinement'][method];refinement=float(values[other.integers(len(values))])
    total=acquisition+policy+refinement
    return {'acquisition':acquisition,'policy':policy,'refinement':refinement,'total':total,'ticks':delay_ticks(total)}
