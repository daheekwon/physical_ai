"""Additional probabilities and uncertainty; never refits calibration."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from zj_wam.io import resolve,write_json
from zj_wam.metrics import cluster_bootstrap

out=resolve('artifacts/phase1');f=pd.read_csv(out/'heldout_candidates.csv')
cfg=json.loads((out/'manifest.json').read_text())['config'];rng=np.random.default_rng(cfg['seed']+991)
rows=[]
for phase in ['all','contact','free_motion','near_contact']:
    phase_data=f if phase=='all' else f[f.interaction_phase==phase]
    for radius in cfg['relative_radii']:
        g=phase_data[np.isclose(phase_data.relative_radius,radius)]
        low=g[g.direction=='low'][['episode','state_id','sign','object_error']].rename(columns={'object_error':'low_error'})
        for group in ['random','high']:
            pairs=g[g.group==group].merge(low,on=['episode','state_id','sign'])
            if pairs.empty:continue
            ties=np.isclose(pairs.low_error,pairs.object_error,rtol=1e-7,atol=1e-10)
            pairs['win']=(pairs.low_error<pairs.object_error)&~ties
            per=pairs.groupby(['episode','state_id']).win.mean().reset_index()
            rows.append({'phase':phase,'relative_radius':radius,'comparison':f'low_vs_{group}_draw',
                         'probability':float(per.win.mean()),'probability_ci':cluster_bootstrap(per.win,per.episode,rng,cfg['bootstrap_samples']),
                         'tie_rate':float(ties.mean()),'comparisons':len(pairs),'states':len(per),'families':per.episode.nunique()})
# Primary correlation CI, preserving every state/sign/direction within a family.
primary=f[(f.interaction_phase=='contact')&np.isclose(f.relative_radius,.1)&f.group.isin(['low','random','high'])]
ids=primary.episode.unique();groups=[primary[primary.episode==i] for i in ids]
corr={}
for predictor in ['directional_sensitivity','predicted_change']:
    pairs=[g[[predictor,'object_error']].to_numpy() for g in groups]
    estimates=[]
    for _ in range(cfg['bootstrap_samples']):
        a=np.concatenate([pairs[i] for i in rng.integers(len(pairs),size=len(pairs))])
        estimates.append(float(spearmanr(a[:,0],a[:,1]).statistic))
    corr[predictor]={'spearman':float(spearmanr(primary[predictor],primary.object_error).statistic),
                     'cluster_ci':np.quantile(estimates,[.025,.975]).tolist()}
# All three action allowance basis/mix methods at their actual maximum selection.
d=f[f.deployed_maximum&f.group.isin(['low','low_basis','low_mix'])]
deployment={}
for phase in ['all','contact']:
    g=d if phase=='all' else d[d.interaction_phase==phase]
    per=g.groupby(['episode','state_id']).preserved.mean().reset_index()
    deployment[phase]={'preservation_rate':float(per.preserved.mean()),'ci':cluster_bootstrap(per.preserved,per.episode,rng,cfg['bootstrap_samples']),
                       'selected_corrections':len(g),'states':len(per)}
write_json(out/'supplement_statistics.json',{'pairwise_probabilities':rows,'primary_contact_correlation':corr,'deployed_low_basis_and_mixes':deployment})
print(json.dumps({'primary_contact_correlation':corr,'deployment':deployment},indent=2))
