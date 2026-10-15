"""F6 matched control audit and paired family intervals for internal effects."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from zj_wam.io import resolve,write_json
from scripts.analyze_followup_effects import family_interval

out=resolve('artifacts/followup_v1/activation');f=pd.read_csv(out/'candidates.csv')
assert len(f)==4800
assert f[f.kind=='sham'].latent_shift.max()==0
rows=[];maximum_norm_error=0.
keys=['state_id','index','family','split','layer','receiver','dose']
for name in ['patch','ablation']:
    target=f[f.kind=='target_'+name].set_index(keys)
    random=f[f.kind=='random_'+name].groupby(keys).mean(numeric_only=True)
    paired=target[['intervention_norm','latent_shift','donor_projection_fraction','object_effect_position_error','object_effect_rotation_error_deg','latent_prediction_error']].join(random,rsuffix='_random',validate='one_to_one').reset_index()
    maximum_norm_error=max(maximum_norm_error,float(np.max(np.abs(paired.intervention_norm-paired.intervention_norm_random))))
    for split,g in paired.groupby('split'):
      for receiver,k in g.groupby('receiver'):
        one=k[k.dose==1.]
        for label,v in [('all_layers',one)]+[(f'layer_{i}',one[one.layer==i]) for i in range(6)]:
            row={'intervention':name,'split':split,'receiver':receiver,'layer':label,'cases':v.state_id.nunique(),'families':v.family.nunique()}
            for column in ['latent_shift','donor_projection_fraction','object_effect_position_error','object_effect_rotation_error_deg','latent_prediction_error']:
                delta=v[column]-v[column+'_random']
                row[column+'_target_minus_random']=float(delta.mean());row[column+'_ci']=family_interval(delta,v.family)
            rows.append(row)
assert maximum_norm_error<1e-5,maximum_norm_error
write_json(out/'analysis.json',{'comparisons':rows,'maximum_norm_matching_error':maximum_norm_error,'sham_max_latent_shift':0.,
           'scope':'Fixed development-selected 16 channels, matched count/norm, two patch directions, two doses; calibration 8 independent diagnostic families. Internal influence, not proof of physical safety or an intact-model performance improvement.'})
print(json.dumps([r for r in rows if r['split']=='calibration' and r['receiver']=='modified' and r['layer']=='all_layers'],indent=2))
