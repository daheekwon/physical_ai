"""F3 label-budget and family-level uncertainty, using development only."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scripts.run_followup_probes import load_split,physical_errors
from zj_wam.followup import angular_error
from zj_wam.io import resolve,write_json


def main():
    out=resolve('artifacts/followup_v1');directory=out/'lewm_audit/readouts';selection=json.loads((directory/'selection.json').read_text())
    train=load_split(out,'probe_train');dev=load_split(out,'development');pred=dict(np.load(directory/'development_predictions.npz'))
    rng=np.random.default_rng(20261011);families=rng.permutation(np.unique(train['family']));rows=[];cis=[]
    for name,choice in selection['selected_ridge'].items():
      dims=9 if name=='history_projected' else 6
      for count in [5,10,25,50,100]:
        take=np.isin(train['family'],families[:count]);xs=StandardScaler().fit(train[name][take]);ys=StandardScaler().fit(train['targets'][take,:dims])
        reg=Ridge(alpha=choice['alpha']).fit(xs.transform(train[name][take]),ys.transform(train['targets'][take,:dims]))
        prediction=ys.inverse_transform(reg.predict(xs.transform(dev[name])))
        rows.append({'representation':name,'families':count,'labels':int(take.sum()),'alpha':choice['alpha'],**physical_errors(prediction,dev['targets'][:,:dims])})
      for kind in ['ridge','mlp64','shuffle']:
        p=pred[name+'_'+kind];truth=dev['targets'][:,:dims]
        values={'position':np.linalg.norm(p[:,:2]-truth[:,:2],axis=1),
                'rotation_deg':np.rad2deg(angular_error(np.arctan2(p[:,2],p[:,3]),np.arctan2(truth[:,2],truth[:,3])))}
        for metric,v in values.items():
          per_family=np.array([v[dev['family']==f].mean() for f in np.unique(dev['family'])])
          draws=per_family[rng.integers(0,len(per_family),(2000,len(per_family)))].mean(1)
          cis.append({'representation':name,'readout':kind,'metric':metric,'mean':float(v.mean()),'ci':np.quantile(draws,[.025,.975]).tolist()})
    pd.DataFrame(rows).to_csv(directory/'label_budget.csv',index=False)
    write_json(directory/'family_uncertainty.json',{'scope':'Development-selected readouts; conditional CIs, final test still locked',
               'label_budget_rule':'Nested seeded train families; regularization fixed to the full-training development choice; no neural weights changed',
               'intervals':cis})
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
