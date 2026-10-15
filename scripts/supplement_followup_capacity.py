"""Match spatial/history readout width to 192 latent dimensions without altering frozen primary probes."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scripts.run_followup_probes import load_split,physical_errors
from zj_wam.io import resolve,write_json

out=resolve('artifacts/followup_v1');train=load_split(out,'probe_train');dev=load_split(out,'development');cal=load_split(out,'calibration');rows=[]
for name in ['spatial_quadrants','history_projected']:
    scale=StandardScaler().fit(train[name]);pca=PCA(n_components=192,svd_solver='full').fit(scale.transform(train[name]))
    x=pca.transform(scale.transform(train[name]));xd=pca.transform(scale.transform(dev[name]));xc=pca.transform(scale.transform(cal[name]))
    ys=StandardScaler().fit(train['targets'][:,:6]);y=ys.transform(train['targets'][:,:6]);choices=[]
    for alpha in [.1,1.,10.,100.,1000.]:
        r=Ridge(alpha=alpha).fit(x,y);p=ys.inverse_transform(r.predict(xd));e=physical_errors(p,dev['targets'][:,:6])
        choices.append((e['position_mae']/2+e['rotation_mae_deg']/2,alpha,r,e))
    _,alpha,r,development=min(choices,key=lambda z:z[0]);prediction=ys.inverse_transform(r.predict(xc))
    rows.append({'representation':name,'pca_dimensions':192,'linear_parameters':192*6+6,'alpha':alpha,
                 'pca_train_variance_retained':float(pca.explained_variance_ratio_.sum()),
                 'development':development,'calibration':physical_errors(prediction,cal['targets'][:,:6])})
write_json(out/'lewm_audit/readouts/matched_capacity.json',{'scope':'Additional capacity control; PCA fitted on probe_train only, alpha selected on development; primary deployed/probe choices unchanged','rows':rows})
print(json.dumps(rows,indent=2))
