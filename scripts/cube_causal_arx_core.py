"""Small causal EEF response fits; future physical branches never enter fitting."""
import numpy as np
from zj_wam.cube_experiment import fit_response

def training(points,targets):
    p=np.asarray(points,float);v=np.diff(p,axis=0);e=np.asarray(targets,float)[1:]-p[1:-1];e[:,3]=np.arctan2(np.sin(e[:,3]),np.cos(e[:,3]));return np.c_[v[:-1],e],v[1:]

def radius(A,B):
    I=np.eye(len(A));return float(np.max(abs(np.linalg.eigvals(np.block([[I-B,A],[-B,A]])))))

def fit(points,targets,variant,cfg):
    points=np.asarray(points)[-cfg['history_physics_steps']-1:];targets=np.asarray(targets)[-cfg['history_physics_steps']:];g,residual=fit_response(points,targets,cfg['physics_dt']);A=np.diag(1-cfg['physics_dt']*g[:,1]);B=np.diag(cfg['physics_dt']**2*g[:,0]);c=np.zeros(4);X,Y=training(points,targets);records=[]
    if variant=='past_pd':return {'A':A,'B':B,'c':c,'records':records,'supported_axes':0,'gains':g,'residual':residual}
    groups=[([i,i+4],[i]) for i in range(4)] if variant=='diagonal_arx' else [(list(range(8)),list(range(4)))];supported=0
    for columns,outputs in groups:
        x=X[:,columns];y=Y[:,outputs];mu=x.mean(0);std=x.std(0);center=y.mean(0);valid=bool(np.all(std>=cfg['minimum_feature_std']));ratio=0.;rho=None;reason='unexcited'
        if valid:
            xn=(x-mu)/std;singular=np.linalg.svd(xn,compute_uv=False);ratio=float(singular[-1]/singular[0]);valid=bool(ratio>=cfg['minimum_singular_ratio']);reason='rank_deficient'
        if valid:
            w=np.linalg.solve(xn.T@xn+cfg['ridge']*np.eye(len(columns)),xn.T@(y-center));coef=(w/std[:,None]).T;intercept=center-coef@mu;trialA=A.copy();trialB=B.copy();trialc=c.copy()
            for j,k in enumerate(outputs):
                trialA[k,:]=0;trialB[k,:]=0
                for q,col in enumerate(columns):
                    if col<4:trialA[k,col]=coef[j,q]
                    else:trialB[k,col-4]=coef[j,q]
                trialc[k]=intercept[j]
            rho=radius(trialA[np.ix_(outputs,outputs)],trialB[np.ix_(outputs,outputs)]);valid=bool(np.isfinite(coef).all() and rho<=cfg['maximum_spectral_radius']);reason='unstable'
            if valid:A,B,c=trialA,trialB,trialc;supported+=len(outputs);reason='accepted'
        records.append({'columns':columns,'outputs':outputs,'accepted':valid,'reason':reason,'minimum_std':float(std.min()),'singular_ratio':ratio,'spectral_radius':rho})
    return {'A':A,'B':B,'c':c,'records':records,'supported_axes':supported,'gains':g,'residual':residual}

def predict(actions,root,previous_increment,model,bounds,cfg):
    p=np.asarray(root,float).copy();d=np.asarray(previous_increment,float).copy();trace=[]
    for action in actions:
        target=p+np.asarray(action[:4])*np.array([.05,.05,.05,.3]);target[:3]=np.clip(target[:3],bounds[0],bounds[1])
        for _ in range(cfg['substeps']):
            e=target-p;e[3]=np.arctan2(np.sin(e[3]),np.cos(e[3]));d=model['A']@d+model['B']@e+model['c'];p=p+d;trace.append(p.copy())
    return np.array(trace)
