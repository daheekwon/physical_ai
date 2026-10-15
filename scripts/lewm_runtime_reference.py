"""Native single-candidate CPU reference, independent of the runtime batch path."""
import numpy as np
import torch


def native_scores(env,adapter,context,actions):
    if env=='pusht':predict,_=adapter.predictor(context['images'],context['past'])
    else:_,predict=adapter.predictor(context['images'],context['past'])
    values=[]
    with torch.no_grad():
        for action in actions:
            if env=='pusht':normalized=adapter.normalize(action).reshape(-1)
            else:normalized=((action-context['mean'])/context['scale']).reshape(-1).astype(np.float32)
            values.append(predict(torch.from_numpy(normalized)).numpy())
    values=np.asarray(values)
    return np.linalg.norm(values-values[:1],axis=1)
