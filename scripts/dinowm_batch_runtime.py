"""Native DINO candidate batching; same normalized actions and frozen model."""
import numpy as np
import torch


def predict_batch(dino,encoded,past,actions,batch_size=8):
    assert batch_size>0 and encoded.shape[0]==1
    past=torch.as_tensor(past,device=dino.device,dtype=torch.float32)
    future=torch.as_tensor(actions,device=dino.device,dtype=torch.float32)
    assert past.shape==(10,len(dino.mean)) and future.shape[1:]==(5,len(dino.mean))
    output=[]
    with torch.no_grad():
        for start in range(0,len(future),batch_size):
            block=future[start:start+batch_size];n=len(block)
            grouped=((torch.cat([past.unsqueeze(0).expand(n,-1,-1),block],1)-dino.mean)/dino.scale).reshape(n,3,5*len(dino.mean))
            output.append(dino.model.rollout_last_visual(encoded.expand(n,-1,-1,-1),grouped).cpu().numpy())
    return np.concatenate(output)
