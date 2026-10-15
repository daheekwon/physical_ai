"""Batched native LeWM inference and prespecified representation readouts."""
import numpy as np
import torch


@torch.no_grad()
def encode_representations(adapter, images, batch_size=32):
    outputs={k:[] for k in ['projected_cls','encoder_cls','middle_cls','spatial_quadrants']}
    for start in range(0,len(images),batch_size):
        pixels=torch.stack([adapter.transform(im) for im in images[start:start+batch_size]]).to(adapter.device)
        result=adapter.model.encoder(pixels,interpolate_pos_encoding=True,output_hidden_states=True)
        hidden=result.last_hidden_state
        cls=hidden[:,0]
        patches=hidden[:,1:].reshape(len(pixels),16,16,-1)
        quadrants=torch.cat([patches[:,a:a+8,b:b+8].mean((1,2)) for a,b in [(0,0),(0,8),(8,0),(8,8)]],-1)
        values={'projected_cls':adapter.model.projector(cls),'encoder_cls':cls,
                'middle_cls':result.hidden_states[6][:,0],'spatial_quadrants':quadrants}
        for k,v in values.items():outputs[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in outputs.items()}


@torch.no_grad()
def predict_candidates(adapter, history_embedding, past_normalized, raw_actions, batch_size=64):
    """Return (candidate, block, latent), native five-action blocks only."""
    result=[]
    for start in range(0,len(raw_actions),batch_size):
        raw=np.asarray(raw_actions[start:start+batch_size]);n,h,_=raw.shape
        if h%5:raise ValueError('Native predictor requires complete five-action blocks')
        actions=torch.as_tensor(adapter.normalize(raw.reshape(-1,2)).reshape(n,-1,10),device=adapter.device)
        past=past_normalized.expand(n,-1,-1)
        encoded_actions=adapter.model.action_encoder(torch.cat([past,actions],1))
        emb=history_embedding.expand(n,-1,-1);future=[]
        for i in range(h//5):
            end=history_embedding.shape[1]+i;begin=max(0,end-3)
            z=adapter.model.predict(emb[:,begin:],encoded_actions[:,begin:end])[:,-1:]
            future.append(z);emb=torch.cat([emb,z],1)
        result.append(torch.cat(future,1).cpu().numpy())
    return np.concatenate(result)
