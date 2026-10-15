import numpy as np,torch
from scripts.run_followup_long_horizon import rollout

class FakeModel:
    def action_encoder(self,x):return x
    def predict(self,emb,act):return emb.cumsum(1)+act.mean(-1,keepdim=True)
class FakeAdapter:
    device='cpu';model=FakeModel()
    def normalize(self,x):return np.asarray(x,dtype=np.float32)

def test_refresh_never_consumes_current_or_future_target():
    model=FakeAdapter();context=torch.ones((1,3,2));past=torch.zeros((1,2,10))
    actions=np.zeros((2,20,2));real=np.arange(16,dtype=np.float32).reshape(2,4,2)
    for refresh in [None,1,2]:
        base,_=rollout(model,context,past,actions,real,refresh)
        for b in range(4):
            changed=real.copy();changed[:,b:]+=1000
            actual,_=rollout(model,context,past,actions,changed,refresh)
            np.testing.assert_array_equal(base[:,:b+1],actual[:,:b+1])
    updated,_=rollout(model,context,past,actions,real+1000,1)
    original,_=rollout(model,context,past,actions,real,1)
    assert not np.array_equal(updated[:,1:],original[:,1:])
