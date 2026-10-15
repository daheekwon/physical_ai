"""Restore real shifted histories without giving changed physics parameters to selectors."""
import numpy as np
from zj_wam.followup import DenseSimulator

class ShiftedSimulator(DenseSimulator):
 def __init__(self,*args,condition=None,lead_actions=10,**kwargs):
  super().__init__(*args,**kwargs);self.condition=condition or {};self.shift_step=max(0,self.timestep-lead_actions)
 def apply_shift(self):
  if 'pair_friction' in self.condition:
   for body in [self.env.agent,self.env.block]:
    for shape in body.shapes:shape.friction=np.sqrt(self.condition['pair_friction'])
  if 'kp_multiplier' in self.condition:self.env.k_p=100*self.condition['kp_multiplier']
 def restore(self,capture=False):
  original=self.env.step;self.env.k_p=100.;self.env.k_v=20.;index=0
  def step(action):
   nonlocal index
   if index==self.shift_step:self.apply_shift()
   index+=1;return original(action)
  self.env.step=step
  try:return super().restore(capture=capture)
  finally:self.env.step=original

def noisy_history(images,condition,seed):
 images=np.asarray(images,dtype=np.uint8)
 if not condition.get('rgb_noise_std',0):return images.copy()
 rng=np.random.default_rng(seed);noise=rng.normal(0,condition['rgb_noise_std'],images.shape)
 return np.clip(np.rint(images.astype(float)+noise),0,255).astype(np.uint8)
