"""Pinned DINO-WM adapter with restricted, explicitly allowlisted loading."""
import hashlib,json,os,sys,pickletools,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]
INVENTORY=ROOT/'artifacts/followup_v1/protocol/dinowm_retrained_inventory'
SOURCE=ROOT/'external/point-lewm-dinowm-audit'
read=lambda p:json.loads(p.read_text())


def verify_assets(env):
    for record in [read(INVENTORY/(env+'_acquired.json')),read(INVENTORY/'source_acquired.json'),read(INVENTORY/'encoder_source.json')]:
        for item in record['files']:assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    weights=read(INVENTORY/'encoder_weights.json');assert hashlib.sha256((ROOT/weights['path']).read_bytes()).hexdigest()==weights['sha256']
    return weights


def restricted_payload(path):
    # These classes were inspected in the hash-pinned source. Do not derive
    # allowlisted classes dynamically from artifact strings and do not fall
    # back to unrestricted pickle loading.
    from models.proprio import ProprioceptiveEmbedding
    from models.vit import Attention,FeedForward,Transformer,ViTPredictor
    allowed=[set,ProprioceptiveEmbedding,Attention,FeedForward,Transformer,ViTPredictor,nn.GELU,nn.Softmax,nn.ModuleList,nn.Sequential,nn.Conv1d,nn.Dropout,nn.Linear,nn.LayerNorm]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path,map_location='cpu',weights_only=True)


class PinnedEncoder(nn.Module):
    def __init__(self,path):
        super().__init__();self.name='dinov2_vits14';self.feature_key='x_norm_patchtokens'
        self.base_model=torch.hub.load(str(ROOT/'external/dinov2-frozen-audit'),self.name,source='local',pretrained=False)
        self.base_model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.emb_dim=self.base_model.num_features;self.patch_size=self.base_model.patch_size;self.latent_ndim=2
    def forward(self,x):return self.base_model.forward_features(x)[self.feature_key]


class FrozenRetrainedDinoWM:
    def __init__(self,env,device='cpu',mixed_precision='no'):
        weights=verify_assets(env);os.environ.setdefault('XFORMERS_DISABLED','1')
        sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'third_party/dino_wm'))
        from models.visual_world_model import VWorldModel
        from dinowm_swm.planning import DinoWMCostModel
        folder=ROOT/'data/checkpoints/dinowm-retrained-fafraob'/env;cfg=OmegaConf.load(folder/'hydra.yaml');assert cfg.num_hist==3 and cfg.frameskip==5 and cfg.concat_dim==1
        payload=restricted_payload(folder/'checkpoints/model_latest.pth')
        encoder=PinnedEncoder(ROOT/weights['path']);prop=payload['proprio_encoder'];act=payload['action_encoder'];assert prop.in_chans==1 and act.in_chans==(10 if env=='pusht' else 25)
        wm=VWorldModel(image_size=cfg.img_size,num_hist=cfg.num_hist,num_pred=cfg.num_pred,encoder=encoder,proprio_encoder=prop,action_encoder=act,decoder=None,predictor=payload['predictor'],proprio_dim=prop.emb_dim,action_dim=act.emb_dim,concat_dim=cfg.concat_dim,num_action_repeat=cfg.num_action_repeat,num_proprio_repeat=cfg.num_proprio_repeat,train_encoder=False,train_predictor=False,train_decoder=False)
        self.stats=read(folder/'swm_h5_norm_stats.json');self.model=DinoWMCostModel(wm,cfg,self.stats,img_size=int(cfg.img_size),mixed_precision=mixed_precision).to(device)
        self.model.eval();self.model.requires_grad_(False);self.device=torch.device(device);self.mean=torch.tensor(self.stats['action_mean'],device=device,dtype=torch.float32);self.scale=torch.tensor(self.stats['action_std'],device=device,dtype=torch.float32)
        self.epoch=payload.get('epoch');self.environment=env;self.mixed_precision=mixed_precision
        assert all(torch.isfinite(p).all() for p in self.model.parameters())

    def pixels(self,images):
        assert np.asarray(images).shape==(3,224,224,3)
        return torch.stack([self.model.img_transform(x) for x in images]).unsqueeze(0).to(self.device)

    def encode(self,images):
        with torch.no_grad():return self.model.encode_visual(self.pixels(images)).detach()

    def normalize_group(self,past,future):
        past=torch.as_tensor(past,device=self.device,dtype=torch.float32);future=torch.as_tensor(future,device=self.device,dtype=torch.float32)
        assert past.shape==(10,len(self.mean)) and future.shape[1]==len(self.mean) and len(future)%5==0
        return ((torch.cat([past,future])-self.mean)/self.scale).reshape(1,-1,5*len(self.mean))

    def predict(self,encoded,past,future,proprio=None):
        # Released noprop adapter always supplies zeros internally; external
        # robot state is deliberately not consumed.
        return self.model.rollout_last_visual(encoded,self.normalize_group(past,future))
