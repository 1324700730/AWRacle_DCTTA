"""DCTTA losses from DCTTA/utils/loss_utils.py, with explicit local weight paths."""
from pathlib import Path
import scipy.io as sio
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg19, VGG19_Weights

class WaveletTransform(nn.Module):
    def __init__(self, path, scale=1, cdim=3):
        super().__init__(); self.scale=scale; self.ks=2**scale; self.cdim=cdim
        mat=sio.loadmat(str(path)); self.weight=nn.Parameter(torch.from_numpy(mat[f'rec{self.ks}'][:self.ks*self.ks]),requires_grad=False)
    def forward(self,x):
        b,h,w,c=x.shape; x=x.permute(0,3,1,2); weight=(self.weight/self.ks).to(x.device)
        y=F.conv2d(x.reshape(b*c,1,h,w),weight,stride=self.ks)
        return y.reshape(b,c,y.shape[1],y.shape[2],y.shape[3]).permute(0,2,3,4,1)

class DCTTAPerceptualLoss(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        if weights_path is None:
            weights_path=Path(torch.hub.get_dir())/'checkpoints'/'vgg19-dcbb9e9d.pth'
        weights_path=Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f'DCTTA requires ImageNet VGG19 weights at {weights_path}; provide --vgg19-weights. No substitute perceptual loss is used.')
        vgg=vgg19(weights=None); state=torch.load(weights_path,map_location='cpu',weights_only=True); vgg.load_state_dict(state,strict=True)
        layers=list(vgg.features.children()); ends=[2,7,16,25,34]; starts=[0]+[i+1 for i in ends[:-1]]
        self.blocks=nn.ModuleList([nn.Sequential(*layers[s:e+1]) for s,e in zip(starts,ends)])
        for p in self.parameters(): p.requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1)); self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        self.weights=(.1,.1,1.,1.,1.)
    def forward(self,x,target):
        x=(x-self.mean)/self.std; target=(target.detach()-self.mean)/self.std
        loss=x.new_zeros(())
        for block,w in zip(self.blocks,self.weights):
            x=block(x); target=block(target); loss=loss+w*F.l1_loss(x,target)
        return loss

def dctta_consistency_loss(pred,target,confidence,pred_wave,target_wave,perceptual):
    # Match DCTTA compute_loss2: confidence-weighted L1, VGG perceptual,
    # and L1 between the source's scale-1 (2x2 block) DCT coefficients.
    return (F.l1_loss(confidence*pred,confidence*target.detach())
            +0.01*perceptual(pred,target)
            +0.1*F.l1_loss(pred_wave,target_wave.detach()))
