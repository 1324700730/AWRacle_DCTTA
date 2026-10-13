"""DCTTA's online Residual Diffusion TRG, adapted from RDDM/net.py + Trainer.

The official DCTTA run constructs one generator per test stream, then makes five
RAdam updates per query. Its Trainer EMA is synchronized to the online model by
ema_model.init() immediately before sampling, so sampling uses current weights.
"""
import torch
from torch.optim import RAdam
from models.dctta.rddm.src.residual_denoising_diffusion_pytorch import UnetRes, ResidualDiffusion

class OnlineResidualDiffusionTRG:
    def __init__(self, device, image_size=256, train_steps=5):
        self.device=torch.device(device)
        self.train_steps=int(train_steps)
        self.net=UnetRes(dim=64,dim_mults=(1,2,4,8),num_unet=1,condition=True,
                         input_condition=False,objective='pred_res',test_res_or_noise='res').to(self.device)
        self.diffusion=ResidualDiffusion(self.net,image_size=image_size,timesteps=1000,
                         sampling_timesteps=5,objective='pred_res',loss_type='l2',
                         condition=True,sum_scale=0.01,input_condition=False,
                         input_condition_mask=False,test_res_or_noise='res').to(self.device)
        self.optimizer=RAdam(self.diffusion.parameters(),lr=2e-4,weight_decay=0.0)

    def __call__(self, degraded, teacher_target):
        degraded=degraded.detach().to(self.device)
        teacher_target=teacher_target.detach().to(self.device)
        losses=[]
        self.diffusion.train()
        for _ in range(self.train_steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss_list=self.diffusion([teacher_target,degraded])
            loss=sum(loss_list)/len(loss_list)
            # Official Trainer uses gradient_accumulate_every=2 with one batch
            # per inner iteration, so each backward is divided by two.
            (loss/2).backward()
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(),1.0)
            self.optimizer.step()
            losses.append(float(loss.detach()))
        self.diffusion.eval()
        with torch.no_grad():
            # Official test() resets EMA weights to the latest online model,
            # then samples 5-step deterministic DDIM (eta=0; initial noise seeded).
            generated=self.diffusion.sample(degraded,batch_size=degraded.shape[0],last=True)[-1]
        if generated.shape != degraded.shape:
            raise RuntimeError(f'TRG shape mismatch: {generated.shape} vs {degraded.shape}')
        if not torch.isfinite(generated).all():
            raise FloatingPointError('TRG generated NaN/Inf')
        return generated, {'trg_train_loss_mean':sum(losses)/len(losses),'trg_steps':len(losses)}
