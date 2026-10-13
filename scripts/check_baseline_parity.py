"""One-query CPU/GPU numerical parity check against the untouched source model."""
import argparse, json, sys
from pathlib import Path
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.awracle.model import AWRaCLe as NewModel
from scripts.infer import image_tensor, context_tensor

def load(cls, checkpoint, device):
    model=cls(decoder=True)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    sd={k.removeprefix('net.'):v for k,v in ck['state_dict'].items()}
    model.load_state_dict(sd,strict=True)
    return model.to(device).eval()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True); p.add_argument('--query',required=True)
    p.add_argument('--source-root',required=True,help='Path to the separate original AWRaCLe source checkout')
    p.add_argument('--support-degraded',required=True); p.add_argument('--support-clean',required=True)
    p.add_argument('--gt',required=True); p.add_argument('--device',default='cpu')
    p.add_argument('--output-json',required=True); a=p.parse_args()
    sys.path.insert(0,str(Path(a.source_root).resolve()))
    from net.model import AWRaCLe as OriginalModel
    torch.manual_seed(23); d=torch.device(a.device)
    new=load(NewModel,a.checkpoint,d); original=load(OriginalModel,a.checkpoint,d)
    x=image_tensor(a.query).to(d); sd=context_tensor(a.support_degraded).to(d); sc=context_tensor(a.support_clean).to(d)
    with torch.inference_mode(): yn=new(x,[sd,sc]); yo=original(x,[sd,sc])
    delta=(yn-yo).abs(); gt=image_tensor(a.gt)[:,:,:yn.shape[-2],:yn.shape[-1]][0].permute(1,2,0).numpy()
    metrics={}
    for name,out in [('original',yo),('new',yn)]:
        arr=out[0].cpu().permute(1,2,0).numpy().clip(0,1)
        metrics[name]={'psnr':float(peak_signal_noise_ratio(gt,arr,data_range=1.0)),
                       'ssim':float(structural_similarity(gt,arr,data_range=1.0,channel_axis=-1))}
    result={'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),
            'original_metrics':metrics['original'],'new_metrics':metrics['new'],
            'psnr_abs_difference':abs(metrics['original']['psnr']-metrics['new']['psnr']),
            'ssim_abs_difference':abs(metrics['original']['ssim']-metrics['new']['ssim']),
            'query_shape':list(x.shape),'support_shapes':[list(sd.shape),list(sc.shape)]}
    Path(a.output_json).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output_json).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
