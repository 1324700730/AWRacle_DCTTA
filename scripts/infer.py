"""Minimal AWRaCLe-CLIP runner. Images are RGB [0,1]; GT is never used by adaptation."""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.awracle.model import AWRaCLe

CLIP_NORM = Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))
def image_tensor(path):
    im=Image.open(path).convert('RGB')
    w,h=im.size; w-=w%16; h-=h%16
    return ToTensor()(im.crop((0,0,w,h))).unsqueeze(0)
def context_tensor(path):
    # Match AWRaCLe/utils/dataset_utils.py::TestDataset_IC.context_transform:
    # torchvision BICUBIC resize, center crop, ToTensor, then CLIP normalization.
    return CLIP_NORM(ToTensor()(CenterCrop(224)(Resize(224,interpolation=InterpolationMode.BICUBIC)(Image.open(path).convert('RGB'))))).unsqueeze(0)
def load_model(path, device):
    m=AWRaCLe(decoder=True)
    ck=torch.load(path,map_location='cpu',weights_only=False)
    sd=ck.get('state_dict',ck)
    sd={k.removeprefix('net.'):v for k,v in sd.items() if k.startswith('net.') or not 'state_dict' in ck}
    missing,unexpected=m.load_state_dict(sd,strict=False)
    if len(missing)>0 or len(unexpected)>0:
        raise RuntimeError(f'Checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected')
    return m.to(device)
def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=None,help='Optional YAML config; explicit CLI values take precedence')
    p.add_argument('--mode',choices=['baseline'],default=None)
    p.add_argument('--checkpoint',default=None); p.add_argument('--input-dir',default=None)
    p.add_argument('--support-degraded',default=None); p.add_argument('--support-clean',default=None)
    p.add_argument('--output-dir',default=None); p.add_argument('--device',default=None)
    p.add_argument('--operator',default=None,help='Optional operator/dataset label used in the run summary')
    p.add_argument('--max-samples',type=int,default=None); p.add_argument('--seed',type=int,default=None)
    p.add_argument('--gt-dir',default=None,help='Optional evaluation-only GT directory; never used for adaptation')
    a=p.parse_args()
    values={}
    if a.config:
        import yaml
        values=yaml.safe_load(Path(a.config).read_text()) or {}
    keymap={'input-dir':'input_dir','support-degraded':'support_degraded','support-clean':'support_clean','output-dir':'output_dir','max-samples':'max_samples','gt-dir':'gt_dir'}
    for action in p._actions:
        dest=action.dest
        if not hasattr(a,dest):
            continue
        cfg=keymap.get(dest,dest)
        if getattr(a,dest) is None and cfg in values:
            setattr(a,dest,values[cfg])
    a.mode=a.mode or 'baseline'; a.device=a.device or 'cuda:0'
    a.max_samples=0 if a.max_samples is None else a.max_samples
    a.seed=23 if a.seed is None else a.seed
    missing=[k for k in ('checkpoint','input_dir','support_degraded','support_clean','output_dir') if not getattr(a,k)]
    if missing: p.error('missing required values (CLI or config): '+', '.join(missing))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    wall_start=time.perf_counter()
    dev=torch.device(a.device)
    if dev.type=='cuda':
        torch.empty((),device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
    net=load_model(a.checkpoint,dev); net.eval()
    paths=sorted(x for x in Path(a.input_dir).iterdir() if x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif'})
    if a.max_samples: paths=paths[:a.max_samples]
    sd=context_tensor(a.support_degraded).to(dev); sc=context_tensor(a.support_clean).to(dev)
    outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in paths:
        sample_start=time.perf_counter()
        x=image_tensor(path).to(dev)
        net.eval()
        with torch.inference_mode(): y=net(x,[sd,sc]).clamp(0,1)
        adaptation={}
        arr=(y[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        Image.fromarray(arr).save(outdir/(path.stem+'.png'))
        if a.gt_dir:
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            gt=image_tensor(Path(a.gt_dir)/(path.stem+'.png'))
            gt=gt[:,:,:y.shape[-2],:y.shape[-1]]
            pr=y.detach().cpu()[0].permute(1,2,0).numpy()
            ref=gt[0].permute(1,2,0).numpy()
            adaptation.update(psnr=float(peak_signal_noise_ratio(ref,pr,data_range=1.0)),
                              ssim=float(structural_similarity(ref,pr,data_range=1.0,channel_axis=-1)))
        rows.append({'sample':path.name,'shape':list(y.shape),'mode':a.mode,
                     'evaluation_seconds':time.perf_counter()-sample_start,**adaptation})
    (outdir/'manifest.json').write_text(json.dumps(rows,indent=2))
    metric_rows=[r for r in rows if 'psnr' in r and 'ssim' in r]
    summary={'mode':'baseline','operator':a.operator or Path(a.input_dir).name,
             'sample_count':len(rows),'evaluated_sample_count':len(metric_rows),
             'mean_psnr':float(np.mean([r['psnr'] for r in metric_rows])) if metric_rows else None,
             'mean_ssim':float(np.mean([r['ssim'] for r in metric_rows])) if metric_rows else None,
             'total_evaluation_seconds':float(sum(r['evaluation_seconds'] for r in rows)),
             'total_wall_seconds':time.perf_counter()-wall_start,
             'peak_cuda_memory_bytes':int(torch.cuda.max_memory_allocated(dev)) if dev.type=='cuda' else 0,
             'seed':a.seed,'checkpoint':str(Path(a.checkpoint).resolve()),
             'checkpoint_sha256':file_sha256(a.checkpoint),
             'config_file':str(Path(a.config).resolve()) if a.config else None,
             'config_sha256':file_sha256(a.config) if a.config else None,
             'awracle_source_commit':'e43160b6aed6b7495761243fc1fbf198978db594',
             'input_dir':str(Path(a.input_dir).resolve()),
             'support_degraded':str(Path(a.support_degraded).resolve()),
             'support_clean':str(Path(a.support_clean).resolve()),
             'gt_dir':str(Path(a.gt_dir).resolve()) if a.gt_dir else None,
             'device':str(dev),'max_samples_cap':a.max_samples,'samples':rows}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__': main()
