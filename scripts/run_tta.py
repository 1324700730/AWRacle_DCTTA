"""DCTTA online adaptation for AWRaCLe-CLIP (one invocation = one target stream)."""
import argparse,copy,json,random,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.infer import image_tensor,context_tensor,load_model
from adaptation.trg import OnlineResidualDiffusionTRG
from adaptation.tips import FisherTIPS
from adaptation.dir import DegradationConsistentAdapter,rebind_context_encoder_hooks
from adaptation.losses import WaveletTransform,DCTTAPerceptualLoss

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--config',default=None,help='Optional YAML configuration')
 p.add_argument('--checkpoint');p.add_argument('--input-dir')
 p.add_argument('--support-degraded');p.add_argument('--support-clean')
 p.add_argument('--support-protocol',choices=['seen_pool','few_shot'])
 p.add_argument('--support-manifest',help='Required for default seen_pool protocol')
 p.add_argument('--output-dir');p.add_argument('--device');p.add_argument('--operator',help='Optional operator/dataset label used in the run summary')
 p.add_argument('--seed',type=int);p.add_argument('--max-samples',type=int)
 p.add_argument('--iterations',type=int);p.add_argument('--learning-rate',type=float)
 p.add_argument('--fisher-ratio',type=float);p.add_argument('--teacher-weight',type=float)
 p.add_argument('--vgg19-weights',default=None);p.add_argument('--gt-dir',default=None)
 p.add_argument('--disable-trg',action='store_true');p.add_argument('--disable-tips',action='store_true')
 p.add_argument('--trg-train-steps',type=int,help='Official DCTTA uses five online RDDM updates per query')
 first,_=p.parse_known_args()
 if first.config:
  import yaml
  p.set_defaults(**(yaml.safe_load(Path(first.config).read_text()) or {}))
 a=p.parse_args()
 defaults={'device':'cuda:0','seed':23,'max_samples':0,'iterations':1,'learning_rate':2e-4,'fisher_ratio':.6,'teacher_weight':5.,'trg_train_steps':5,'support_protocol':'seen_pool'}
 for k,v in defaults.items():
  if getattr(a,k) is None:setattr(a,k,v)
 missing=[k for k in ('checkpoint','input_dir','support_degraded','support_clean','output_dir') if not getattr(a,k)]
 if missing:p.error('missing required values (CLI or config): '+', '.join(missing))
 support_entry=None;support_manifest=None
 if a.support_protocol=='seen_pool':
  if not a.support_manifest:p.error('seen_pool protocol requires --support-manifest; use --support-protocol few_shot only for a separately labeled target-paired protocol')
  support_manifest=Path(a.support_manifest).resolve()
  manifest_data=json.loads(support_manifest.read_text())
  if manifest_data.get('protocol')!='seen_support_pool' or manifest_data.get('split')!='train':
   p.error('support manifest must declare protocol=seen_support_pool and split=train')
  sd_path=str(Path(a.support_degraded).resolve());sc_path=str(Path(a.support_clean).resolve())
  def manifest_image_path(value):
   candidate=Path(value)
   return candidate.resolve() if candidate.is_absolute() else (support_manifest.parent/candidate).resolve()
  for entry in manifest_data.get('pairs',[]):
   if str(manifest_image_path(entry.get('degraded','')))==sd_path and str(manifest_image_path(entry.get('clean','')))==sc_path:
    support_entry=entry;break
  if support_entry is None:p.error('selected Support pair is not listed in the Seen Support Pool manifest')
 if not 1<=a.iterations<=10: p.error('iterations must be in [1,10] for this bounded runner')
 if a.trg_train_steps!=5: p.error('Faithful DCTTA TRG uses exactly five inner updates')
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 wall_start=time.perf_counter()
 dev=torch.device(a.device); out=Path(a.output_dir)
 if dev.type=='cuda':
  torch.empty((),device=dev)
  torch.cuda.reset_peak_memory_stats(dev)
 vgg_path=Path(a.vgg19_weights) if a.vgg19_weights else ROOT/'models/dctta/assets/vgg19-dcbb9e9d.pth'
 if not vgg_path.is_file():
  raise FileNotFoundError(f'DCTTA DIR requires ImageNet VGG19 weights at {vgg_path}; pass --vgg19-weights. No random or reduced loss is substituted.')
 perceptual=DCTTAPerceptualLoss(vgg_path).to(dev)
 if out.exists() and any(out.iterdir()):
  raise FileExistsError(f'Output directory is not empty: {out}; choose a fresh experiment directory')
 out.mkdir(parents=True,exist_ok=True)
 paths=sorted(q for q in Path(a.input_dir).iterdir() if q.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif'})
 if a.max_samples: paths=paths[:a.max_samples]
 if not paths: raise ValueError('No query images found')
 query_paths={str(p.resolve()) for p in paths}
 if str(Path(a.support_degraded).resolve()) in query_paths or str(Path(a.support_clean).resolve()) in query_paths:
  raise ValueError('Support images must not be members of the Query directory')
 order_rng=torch.Generator(device='cpu').manual_seed(a.seed)
 def seeded_shuffle(items):
  permutation=torch.randperm(len(items),generator=order_rng).tolist()
  return [items[i] for i in permutation]
 fisher_paths=seeded_shuffle(paths)
 adapt_paths=seeded_shuffle(paths)
 support=[context_tensor(a.support_degraded),context_tensor(a.support_clean)]
 support=[z.to(dev) for z in support]
 net=load_model(a.checkpoint,dev)
 train_names=[]
 for name,param in net.named_parameters():
  top=name.split('.')[0]
  enabled=top.startswith(('prompt','encoder','decoder'))
  param.requires_grad_(enabled)
  if enabled and name.rsplit('.',1)[-1] in ('weight','bias'):train_names.append(name)
 params=[dict(net.named_parameters())[n] for n in train_names]
 all_named=dict(net.named_parameters())
 frozen_names=[n for n,pv in all_named.items() if not pv.requires_grad]
 if not params:raise RuntimeError('AWRaCLe DCTTA trainable parameter selection is empty')
 optimizer=torch.optim.Adam(params,lr=a.learning_rate,betas=(.9,.999))
 teacher=copy.deepcopy(net)
 rebind_context_encoder_hooks(teacher)
 for q in teacher.parameters():q.requires_grad_(False);q.detach_()
 # Match source construction order: one RDDM/Trainer per stream before
 # compute_fisher, then online generator and optimizer state persist in-stream.
 trg=None
 if not a.disable_trg:trg=OnlineResidualDiffusionTRG(dev,image_size=256,train_steps=5)
 tips=None;tips_stats=None;fisher_seconds=0.0
 if not a.disable_tips:
  tips=FisherTIPS(net,train_names,a.fisher_ratio)
  net.train()
  def fisher_queries():
   for path in fisher_paths:yield image_tensor(path).to(dev)
  fisher_start=time.perf_counter()
  tips_stats=tips.fit(fisher_queries(),support)
  fisher_seconds=time.perf_counter()-fisher_start
 wavelet=WaveletTransform(ROOT/'models/dctta/assets/wavelet.mat').to(dev)
 adapter=DegradationConsistentAdapter(net,teacher,optimizer,tips,trg,wavelet,perceptual,a.teacher_weight,.45)
 adaptation_records=[]
 for i,path in enumerate(adapt_paths):
  t=time.perf_counter();x=image_tensor(path).to(dev)
  if trg is None:
   class IdentityTRG:
    def __call__(self,query,pseudo):return query,{'trg_disabled':True}
   adapter.trg=IdentityTRG()
  _,steps=adapter.adapt(x,support,a.iterations)
  adaptation_records.append({'sample':path.name,'adaptation':steps,'adaptation_seconds':time.perf_counter()-t})
 # Match DCTTA/train_test_promptir.py: finish the unlabeled adaptation stream
 # first, then evaluate the final adapted student on the whole Query set.
 net.eval();records=[];adaptation_by_name={r['sample']:r for r in adaptation_records}
 for i,path in enumerate(paths):
  t=time.perf_counter();x=image_tensor(path).to(dev)
  with torch.no_grad(): restored=net(x,support).clamp(0,1)
  arr=(restored[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
  Image.fromarray(arr).save(out/(path.stem+'.png'))
  adaptation_record=adaptation_by_name[path.name]
  row={'sample':path.name,'shape':list(restored.shape),'adaptation':adaptation_record['adaptation'],
       'adaptation_seconds':adaptation_record['adaptation_seconds'],'evaluation_seconds':time.perf_counter()-t}
  if a.gt_dir:
   from skimage.metrics import peak_signal_noise_ratio,structural_similarity
   gt=image_tensor(Path(a.gt_dir)/(path.stem+'.png'))[:,:,:restored.shape[-2],:restored.shape[-1]][0].permute(1,2,0).numpy()
   pr=restored[0].permute(1,2,0).cpu().numpy()
   row.update(psnr=float(peak_signal_noise_ratio(gt,pr,data_range=1.)),ssim=float(structural_similarity(gt,pr,data_range=1.,channel_axis=-1)))
  records.append(row)
 peak=int(torch.cuda.max_memory_allocated(dev)) if dev.type=='cuda' else 0
 metric_records=[r for r in records if 'psnr' in r and 'ssim' in r]
 operator=a.operator or Path(a.input_dir).name
 summary={'mode':'dctta','operator':operator,'sample_count':len(records),
          'evaluated_sample_count':len(metric_records),
          'mean_psnr':float(np.mean([r['psnr'] for r in metric_records])) if metric_records else None,
          'mean_ssim':float(np.mean([r['ssim'] for r in metric_records])) if metric_records else None,
          'total_adaptation_seconds':float(sum(r['adaptation_seconds'] for r in adaptation_records)),
          'total_evaluation_seconds':float(sum(r['evaluation_seconds'] for r in records)),
          'fisher_seconds':fisher_seconds,'total_wall_seconds':time.perf_counter()-wall_start,
          'peak_cuda_memory_bytes':peak}
 meta={'mode':'dctta','protocol':'GT-free Fisher over shuffled Query pool -> persistent online adaptation stream -> final-model evaluation over sorted Query pool; reset by a fresh run/checkpoint per target-domain experiment',
       'operator':operator,'summary':summary,
       'seed':a.seed,'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':sha(a.checkpoint),
       'awracle_source_commit':'e43160b6aed6b7495761243fc1fbf198978db594',
       'dctta_source_commit':'0526bf7b87c2a54574ae1cb3af916fd5568fc0b1',
       'dctta_source_worktree_was_dirty_at_audit':True,
       'config_file':str(Path(a.config).resolve()) if a.config else None,
       'config_sha256':sha(a.config) if a.config else None,
       'device':str(dev),'max_samples_cap':a.max_samples,'vgg19_weights':str(vgg_path.resolve()),
       'vgg19_weights_sha256':sha(vgg_path),'gt_dir':str(Path(a.gt_dir).resolve()) if a.gt_dir else None,
       'trg_disabled':a.disable_trg,'tips_disabled':a.disable_tips,
       'input_dir':str(Path(a.input_dir).resolve()),'support_degraded':str(Path(a.support_degraded).resolve()),
       'support_clean':str(Path(a.support_clean).resolve()),'query_count':len(paths),
       'support_protocol':a.support_protocol,'support_manifest':str(support_manifest) if support_manifest else None,
       'support_manifest_sha256':sha(support_manifest) if support_manifest else None,'support_pool_entry':support_entry,
       'support_degraded_sha256':sha(a.support_degraded),'support_clean_sha256':sha(a.support_clean),
       'fisher_order':[p.name for p in fisher_paths],'adaptation_order':[p.name for p in adapt_paths],
       'fisher_ratio':a.fisher_ratio,
       'teacher_weight':a.teacher_weight,'learning_rate':a.learning_rate,'iterations':a.iterations,
       'trg':'official RDDM online, five RAdam updates/query, five-step DDIM sampling' if trg else 'ablation disabled',
       'tips':'Fisher top-k protection' if tips else 'ablation disabled','trainable_parameter_tensors':len(train_names),
       'trainable_elements':sum(p.numel() for p in params),'frozen_parameter_tensors':len(frozen_names),
       'frozen_elements':sum(all_named[n].numel() for n in frozen_names),'frozen_parameter_names':frozen_names,
       'parameter_selection':tips_stats,
       'selected_parameter_names':tips_stats['selected_names'] if tips_stats else [],'peak_cuda_memory_bytes':peak,
       'gt_used_for_adaptation':False,'samples':records}
 (out/'metrics.json').write_text(json.dumps(meta,indent=2)+'\n')
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'query_count':len(records),'output_dir':str(out),
                   'tips_selected_elements':tips_stats['selected_elements'] if tips_stats else None,
                   'summary':summary},indent=2))
if __name__=='__main__':main()
