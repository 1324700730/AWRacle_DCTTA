"""Bounded real-query test of the official five-update RDDM TRG port."""
import json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.infer import image_tensor,context_tensor,load_model
from adaptation.trg import OnlineResidualDiffusionTRG

def main():
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');torch.manual_seed(23)
 cp=ROOT/'assets/checkpoints/awracle_best.ckpt'
 q=ROOT/'data/demo/smoke/q.png'
 sd=context_tensor(ROOT/'data/demo/support/degraded.jpg').to(device)
 sc=context_tensor(ROOT/'data/demo/support/clean.jpg').to(device)
 net=load_model(cp,device).eval()
 with torch.no_grad(): target=net(image_tensor(q).to(device),[sd,sc])
 del net;torch.cuda.empty_cache() if device.type=='cuda' else None
 trg=OnlineResidualDiffusionTRG(device,image_size=256,train_steps=5)
 start=time.perf_counter();generated,stats=trg(image_tensor(q).to(device),target)
 result={'status':'PASS','query_shape':list(image_tensor(q).shape),'target_shape':list(target.shape),
         'generated_shape':list(generated.shape),'finite':bool(torch.isfinite(generated).all()),**stats,
         'elapsed_seconds':time.perf_counter()-start,
         'generated_min':float(generated.min()),'generated_max':float(generated.max()),
         'support_protocol':'seen_pool_component_smoke','support_degraded':'data/demo/support/degraded.jpg',
         'support_clean':'data/demo/support/clean.jpg',
         'note':'Mechanics-only component check; not a formal Seen or Unseen benchmark result.'}
 if device.type=='cuda':result['peak_cuda_memory_bytes']=torch.cuda.max_memory_allocated(device)
 (ROOT/'experiments/dctta_component_smoke').mkdir(parents=True,exist_ok=True)
 (ROOT/'experiments/dctta_component_smoke/trg.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
