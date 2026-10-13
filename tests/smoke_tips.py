"""Bounded query-only test of AWRaCLe Fisher-TIPS mask and parameter protection."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.infer import image_tensor,context_tensor,load_model
from adaptation.tips import FisherTIPS

def main():
 d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');cp=ROOT/'assets/checkpoints/awracle_best.ckpt'
 q=image_tensor(ROOT/'data/demo/smoke/q.png').to(d)
 support=[context_tensor(ROOT/'data/demo/support/degraded.jpg').to(d),context_tensor(ROOT/'data/demo/support/clean.jpg').to(d)]
 model=load_model(cp,d)
 names=[]
 for n,p in model.named_parameters():
  enabled=n.split('.')[0].startswith(('prompt','encoder','decoder'));p.requires_grad_(enabled)
  if enabled and n.rsplit('.',1)[-1] in ('weight','bias'):names.append(n)
 params=dict(model.named_parameters());tips=FisherTIPS(model,names,.6);model.train();stats=tips.fit([q],support)
 before={n:p.detach().clone() for n,p in params.items() if n in names}
 frozen_versions={n:p._version for n,p in params.items() if not p.requires_grad}
 opt=torch.optim.Adam([params[n] for n in names],lr=2e-4)
 # Use a flipped-view pseudo-restoration, inverse-warped back to the query;
 # this is a no-GT self-consistency signal and avoids a degenerate zero loss.
 with torch.no_grad():target=torch.flip(model(torch.flip(q,(3,)),support),(3,)).detach()
 pred=model(q,support);loss=torch.nn.functional.l1_loss(pred,target);opt.zero_grad(set_to_none=True);loss.backward();opt.step();tips.protect()
 selected_exact=True;free_changed=0;selected=0
 for n,mask in tips.mask.items():
  p=params[n];old=before[n];selected+=int(mask.sum());selected_exact &= bool(torch.equal(p[mask],old[mask]));free_changed+=int((p[~mask]!=old[~mask]).sum())
 frozen_exact=all(params[n]._version==v for n,v in frozen_versions.items())
 result={'status':'PASS' if selected_exact and free_changed>0 and frozen_exact else 'FAIL','query_shape':list(q.shape),
         'fisher_stats':stats,'adaptation_loss':float(loss.detach()),'selected_entries_exactly_restored':selected_exact,
         'unselected_entries_changed':free_changed,'frozen_parameter_versions_unchanged':frozen_exact,'gt_used':False,
         'support_protocol':'seen_pool_component_smoke','support_degraded':'data/demo/support/degraded.jpg',
         'support_clean':'data/demo/support/clean.jpg',
         'note':'Mechanics-only component check; not a formal Seen or Unseen benchmark result.'}
 (ROOT/'experiments/dctta_component_smoke').mkdir(parents=True,exist_ok=True)
 (ROOT/'experiments/dctta_component_smoke/tips.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
