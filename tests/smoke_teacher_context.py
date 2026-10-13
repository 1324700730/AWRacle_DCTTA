"""Check AWRaCLe's copied CLIP hook writes Teacher context into the Teacher."""
import json,copy,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.infer import image_tensor,context_tensor,load_model
from adaptation.dir import rebind_context_encoder_hooks,DegradationConsistentAdapter

def main():
 d=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 net=load_model(ROOT/'assets/checkpoints/awracle_best.ckpt',d).eval()
 teacher=copy.deepcopy(net);rebind_context_encoder_hooks(teacher);teacher.eval()
 x=image_tensor(ROOT/'data/demo/smoke/q.png').to(d)
 s=[context_tensor(ROOT/'data/demo/support/degraded.jpg').to(d),context_tensor(ROOT/'data/demo/support/clean.jpg').to(d)]
 with torch.no_grad(): y=teacher(x,s)
 key=teacher.target_layers[0]; z=teacher.intermediate_features.get(key)
 adapter=DegradationConsistentAdapter(net,teacher,None,None,None,None,None)
 with torch.no_grad(): pseudo,confidence=adapter._teacher_pseudo(x,s)
 ok=z is not None and y.shape==x.shape and pseudo.shape==x.shape and torch.isfinite(confidence).all()
 result={'status':'PASS' if ok else 'FAIL','query_shape':list(x.shape),'output_shape':list(y.shape),'teacher_context_shape':list(z.shape) if z is not None else None,'teacher_context_populated':z is not None,'student_context_untouched':key not in net.intermediate_features,'dir_pseudo_shape':list(pseudo.shape),'confidence_finite':bool(torch.isfinite(confidence).all()),'confidence_mean':float(confidence.mean()),
         'support_protocol':'seen_pool_component_smoke','support_degraded':'data/demo/support/degraded.jpg',
         'support_clean':'data/demo/support/clean.jpg',
         'note':'Mechanics-only component check; not a formal Seen or Unseen benchmark result.'}
 (ROOT/'experiments/dctta_component_smoke/teacher_context.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
