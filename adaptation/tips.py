"""DCTTA Fisher-based TIPS protection adapted to AWRaCLe Support-Query calls."""
import torch

def augment8(x):
    # Matches DCTTA/utils/utils_tta.py operator set ['', 'v','h','t','vh','vt','ht','vht'].
    ops=('', 'v', 'h', 't', 'vh', 'vt', 'ht', 'vht')
    out=[]
    for op in ops:
        y=x
        if 'v' in op: y=torch.flip(y,(3,))
        if 'h' in op: y=torch.flip(y,(2,))
        if 't' in op: y=y.transpose(2,3).contiguous()
        out.append((op,y))
    return out

def invert_aug(y,op):
    if 't' in op: y=y.transpose(2,3).contiguous()
    if 'v' in op: y=torch.flip(y,(3,))
    if 'h' in op: y=torch.flip(y,(2,))
    return y

class FisherTIPS:
    def __init__(self, model, optimizer_names, ratio=0.6):
        if not 0.0 <= ratio <= 1.0: raise ValueError('fisher_ratio must be in [0,1]')
        self.model=model; self.ratio=float(ratio); self.names=tuple(optimizer_names)
        self.initial={n:p.detach().clone() for n,p in model.named_parameters() if n in self.names}
        self.fisher={}; self.mask={}
    @torch.no_grad()
    def _pseudo(self,x,support):
        restored=[]
        for op,xx in augment8(x): restored.append(invert_aug(self.model(xx,support),op))
        return torch.stack(restored).mean(0)
    def fit(self, queries, support):
        # GT is intentionally absent from this API. Fisher is accumulated only
        # from query self-consistency, as in DCTTA/tta.py::compute_fisher.
        for x in queries:
            self.model.zero_grad(set_to_none=True)
            transformed=list(reversed(augment8(x)))
            restored=[]
            for i,(op,xx) in enumerate(transformed):
                if i < len(transformed)-1:
                    with torch.no_grad(): y=self.model(xx,support)
                else: y=self.model(xx,support)
                restored.append(invert_aug(y,op))
            pseudo=torch.stack([z.detach() for z in restored]).mean(0)
            loss=torch.sqrt((restored[-1]-pseudo).square()+1e-3).mean()
            loss.backward()
            for name,p in self.model.named_parameters():
                if name in self.initial and p.grad is not None:
                    value=p.grad.detach().square()
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f'Non-finite Fisher score for {name}')
                    self.fisher[name]=self.fisher.get(name,torch.zeros_like(value))+value
        for name,score in self.fisher.items():
            flat=score.flatten(); k=int(flat.numel()*self.ratio)
            mask=torch.zeros_like(flat,dtype=torch.bool)
            if k: mask[torch.topk(flat,k=k,sorted=False).indices]=True
            self.mask[name]=mask.view_as(score)
        self.model.zero_grad(set_to_none=True)
        return self.stats()
    @torch.no_grad()
    def protect(self):
        params=dict(self.model.named_parameters())
        for name,mask in self.mask.items(): params[name].copy_(torch.where(mask,self.initial[name],params[name]))
    def stats(self):
        n=sum(m.numel() for m in self.mask.values()); selected=sum(int(m.sum()) for m in self.mask.values())
        return {'selected_parameter_tensors':len(self.mask),'selected_elements':selected,'trainable_elements':n,
                'selected_ratio':selected/n if n else 0.0,'selected_names':list(self.mask)}
