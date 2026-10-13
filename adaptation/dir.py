"""Degradation-Consistent DIR objective and teacher-student update."""
import torch
from torchvision.transforms import RandomHorizontalFlip,RandomRotation
from adaptation.losses import dctta_consistency_loss

def rebind_context_encoder_hooks(model):
    """Deepcopy preserves hook functions and their original `self` closure.

    Re-register AWRaCLe's CLIP intermediate hook on the clone so its forward
    populates the cloned model's own intermediate_features dictionary.
    """
    handles=[]
    for layer_name in model.target_layers:
        layer=model.clip
        for part in layer_name.split('.'):
            layer=getattr(layer,part)
        layer._forward_hooks.clear()
        handles.append(layer.register_forward_hook(
            lambda module,inputs,output,name=layer_name:model.hook_fn(module,inputs,output,name)))
    model._tta_hook_handles=handles
    return model

class DegradationConsistentAdapter:
    def __init__(self,student,teacher,optimizer,tips,trg,wavelet,perceptual,
                 teacher_weight=5.0,ema_decay=0.45):
        self.student=student; self.teacher=teacher; self.optimizer=optimizer; self.tips=tips
        self.trg=trg; self.wavelet=wavelet; self.perceptual=perceptual
        self.teacher_weight=float(teacher_weight); self.ema_decay=float(ema_decay)
        self.flip=RandomHorizontalFlip(p=1.0)
        self.rot={90:RandomRotation((90,90)),180:RandomRotation((180,180)),270:RandomRotation((270,270)),
                  -90:RandomRotation((-90,-90)),-180:RandomRotation((-180,-180)),-270:RandomRotation((-270,-270))}
    def _teacher_pseudo(self,x,support):
        # Exact 8-view pattern in DCTTA/tta.py::test_time_adaptation (airnet==0).
        F=self.flip; R=self.rot; t=self.teacher
        def fwd(z): return t(z,support)
        views=[F(fwd(F(x))), F(R[-90](fwd(R[90](F(x))))),
               F(R[-180](fwd(R[180](F(x))))), F(R[-270](fwd(R[270](F(x))))),
               fwd(x), R[-90](fwd(R[90](x))),R[-180](fwd(R[180](x))),R[-270](fwd(R[270](x)))]
        avg=torch.stack(views).mean(0)
        # Official code repeats this deterministic ensemble four times then
        # computes variance over repeats; its AWRaCLe/PromptIR have no dropout,
        # so repetition preserves the algorithm's zero-variance confidence map.
        repeated=torch.stack([avg,avg,avg,avg])
        var=torch.var(repeated,unbiased=False,dim=0)
        confidence=1.5-torch.sigmoid(var/0.0004)
        return avg.detach(),confidence.detach()
    def _ema(self):
        teacher=dict(self.teacher.named_parameters()); student=dict(self.student.named_parameters())
        delta_sq=0.0
        with torch.no_grad():
            for name,tp in teacher.items():
                delta=(student[name]-tp)*(1-self.ema_decay)
                delta_sq=delta_sq+delta.square().sum()
                tp.mul_(self.ema_decay).add_(student[name],alpha=1-self.ema_decay)
        return float(delta_sq.sqrt())
    def adapt(self,x,support,iterations=1):
        records=[]; self.student.train(); self.teacher.train()
        for _ in range(iterations):
            with torch.no_grad(): pseudo,confidence=self._teacher_pseudo(x,support)
            second_lq,trg_stats=self.trg(x,pseudo)
            with torch.no_grad(): anchor=self.student(x,support)
            pred=self.student(second_lq,support)
            pred_w=self.wavelet(pred.permute(0,2,3,1))[:,0,...]
            anchor_w=self.wavelet(anchor.permute(0,2,3,1))[:,0,...]
            pseudo_w=self.wavelet(pseudo.permute(0,2,3,1))[:,0,...]
            loss_s=dctta_consistency_loss(pred,anchor,confidence,pred_w,anchor_w,self.perceptual)
            loss_t=dctta_consistency_loss(pred,pseudo,confidence,pred_w,pseudo_w,self.perceptual)
            loss=loss_s+self.teacher_weight*loss_t
            if not torch.isfinite(loss): raise FloatingPointError('DIR loss is NaN/Inf')
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            grad_terms=[p.grad.detach().square().sum() for p in self.student.parameters() if p.grad is not None]
            grad_norm=float(torch.stack(grad_terms).sum().sqrt()) if grad_terms else 0.0
            if not grad_norm>0: raise RuntimeError('Student received no nonzero gradient')
            before=[p.detach().clone() for group in self.optimizer.param_groups for p in group['params']]
            self.optimizer.step()
            if self.tips is not None: self.tips.protect()
            updated=sum((p.detach()!=old).sum() for old,p in zip(before,[p for group in self.optimizer.param_groups for p in group['params']]))
            updated_elements=int(updated)
            teacher_ema_delta=self._ema()
            if self.tips is not None and updated_elements<=0:
                raise RuntimeError('Student has gradients but selected TIPS mask prevented every parameter update')
            records.append({'loss':float(loss.detach()),'loss_student':float(loss_s.detach()),
                            'loss_teacher':float(loss_t.detach()),'gradient_l2':grad_norm,
                            'updated_elements':updated_elements,'teacher_ema_delta_l2':teacher_ema_delta,
                            'trg':trg_stats,'confidence_mean':float(confidence.mean())})
        self.student.eval()
        with torch.no_grad(): restored=self.student(x,support)
        return restored,records
