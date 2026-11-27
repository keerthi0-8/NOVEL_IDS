"""Global temperature scaling fitted from validation logits and labels only."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

def expected_calibration_error(logits:torch.Tensor,labels:torch.Tensor,n_bins:int=15)->float:
    if logits.ndim!=2 or labels.shape!=(logits.shape[0],) or n_bins<1: raise ValueError("Expected logits [N,C], labels [N], n_bins >=1")
    if not torch.isfinite(logits).all(): raise ValueError("logits contain NaN or Inf")
    probs=torch.softmax(logits,dim=1); confidence,pred=probs.max(dim=1); correct=pred.eq(labels.long()).float()
    edges=torch.linspace(0,1,n_bins+1,device=logits.device); ece=logits.new_zeros(())
    for i in range(n_bins):
        mask=(confidence>edges[i])&(confidence<=edges[i+1])
        if mask.any(): ece += mask.float().mean()*(confidence[mask].mean()-correct[mask].mean()).abs()
    return float(ece.item())

class TemperatureScaler:
    """Single global scalar; no training labels are accepted by this API."""
    def __init__(self): self.temperature:float|None=None
    def fit(self,validation_logits:torch.Tensor,validation_labels:torch.Tensor,max_iter:int=100):
        logits=validation_logits.detach().float(); labels=validation_labels.detach().long()
        if logits.ndim!=2 or labels.shape!=(logits.shape[0],) or logits.shape[0]==0: raise ValueError("Invalid validation logits/labels shape")
        if not torch.isfinite(logits).all() or labels.min()<0 or labels.max()>=logits.shape[1]: raise ValueError("Invalid validation logits/labels")
        log_t=nn.Parameter(torch.zeros((),dtype=logits.dtype,device=logits.device))
        optimizer=torch.optim.LBFGS([log_t],lr=0.1,max_iter=max_iter,line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad(); loss=F.cross_entropy(logits/log_t.exp().clamp(0.05,20.0),labels); loss.backward(); return loss
        optimizer.step(closure); self.temperature=float(log_t.detach().exp().clamp(0.05,20.0).item()); return self
    def transform_logits(self,logits:torch.Tensor)->torch.Tensor:
        if self.temperature is None: raise RuntimeError("Fit temperature on validation data first")
        if not torch.isfinite(logits).all(): raise ValueError("logits contain NaN or Inf")
        return logits/self.temperature
    def probabilities(self,logits:torch.Tensor)->torch.Tensor: return torch.softmax(self.transform_logits(logits),dim=1)
