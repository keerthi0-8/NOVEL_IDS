"""Training-fitted class-conditional Mahalanobis and energy OOD scores."""
from __future__ import annotations
from dataclasses import dataclass
import torch

def energy_score(logits:torch.Tensor,temperature:float=1.0)->torch.Tensor:
    if logits.ndim!=2 or not torch.isfinite(logits).all() or temperature<=0: raise ValueError("logits finite [B,C] and temperature >0 required")
    return -float(temperature)*torch.logsumexp(logits/float(temperature),dim=1)

@dataclass
class ClassConditionalMahalanobis:
    min_class_samples:int
    shrinkage:float
    ridge:float
    means:dict[int,torch.Tensor]|None=None
    precision:torch.Tensor|None=None
    excluded_classes:dict[int,int]|None=None
    included_counts:dict[int,int]|None=None
    def fit(self,training_embeddings:torch.Tensor,training_labels:torch.Tensor):
        x=training_embeddings.detach().float(); y=training_labels.detach().long()
        if x.ndim!=2 or x.shape[1]!=128 or y.shape!=(x.shape[0],): raise ValueError("Expected source train embeddings [N,128], labels [N]")
        if x.shape[0]<2 or not torch.isfinite(x).all(): raise ValueError("Training embeddings need >=2 finite rows")
        if self.min_class_samples<2 or not 0<=self.shrinkage<=1 or self.ridge<=0: raise ValueError("min_class_samples>=2, shrinkage in [0,1], ridge>0 required")
        means,excluded,included={}, {}, {}; residuals=[]
        for class_id in torch.unique(y).tolist():
            rows=x[y==class_id]; count=int(rows.shape[0])
            if count<self.min_class_samples: excluded[int(class_id)]=count; continue
            mean=rows.mean(0); means[int(class_id)]=mean; included[int(class_id)]=count; residuals.append(rows-mean)
        if not means: raise ValueError("No classes meet min_class_samples")
        within=torch.cat(residuals,0); denom=max(int(within.shape[0])-len(means),1)
        covariance=within.T@within/float(denom); diagonal=torch.diag(torch.diagonal(covariance))
        covariance=(1-self.shrinkage)*covariance+self.shrinkage*diagonal
        scale=torch.diagonal(covariance).mean().clamp_min(1e-8)
        covariance=covariance+(self.ridge*scale)*torch.eye(128,dtype=x.dtype,device=x.device)
        self.means,self.precision=means,torch.cholesky_inverse(torch.linalg.cholesky(covariance))
        self.excluded_classes,self.included_counts=excluded,included; return self
    def score(self,embeddings:torch.Tensor)->torch.Tensor:
        if self.means is None or self.precision is None: raise RuntimeError("Fit on source training data first")
        x=embeddings.float()
        if x.ndim!=2 or x.shape[1]!=128 or not torch.isfinite(x).all(): raise ValueError("embeddings must be finite [B,128]")
        distances=[]
        for mean in self.means.values():
            delta=x-mean; distances.append(torch.einsum("bi,ij,bj->b",delta,self.precision,delta).clamp_min(0))
        return torch.stack(distances,dim=1).min(dim=1).values

@dataclass
class OODScoreCombiner:
    """Score z-normalization fitted only from source training scores."""
    weights:tuple[float,float]
    mean_mahalanobis:float|None=None
    std_mahalanobis:float|None=None
    mean_energy:float|None=None
    std_energy:float|None=None
    def fit(self,training_mahalanobis:torch.Tensor,training_energy:torch.Tensor):
        m,e=training_mahalanobis.detach().float(),training_energy.detach().float()
        if m.ndim!=1 or e.shape!=m.shape or m.numel()<2 or not torch.isfinite(m).all() or not torch.isfinite(e).all(): raise ValueError("Need aligned finite source-training score vectors")
        if len(self.weights)!=2 or any(w<0 for w in self.weights) or sum(self.weights)<=0: raise ValueError("Weights must be non-negative and sum >0")
        self.mean_mahalanobis=float(m.mean()); self.std_mahalanobis=float(m.std(unbiased=False).clamp_min(1e-8))
        self.mean_energy=float(e.mean()); self.std_energy=float(e.std(unbiased=False).clamp_min(1e-8)); return self
    def score(self,mahalanobis:torch.Tensor,energy:torch.Tensor)->torch.Tensor:
        if self.mean_mahalanobis is None: raise RuntimeError("Fit combiner on source training scores first")
        m,e=mahalanobis.float(),energy.float()
        if m.ndim!=1 or e.shape!=m.shape or not torch.isfinite(m).all() or not torch.isfinite(e).all(): raise ValueError("Inputs must be aligned finite vectors")
        wm,we=self.weights; total=wm+we
        return (wm*(m-self.mean_mahalanobis)/self.std_mahalanobis+we*(e-self.mean_energy)/self.std_energy)/total
