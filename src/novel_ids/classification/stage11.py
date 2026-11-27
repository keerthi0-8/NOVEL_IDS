"""Generic Stage 11 heads consuming only Stage 10 Z_t."""
from __future__ import annotations
from collections import Counter
from typing import Mapping,Sequence
import torch
from torch import nn
from torch.nn import functional as F
from novel_ids.taxonomy.stage11 import Taxonomy,TaxonomyNotApprovedError

def compute_class_weights(training_labels: Sequence[str],classes: Sequence[str],clip_ratio: float=15.0):
    if not 10.0<=float(clip_ratio)<=20.0: raise ValueError("clip_ratio must be between 10 and 20")
    if not classes or len(set(classes))!=len(classes): raise ValueError("classes must be non-empty and unique")
    counts=Counter(str(x) for x in training_labels); unknown=set(counts)-set(classes)
    if unknown: raise ValueError(f"Training labels outside taxonomy: {sorted(unknown)}")
    absent=[c for c in classes if counts[c]==0]
    if absent: raise ValueError(f"No training samples for classes: {absent}")
    w=torch.tensor([counts[c]**-0.5 for c in classes],dtype=torch.float64); w=w/w.mean()
    w=torch.clamp(w,min=w.max()/float(clip_ratio)); w=w/w.mean(); weights=w.float()
    if not torch.isfinite(weights).all(): raise ValueError("Non-finite class weights")
    return weights,{c:int(counts[c]) for c in classes}

class Stage11Classifier(nn.Module):
    def __init__(self,num_classes:int,input_dim:int=128,hidden_dim:int|None=None,label_smoothing:float=0.0,class_weights:torch.Tensor|None=None):
        super().__init__()
        if num_classes<1 or input_dim!=128: raise ValueError("num_classes positive and input_dim exactly 128 required")
        if not 0<=label_smoothing<1: raise ValueError("label_smoothing must be in [0,1)")
        self.num_classes,self.label_smoothing=num_classes,float(label_smoothing)
        self.network=nn.Linear(128,num_classes) if hidden_dim is None else nn.Sequential(nn.Linear(128,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,num_classes))
        if class_weights is not None:
            if class_weights.shape!=(num_classes,) or not torch.isfinite(class_weights).all(): raise ValueError("Invalid class_weights")
            self.register_buffer("class_weights",class_weights.detach().float().clone())
        else: self.register_buffer("class_weights",None)
    def forward(self,z_t:torch.Tensor)->torch.Tensor:
        if z_t.ndim!=2 or z_t.shape[1]!=128 or not torch.isfinite(z_t).all(): raise ValueError("z_t must be finite [B,128]")
        return self.network(z_t)
    def loss(self,logits:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
        if logits.ndim!=2 or logits.shape[1]!=self.num_classes or targets.shape!=(logits.shape[0],): raise ValueError("Invalid logits/target shape")
        return F.cross_entropy(logits,targets.long(),weight=self.class_weights,label_smoothing=self.label_smoothing)

class NativeAblationHeads(nn.Module):
    """Per-dataset native-label heads, available only as an ablation mode."""
    def __init__(self,native_classes:Mapping[str,Sequence[str]]):
        super().__init__(); self.class_names={k:tuple(v) for k,v in native_classes.items()}
        if not self.class_names or any(not v or len(set(v))!=len(v) for v in self.class_names.values()): raise ValueError("Native classes must be non-empty and unique")
        self.heads=nn.ModuleDict({k:nn.Linear(128,len(v)) for k,v in self.class_names.items()})
    def forward(self,z_t:torch.Tensor,dataset:str)->torch.Tensor:
        if dataset not in self.heads: raise KeyError(f"No native head for {dataset}")
        if z_t.ndim!=2 or z_t.shape[1]!=128 or not torch.isfinite(z_t).all(): raise ValueError("z_t must be finite [B,128]")
        return self.heads[dataset](z_t)

def build_classifier(mode:str,*,taxonomy:Taxonomy|None=None,native_classes=None,label_smoothing:float=0.0):
    if mode=="harmonized":
        if taxonomy is None or taxonomy.status!="approved": raise TaxonomyNotApprovedError("Harmonized head requires approved external taxonomy")
        return Stage11Classifier(len(taxonomy.classes),label_smoothing=label_smoothing)
    if mode=="native_separate_ablation":
        if native_classes is None: raise ValueError("native_classes required for ablation mode")
        return NativeAblationHeads(native_classes)
    raise ValueError("mode must be harmonized or native_separate_ablation")
