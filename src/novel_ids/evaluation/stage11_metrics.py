"""Generic Stage 11/12 classification and OOD metrics."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score

def _auc(y,s):
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None
def _ap(y,s):
    return float(average_precision_score(y,s)) if np.sum(y)>0 else None
def _fpr95(y,s):
    pos=int(np.sum(y==1)); neg=int(np.sum(y==0))
    if not pos or not neg: return None
    for threshold in np.unique(s)[::-1]:
        pred=s>=threshold
        if np.sum(pred&(y==1))/pos>=0.95: return float(np.sum(pred&(y==0))/neg)
    return 1.0

def classification_metrics(y_true,y_pred,class_names,*,probabilities=None,uncertain_mask=None,ood_true=None,ood_scores=None):
    """Macro aggregates include only supported classes; OVR AUC/AP where defined.

    OOD positive label is 1 (unfamiliar), with larger scores meaning more OOD.
    Ground-truth labels are used only by this evaluation function, never as model inputs.
    """
    yt=np.asarray(y_true,dtype=int); yp=np.asarray(y_pred,dtype=int); names=tuple(class_names); n=len(names)
    if n<1 or len(set(names))!=n or yt.ndim!=1 or yp.shape!=yt.shape or yt.size==0: raise ValueError("Need non-empty aligned labels and unique class_names")
    if np.any((yt<0)|(yt>=n)|(yp<0)|(yp>=n)): raise ValueError("Label index outside class_names")
    cm=confusion_matrix(yt,yp,labels=np.arange(n)); support=cm.sum(1)
    norm=np.divide(cm,support[:,None],out=np.zeros_like(cm,dtype=float),where=support[:,None]!=0)
    per={}; fprs=[]; fnrs=[]
    for i,name in enumerate(names):
        tp=int(cm[i,i]); fn=int(support[i]-tp); fp=int(cm[:,i].sum()-tp); tn=int(cm.sum()-tp-fn-fp)
        precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        fpr=fp/(fp+tn) if fp+tn else 0.0; fnr=fn/(fn+tp) if fn+tp else 0.0
        per[name]={"precision":precision,"recall":recall,"f1":f1,"support":int(support[i]),"fpr":fpr,"fnr":fnr}
        if support[i]>0: fprs.append(fpr); fnrs.append(fnr)
    present=[per[name] for i,name in enumerate(names) if support[i]>0]
    macro_p=float(np.mean([x["precision"] for x in present])); macro_r=float(np.mean([x["recall"] for x in present])); macro_f=float(np.mean([x["f1"] for x in present]))
    weighted_f=float(sum(per[name]["f1"]*per[name]["support"] for name in names)/yt.size)
    result={"confusion_matrix":cm.tolist(),"normalized_confusion_matrix":norm.tolist(),"accuracy":float(accuracy_score(yt,yp)),
            "precision":macro_p,"recall":macro_r,"f1":macro_f,"macro_f1":macro_f,"weighted_f1":weighted_f,
            "fpr":float(np.mean(fprs)),"fnr":float(np.mean(fnrs)),"detection_rate":macro_r,"per_class":per}
    if probabilities is not None:
        probs=np.asarray(probabilities,dtype=float)
        if probs.shape!=(yt.size,n) or not np.isfinite(probs).all() or np.any(probs<0): raise ValueError("probabilities must be finite non-negative [N,C]")
        aucs=[]; aps=[]; per_ovr={}
        for i,name in enumerate(names):
            binary=(yt==i).astype(int); a=_auc(binary,probs[:,i]); ap=_ap(binary,probs[:,i])
            per_ovr[name]={"roc_auc":a,"pr_auc":ap}
            if a is not None: aucs.append(a)
            if ap is not None: aps.append(ap)
        result["one_vs_rest"]=per_ovr; result["roc_auc_ovr_macro"]=float(np.mean(aucs)) if aucs else None
        result["pr_auc_ovr_macro"]=float(np.mean(aps)) if aps else None
    if uncertain_mask is not None:
        uncertain=np.asarray(uncertain_mask,dtype=bool)
        if uncertain.shape!=yt.shape: raise ValueError("uncertain_mask must have shape [N]")
        accepted=~uncertain; result["uncertain_rate"]=float(np.mean(uncertain))
        result["accuracy_on_accepted"]=float(np.mean(yt[accepted]==yp[accepted])) if accepted.any() else None
    if ood_true is not None or ood_scores is not None:
        if ood_true is None or ood_scores is None: raise ValueError("ood_true and ood_scores must be supplied together")
        oy=np.asarray(ood_true,dtype=int); oscore=np.asarray(ood_scores,dtype=float)
        if oy.shape!=yt.shape or oscore.shape!=yt.shape or not np.isfinite(oscore).all() or np.any((oy!=0)&(oy!=1)): raise ValueError("OOD inputs must be aligned finite scores and binary labels")
        result["ood"]={"auroc":_auc(oy,oscore),"aupr":_ap(oy,oscore),"fpr_at_95_tpr":_fpr95(oy,oscore)}
    return result
