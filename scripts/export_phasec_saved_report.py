from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path.cwd(); RUN=Path(r"C:\NOVEL_IDS_runs\phaseC_recovery_20261126"); PRED=RUN/"predictions"
OUT=ROOT/"docs"; FIG=OUT/"figures"/"phaseC_saved_test"; FIG.mkdir(parents=True,exist_ok=True)
res=json.loads((RUN/"results.json").read_text(encoding="utf-8"))
classes=["Normal","DoS","Probe_Recon","UNSW_Exploits","UNSW_Fuzzers","UNSW_Generic","UNSW_Analysis","UNSW_Backdoor","UNSW_Shellcode","UNSW_Worms","NSL_R2L","NSL_U2R"]
nsl_only={"Normal","DoS","Probe_Recon","NSL_R2L","NSL_U2R"}; unsw_only=set(classes)-nsl_only
def from_cm(cm):
 cm=np.asarray(cm,dtype=np.int64); s=cm.sum(1); p=cm.sum(0); tp=np.diag(cm)
 pr=np.divide(tp,p,out=np.zeros(len(classes)),where=p>0); re=np.divide(tp,s,out=np.zeros(len(classes)),where=s>0)
 f=np.divide(2*pr*re,pr+re,out=np.zeros(len(classes)),where=(pr+re)>0); keep=s>0; total=int(s.sum())
 return {"accuracy":float(tp.sum()/max(1,total)),"macro_precision":float(pr[keep].mean()),"macro_recall":float(re[keep].mean()),"macro_f1":float(f[keep].mean()),"weighted_precision":float(np.average(pr,weights=s)),"weighted_recall":float(np.average(re,weights=s)),"weighted_f1":float(np.average(f,weights=s)),"total":total,"per_class":{c:{"precision":float(pr[i]),"recall":float(re[i]),"f1":float(f[i]),"support":int(s[i])} for i,c in enumerate(classes)}}
def save_cm(name,cm):
 pd.DataFrame(cm,index=classes,columns=classes).to_csv(OUT/f"confusion_{name}.csv",index_label="actual")
 sums=np.asarray(cm).sum(1,keepdims=True); norm=np.divide(cm,sums,out=np.zeros_like(cm,dtype=float),where=sums>0)
 pd.DataFrame(norm,index=classes,columns=classes).to_csv(OUT/f"confusion_{name}_row_normalized.csv",index_label="actual")
 for suffix,arr in (("raw",cm),("row_normalized",norm)):
  fig,ax=plt.subplots(figsize=(12,10)); im=ax.imshow(arr,aspect="auto")
  ax.set_xticks(range(12)); ax.set_xticklabels(classes,rotation=55,ha="right",fontsize=8)
  ax.set_yticks(range(12)); ax.set_yticklabels(classes,fontsize=8)
  ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name+" "+suffix)
  fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(FIG/f"{name}_{suffix}.png",dpi=150); plt.close(fig)
views=[]; md=["# NOVEL-IDS saved source-test evaluation report","","Built only from frozen Phase C saved prediction CSVs and results.json. No model inference was run; source test partitions remain closed.","","Confusion matrices and classification metrics use forced predicted_family classes, including rows selectively marked Uncertain or Unfamiliar. Selective decisions are shown separately.",""]
mats={}
for ds in ("nsl_kdd","unsw_nb15"):
 for split in ("test_independent","test_full"):
  m=res["datasets"][ds][split]["primary"]; df=pd.read_csv(PRED/f"{ds}_{split}_primary.csv"); cm=np.asarray(m["confusion_matrix"],dtype=np.int64)
  name=ds+"_"+split; mats[name]=cm; met=from_cm(cm); views.append({"view":name,**met,"decision_counts":m.get("decision_counts",{}),"thresholds":m.get("thresholds",{}),"ood":m.get("ood",{}),"ood_score_summary":m.get("ood_score_summary",{})}); save_cm(name,cm)
  md += [f"## {name}", "", f"Accuracy {met['accuracy']:.4%}; macro precision {met['macro_precision']:.4%}; macro recall {met['macro_recall']:.4%}; macro F1 {met['macro_f1']:.4%}.", f"Weighted precision {met['weighted_precision']:.4%}; weighted recall {met['weighted_recall']:.4%}; weighted F1 {met['weighted_f1']:.4%}; rows {met['total']:,}.", "Selective decisions: "+json.dumps(m.get("decision_counts",{}),sort_keys=True), "OOD: "+json.dumps(m.get("ood",{}),sort_keys=True), "", "| Class | Precision | Recall | F1 | Support |","|---|---:|---:|---:|---:|"]
  for c in classes:
   v=met["per_class"][c]; vals=["—","—","—"] if v["support"]==0 else [f"{v[k]:.2%}" for k in ("precision","recall","f1")]
   md.append(f"| {c} | {vals[0]} | {vals[1]} | {vals[2]} | {v['support']:,} |")
  md += ["",""]
for split in ("test_independent","test_full"):
 cm=mats["nsl_kdd_"+split]+mats["unsw_nb15_"+split]; name="pooled_"+split; met=from_cm(cm); mats[name]=cm; views.append({"view":name,**met}); save_cm(name,cm)
 md += [f"## {name}","",f"Accuracy {met['accuracy']:.4%}; macro precision {met['macro_precision']:.4%}; macro recall {met['macro_recall']:.4%}; macro F1 {met['macro_f1']:.4%}.",f"Weighted precision {met['weighted_precision']:.4%}; weighted recall {met['weighted_recall']:.4%}; weighted F1 {met['weighted_f1']:.4%}; rows {met['total']:,}.","","| Class | Precision | Recall | F1 | Support |","|---|---:|---:|---:|---:|"]
 for c in classes:
  v=met["per_class"][c]; vals=["—","—","—"] if v["support"]==0 else [f"{v[k]:.2%}" for k in ("precision","recall","f1")]
  md.append(f"| {c} | {vals[0]} | {vals[1]} | {vals[2]} | {v['support']:,} |")
 md += ["",""]
md += ["## Cross-dataset confusion",""]
for ds in ("nsl_kdd","unsw_nb15"):
 allowed=nsl_only if ds=="nsl_kdd" else unsw_only
 for split in ("test_independent","test_full"):
  df=pd.read_csv(PRED/f"{ds}_{split}_primary.csv"); pred=df["predicted_family"].astype(str); foreign=~pred.isin(allowed)
  md.append(f"- {ds} {split}: {int(foreign.sum()):,}/{len(df):,} ({foreign.mean():.3%}) predicted into the other dataset's exclusive label space.")
md += ["","## Weak-class predicted labels",""]
weak=["UNSW_Analysis","UNSW_Backdoor","UNSW_Worms","UNSW_Fuzzers","UNSW_Shellcode","NSL_R2L","NSL_U2R"]
for ds in ("nsl_kdd","unsw_nb15"):
 for split in ("test_independent","test_full"):
  df=pd.read_csv(PRED/f"{ds}_{split}_primary.csv"); md.append(f"### {ds} {split}")
  for c in weak:
   sub=df[df.true_family.astype(str)==c]
   if len(sub): md.append(f"- {c} (n={len(sub)}): "+", ".join(f"{k}={v}" for k,v in sub.predicted_family.astype(str).value_counts().head(3).items()))
md += ["","## Decision statistics and coverage",""]
for ds in ("nsl_kdd","unsw_nb15"):
 for split in ("test_independent","test_full"):
  df=pd.read_csv(PRED/f"{ds}_{split}_primary.csv"); md.append(f"### {ds} {split}")
  md.append("Decision counts: "+json.dumps(df.decision.astype(str).value_counts().to_dict(),sort_keys=True))
  df=df.dropna(subset=["confidence","true_family","predicted_family"]).copy(); df["correct"]=df.true_family.astype(str)==df.predicted_family.astype(str); df=df.sort_values("confidence",ascending=False)
  md += ["| Coverage | Accuracy among highest-confidence rows | Rows |","|---:|---:|---:|"]
  for frac in (.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0):
   n=max(1,int(len(df)*frac)); md.append(f"| {frac:.0%} | {df.head(n).correct.mean():.3%} | {n:,} |")
  md.append("")
md += ["## How to read these numbers","","Accuracy is dominated by large classes such as Normal and UNSW_Generic. Macro metrics weigh supported classes equally and are pulled down by weak rare classes. Support is the true sample count, not an error measure. The two datasets have different label spaces; pooled values are aggregate summaries only. NSL-KDD test includes attack types absent from KDDTrain+ training. The saved prediction files do not retain fine-grained attack subtype seen/test-only metadata, so the requested fine-type recall split cannot be computed. Baseline per-row probability/OOD/decision files are absent; HGB/MLP aggregate results exist. Plain Transformer test results and a sixth ablation are unavailable because test is closed.",""]
(OUT/"EVALUATION_REPORT.md").write_text("\n".join(md),encoding="utf-8")
(OUT/"EVALUATION_REPORT_SAVED_METRICS.json").write_text(json.dumps({"source":str(RUN/"results.json"),"views":views},indent=2),encoding="utf-8")
print("REPORT",OUT/"EVALUATION_REPORT.md")
print("VIEWS",[(x["view"],x["total"],round(x["accuracy"],5),round(x["macro_f1"],5),round(x["weighted_f1"],5)) for x in views])
