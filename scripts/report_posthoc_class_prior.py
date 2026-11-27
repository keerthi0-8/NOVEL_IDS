from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path.cwd(); RUN=Path(r"C:\NOVEL_IDS_runs\phaseC_recovery_20261126")
POST=RUN/"posthoc_class_prior_adjustment"
classes=["Normal","DoS","Probe_Recon","UNSW_Exploits","UNSW_Fuzzers","UNSW_Generic","UNSW_Analysis","UNSW_Backdoor","UNSW_Shellcode","UNSW_Worms","NSL_R2L","NSL_U2R"]
def metric(y,p):
 y=np.asarray(y,dtype=str); p=np.asarray(p,dtype=str); s=[]; f=[]
 for c in classes:
  support=int(np.sum(y==c))
  if not support: continue
  tp=int(np.sum((y==c)&(p==c))); fp=int(np.sum((y!=c)&(p==c))); fn=support-tp
  pr=tp/(tp+fp) if tp+fp else 0.0; re=tp/support; f1=2*pr*re/(pr+re) if pr+re else 0.0
  s.append(support); f.append(f1)
 return {"accuracy":float(np.mean(y==p)),"macro_f1":float(np.mean(f)) if f else 0.0,"weighted_f1":float(np.average(f,weights=s)) if f else 0.0,"rows":len(y)}
out={}
for ds in ("nsl_kdd","unsw_nb15"):
 for split in ("test_independent","test_full"):
  original=pd.read_csv(RUN/"predictions"/f"{ds}_{split}_primary.csv")
  adjusted=pd.read_csv(POST/f"{ds}_{split}_adjusted_predictions.csv")
  y=original["true_family"].astype(str).to_numpy()
  orig=metric(y,original["predicted_family"].astype(str).to_numpy())
  adj=metric(y,adjusted["adjusted_best_guess_class"].astype(str).to_numpy())
  oa=original["decision"].astype(str).isin(["Normal","Known_attack","Known attack"]).to_numpy()
  aa=adjusted["adjusted_decision"].astype(str).isin(["Normal","Known_attack","Known attack"]).to_numpy()
  orig["selective_coverage"]=float(oa.mean()); orig["selective_accepted_accuracy"]=float(np.mean(original.loc[oa,"predicted_family"].astype(str).to_numpy()==y[oa])) if oa.any() else None
  adj["selective_coverage"]=float(aa.mean()); adj["selective_accepted_accuracy"]=float(np.mean(adjusted.loc[aa,"adjusted_predicted_category"].astype(str).to_numpy()==y[aa])) if aa.any() else None
  out[f"{ds}_{split}"]={"original":orig,"posthoc_adjusted":adj}
bias=json.loads((POST/"class_prior_bias.json").read_text(encoding="utf-8"))
payload={"disclosure":"This decision layer / v2 head was designed after the primary model's test results were seen; its test numbers are not an unbiased estimate and are shown beside the original, pre-registered result.","bias_sha256":bias["bias_sha256"],"validation_before":json.loads((POST/"posthoc_adjustment_summary.json").read_text())["validation_before"],"validation_after":json.loads((POST/"posthoc_adjustment_summary.json").read_text())["validation_after"],"test_metrics":out}
(POST/"posthoc_test_metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
lines=["","## S2c — post-hoc validation-tuned class-prior decision layer","",payload["disclosure"],"",f"Bias SHA-256: {payload['bias_sha256']}","", "| Dataset / split | View | Accuracy | Macro-F1 | Weighted-F1 | Selective coverage | Accepted accuracy |","|---|---|---:|---:|---:|---:|---:|"]
for key,v in out.items():
 for view,label in (("original","Original frozen decision layer"),("posthoc_adjusted","Post-hoc adjusted")):
  x=v[view]; lines.append(f"| {key} | {label} | {x['accuracy']:.4%} | {x['macro_f1']:.4%} | {x['weighted_f1']:.4%} | {x['selective_coverage']:.4%} | {x['selective_accepted_accuracy']:.4%} |")
p=ROOT/"docs/EVALUATION_REPORT.md"; current=p.read_text(encoding="utf-8")
if "## S2c — post-hoc validation-tuned class-prior decision layer" in current:
 current=current.split("## S2c — post-hoc validation-tuned class-prior decision layer")[0].rstrip()
p.write_text(current+"\n"+"\n".join(lines)+"\n",encoding="utf-8")
print("WROTE",POST/"posthoc_test_metrics.json")
print(json.dumps(out,indent=2))
