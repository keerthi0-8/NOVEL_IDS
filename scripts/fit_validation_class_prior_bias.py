from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd, yaml
from novel_ids.decision.stage12 import DecisionThresholds, apply_decision_policy
ROOT=Path.cwd(); RUN=Path(r"C:\NOVEL_IDS_runs\phaseC_recovery_20261126")
VAL=Path(r"C:\NOVEL_IDS_runs\phaseE_validation_repro_20261126\validation_distributions.npz")
OUT=RUN/"posthoc_class_prior_adjustment"; OUT.mkdir(parents=True,exist_ok=True)
classes=list(yaml.safe_load((ROOT/"config/taxonomy_unified.yaml").read_text(encoding="utf-8"))["classes"])
z=np.load(VAL)
def macro_f1(y,p):
 cm=np.bincount(y*len(classes)+p,minlength=len(classes)**2).reshape(len(classes),len(classes))
 s=cm.sum(1); pr=cm.sum(0); tp=np.diag(cm)
 precision=np.divide(tp,pr,out=np.zeros(len(classes)),where=pr>0)
 recall=np.divide(tp,s,out=np.zeros(len(classes)),where=s>0)
 f=np.divide(2*precision*recall,precision+recall,out=np.zeros(len(classes)),where=(precision+recall)>0)
 return float(f[s>0].mean()),float((y==p).mean())
data={}
for ds in ("nsl_kdd","unsw_nb15"):
 probs=np.asarray(z[ds+"_probabilities"],dtype=np.float64); y=np.asarray(z[ds+"_labels"],dtype=np.int64)
 if probs.shape[1]!=len(classes) or not np.isfinite(probs).all(): raise RuntimeError("Invalid validation probability matrix")
 logits=np.log(np.clip(probs,1e-12,1.0))
 data[ds]=(logits,y)
grid=np.round(np.arange(-2.0,2.0001,.25),2); b=np.zeros(len(classes),dtype=np.float64)
def objective(candidate):
 vals=[]
 for ds,(logits,y) in data.items():
  p=np.argmax(logits+candidate[None,:],axis=1)
  vals.append(macro_f1(y,p)[0])
 return float(np.mean(vals)-.002*np.sum(candidate*candidate)),vals
seed=42; rng=np.random.default_rng(seed); trace=[]
for pass_no in range(3):
 coords=np.arange(len(classes)); rng.shuffle(coords)
 for j in coords:
  best=(-1e9,None,None)
  for value in grid:
   candidate=b.copy(); candidate[j]=value
   score,vals=objective(candidate)
   if score>best[0]+1e-12: best=(score,float(value),vals)
  b[j]=best[1]
  trace.append({"pass":pass_no+1,"class":classes[int(j)],"bias":float(b[j]),"objective":best[0],"dataset_macro_f1":dict(zip(data.keys(),best[2]))})
base={}; tuned={}
for ds,(logits,y) in data.items():
 p0=np.argmax(logits,axis=1); p1=np.argmax(logits+b[None,:],axis=1)
 f0,a0=macro_f1(y,p0); f1,a1=macro_f1(y,p1)
 base[ds]={"macro_f1":f0,"accuracy":a0}; tuned[ds]={"macro_f1":f1,"accuracy":a1}
payload={"method":"additive per-class bias on calibrated log probabilities","grid":[float(x) for x in grid],"passes":3,"seed":seed,"objective":"mean of per-dataset validation macro-F1 minus 0.002 * sum(bias^2)","classes":classes,"bias":{c:float(v) for c,v in zip(classes,b)},"validation_before":base,"validation_after":tuned,"trace":trace}
bias_bytes=json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
payload["bias_sha256"]=hashlib.sha256(bias_bytes).hexdigest()
(OUT/"class_prior_bias.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
# Apply once to frozen saved prediction probabilities only; no model forward/inference.
results={}
for ds in ("nsl_kdd","unsw_nb15"):
 for split in ("test_independent","test_full"):
  src=RUN/"predictions"/f"{ds}_{split}_primary.csv"; df=pd.read_csv(src)
  prob_cols=["prob_"+c for c in classes]
  probs=df[prob_cols].to_numpy(dtype=np.float64)
  if not np.isfinite(probs).all(): raise RuntimeError("Non-finite saved test probabilities")
  logits=np.log(np.clip(probs,1e-12,1.0)); adj=logits+b[None,:]; adj-=adj.max(1,keepdims=True); ap=np.exp(adj); ap/=ap.sum(1,keepdims=True)
  thresholds=DecisionThresholds(float(payload_threshold:=json.loads((RUN/"results.json").read_text(encoding="utf-8"))["datasets"][ds][split]["primary"]["thresholds"]["tau_ood"]),float(json.loads((RUN/"results.json").read_text(encoding="utf-8"))["datasets"][ds][split]["primary"]["thresholds"]["tau_conf"]))
  dec=apply_decision_policy(ap,df["ood_score"].to_numpy(dtype=float),classes,thresholds)
  out=df.copy()
  for i,c in enumerate(classes): out["adjusted_prob_"+c]=ap[:,i].astype(np.float32)
  out["adjusted_best_guess_class"]=np.asarray(classes,dtype=object)[ap.argmax(1)]
  out["adjusted_best_guess_probability"]=ap.max(1).astype(np.float32)
  out["adjusted_decision"]=dec["decision"].to_numpy()
  out["adjusted_predicted_category"]=dec["predicted_category"].to_numpy()
  out["adjusted_confidence"]=dec["confidence"].to_numpy()
  out.to_csv(OUT/f"{ds}_{split}_adjusted_predictions.csv",index=False)
  results[f"{ds}_{split}"]={"rows":len(out),"original_decisions":df.decision.astype(str).value_counts().to_dict(),"adjusted_decisions":out.adjusted_decision.astype(str).value_counts().to_dict(),"prediction_file":str(OUT/f"{ds}_{split}_adjusted_predictions.csv")}
summary={"disclosure":"This decision layer / v2 head was designed after the primary model's test results were seen; its test numbers are not an unbiased estimate and are shown beside the original, pre-registered result.","bias_sha256":payload["bias_sha256"],"validation_before":base,"validation_after":tuned,"test_saved_prediction_adjustments":results,"ship_in_inference_path":bool((np.mean([tuned[d]["macro_f1"]-base[d]["macro_f1"] for d in data])>=.01) and all(tuned[d]["accuracy"]-base[d]["accuracy"]>=-.02 for d in data)),"ship_rule":"mean validation macro-F1 gain >= 0.01 and no dataset validation accuracy drop > 0.02"}
(OUT/"posthoc_adjustment_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))

