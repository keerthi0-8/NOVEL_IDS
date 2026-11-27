from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
RUN=Path(r"C:\NOVEL_IDS_runs\phaseD_cic_second_run_20261127")
LABELS=pd.read_csv(RUN/"CIC_DDoS2019_second_run_eval_labels.csv")["label"].astype(str)
def map_label(v):
 s=str(v).strip()
 if s.upper()=="BENIGN": return "Normal",""
 if s.lower()=="webddos": return "DoS","partial"
 known={"ldap","mssql","netbios","portmap","syn","udp","udp-lag","udplag","tftp","ntp","dns","ssdp"}
 if s.lower() in known or s.lower().startswith("drdos_") or s.lower().startswith("ddos_"):
  return "DoS",""
 return "UNSUPPORTED_"+s,"unsupported"
def signal_name(x):
 if x is None or not np.isfinite(x): return "not available"
 if x<0.60: return "no usable signal"
 if x<=0.75: return "weak signal"
 return "useful signal"
all_results={}
for mode in ("option_a","option_b"):
 pred=pd.read_csv(RUN/mode/"predictions.csv")
 summary=json.loads((RUN/mode/"summary.json").read_text(encoding="utf-8"))
 mapped=[map_label(v) for v in LABELS]
 truth=np.array([x[0] for x in mapped],dtype=object)
 partial=[i for i,x in enumerate(mapped) if x[1]=="partial"]
 valid=np.array([not x.startswith("UNSUPPORTED_") for x in truth])
 y=(truth[valid]=="DoS").astype(int)
 forced=(pred.loc[valid,"best_guess_class"].astype(str).to_numpy()!="Normal").astype(int)
 pnormal=pd.to_numeric(pred.loc[valid,"prob_Normal"],errors="coerce").to_numpy()
 score_prob=1.0-pnormal
 ood=pd.to_numeric(pred.loc[valid,"ood_score"],errors="coerce").to_numpy()
 def auc_score(score,kind):
  mask=np.isfinite(score)
  if len(np.unique(y[mask]))<2:return {"auroc":None,"aupr":None,"signal":"not available"}
  a=float(roc_auc_score(y[mask],score[mask])); ap=float(average_precision_score(y[mask],score[mask]))
  return {"auroc":a,"aupr":ap,"signal":signal_name(a)}
 cm=confusion_matrix(y,forced,labels=[0,1])
 decisions=pred["decision"].astype(str).replace({"Known_attack":"Known attack"})
 breakdown={k:int((decisions==k).sum()) for k in ("Normal","Known attack","Uncertain","Unfamiliar")}
 breakdown_pct={k:(v/len(pred)) for k,v in breakdown.items()}
 accepted=decisions.isin(["Normal","Known attack"]).to_numpy()
 best=pred["best_guess_class"].astype(str).to_numpy()
 y_family=truth
 accepted_accuracy=float(np.mean((best==y_family)[accepted & valid])) if np.any(accepted & valid) else None
 per_label={}
 for raw in sorted(set(LABELS)):
  idx=np.flatnonzero((LABELS.to_numpy()==raw)&valid)
  if not len(idx):continue
  per_label[str(raw)]={"rows":int(len(idx)),"forced_attack_detection_rate":float(np.mean(pred.iloc[idx]["best_guess_class"].astype(str).to_numpy()!="Normal"))}
 result={
  "normalization_mode":summary.get("normalization_mode",mode),
  "rows":int(len(pred)),"evaluated_rows":int(valid.sum()),"unsupported_rows":int((~valid).sum()),
  "webddos_partial_rows":len(partial),
  "decision_breakdown":breakdown,"decision_breakdown_rate":breakdown_pct,
  "compatibility":summary.get("compatibility"),"trust_level":summary.get("trust_level"),
  "forced_binary":{"accuracy":float(accuracy_score(y,forced)),"precision":float(precision_score(y,forced,zero_division=0)),"recall":float(recall_score(y,forced,zero_division=0)),"f1":float(f1_score(y,forced,zero_division=0)),"fpr":float(cm[0,1]/max(cm[0].sum(),1)),"fnr":float(cm[1,0]/max(cm[1].sum(),1)),"confusion_matrix_labels":["Normal","DoS"],"confusion_matrix":cm.tolist()},
  "threshold_free_signal":{"one_minus_P_Normal":auc_score(score_prob,"probability"),"ood_score":auc_score(ood,"ood")},
  "selective":{"coverage":float(accepted.mean()),"accepted_accuracy":accepted_accuracy},
  "per_attack_label_detection_rate":per_label,
  "per_attack_file_detection_rate":"not available: the existing 13-file aggregate cic_sample.csv did not preserve the original source filename for each row.",
 }
 all_results[mode]=result
 pd.DataFrame(cm,index=["actual_Normal","actual_DoS"],columns=["pred_Normal","pred_DoS"]).to_csv(RUN/f"{mode}_binary_confusion.csv")
first={"rows":162276,"exact_matches":0,"dataset_specific_features":87,"decision_breakdown":{"Unfamiliar":162276},"source":"saved first CIC run"}
report={"run_description":"Second CIC run after fixing the onboarding matching bug; mapping was defined from CICFlowMeter documentation and unlabeled feature descriptors before predictions; no CIC labels/performance used for mapping. Option A is headline zero-shot with unsupervised normalization. Option B is strict source-training statistics for Level-1 numeric features.","sample_manifest":json.loads((RUN/"cic_second_run_manifest.json").read_text(encoding="utf-8")),"mapping":{"level_counts":{"1":4,"2":72,"3":10},"compatibility_score":(4+0.5*72)/86,"trust_level":"Medium","missing_source_common_features":["network_service"],"mapping_sha256":"6cccab77a7892d469731955a519405a6eb1d4b370a3cacfbc7aa6a31f957eb59"},"first_run":first,"second_run":all_results,"interpretation":"If the resulting compatibility remains modest or the threshold-free metrics show no usable signal, transfer is not demonstrated. No further CIC iteration is permitted today."}
(RUN/"cic_second_run_evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
lines=["# CIC-DDoS2019 second-run evaluation","",report["run_description"],"","## Mapping and sampling","",f"- Rows: {report['sample_manifest']['sampled_rows']:,}; seed 42; per-class cap 15,000.",f"- Mapping levels: L1=4, L2=72, L3=10; compatibility={report['mapping']['compatibility_score']:.4f}; trust=Medium.", "- Missing source common feature: network_service.", "- First run: 162,276 rows; 0 exact feature matches; 87 dataset-specific features; all Unfamiliar.","","## Option A and Option B results",""]
for mode,x in all_results.items():
 lines += [f"### {mode.upper()} — {x['normalization_mode']}","",f"- Decision breakdown: {x['decision_breakdown']}",f"- Compatibility: {x['compatibility']}",f"- Trust: {x['trust_level']}",f"- Forced binary metrics: {x['forced_binary']}",f"- AUROC/AUPR for 1-P(Normal): {x['threshold_free_signal']['one_minus_P_Normal']}",f"- AUROC/AUPR for OOD score: {x['threshold_free_signal']['ood_score']}",f"- Selective coverage: {x['selective']['coverage']:.4%}; accepted accuracy: {x['selective']['accepted_accuracy']}",f"- Per-attack-file detection rate: {x['per_attack_file_detection_rate']}",""]
lines += ["## Interpretation","", "The AUROC wording rule is applied separately to 1-P(Normal) and the OOD score: below 0.60 = no usable signal; 0.60–0.75 = weak signal; above 0.75 = useful signal. Compatibility is modest; do not claim zero-shot success unless the actual second-run results support it. Per-attack-file rates cannot be computed because the aggregate sample has no per-row source filename. No further CIC iteration was performed.",""]
(RUN/"CIC_SECOND_RUN_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print("Evaluation script ready; will read only saved predictions and evaluation-only labels after Option A and B finish.")
