from __future__ import annotations
import argparse, gc, hashlib, json, os, shutil, sys, time
from pathlib import Path
from types import MethodType
import numpy as np
import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # Stage6 project loader resolves approved artifacts relative to the project root.
sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(ROOT/"src"))
import run_phaseC_one_shot as c
import run_stage7_11_experiment as r
ORIGINAL_SETUP_SOURCE=c.setup_source

RUN=Path(r"C:\NOVEL_IDS_runs\phaseC_recovery_20261126")
ORIGINAL_RUN=Path(r"C:\NOVEL_IDS_runs\phaseC_one_shot_20261126")
MAP_PATH=ROOT/"config/phaseC_label_map.yaml"
RAW={
 "nsl_kdd":(Path(r"G:\My Drive\NSL-KDD\NSL_KDD-master\KDDTest+.txt"),None,41,22544),
 "unsw_nb15":(Path(r"G:\My Drive\UNSW-NB15\UNSW_NB15_training-set.csv"),0,"attack_cat",82332),
}
TAX_CLASSES=None
MAPPINGS=None

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()

def norm(x): return str(x).strip().casefold()

def load_maps():
 cfg=yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
 out={}
 for ds in ("nsl_kdd","unsw_nb15"):
  mp={}
  if ds=="nsl_kdd":
   for fam,names in cfg[ds]["families"].items():
    for name in names: mp[norm(name)]=cfg[ds]["unified"][fam]
   mp["normal"]="Normal"
  else:
   mp={norm(k):v for k,v in cfg[ds]["mappings"].items()}
   for alias,target in cfg[ds].get("aliases",{}).items(): mp[norm(alias)]=mp[norm(target)]
  out[ds]=mp
 return cfg,out

def mapped_name(ds,raw):
 return MAPPINGS[ds].get(norm(raw),"UNSUPPORTED_"+str(raw).strip())

def raw_label_audit():
 report={"source":"original raw test files; label column only (pandas usecols); no feature values read","datasets":{}}
 for ds,(path,header,col,expected) in RAW.items():
  if not path.is_file(): raise RuntimeError(f"Raw label source missing: {path}")
  kwargs={"header":header,"usecols":[col],"dtype":str,"keep_default_na":False}
  frame=pd.read_csv(path,**kwargs)
  vals=frame.iloc[:,0].astype(str).str.strip()
  if len(vals)!=expected: raise RuntimeError(f"{ds} raw test row count {len(vals)} != expected {expected}")
  unique=sorted(vals.unique().tolist(),key=str.casefold)
  pairs=[{"raw":x,"mapped":mapped_name(ds,x)} for x in unique]
  unsupported=[x for x in pairs if x["mapped"].startswith("UNSUPPORTED_")]
  report["datasets"][ds]={"source_path":str(path),"source_sha256":sha(path),"rows":len(vals),
                           "unique_label_count":len(unique),"labels":pairs,"unsupported_labels":unsupported}
  if unsupported: raise RuntimeError(f"Unsupported raw labels require explicit approved policy: {unsupported}")
 return report

def install_mapping(taxonomy):
 global TAX_CLASSES
 TAX_CLASSES=tuple(taxonomy.classes)
 index={x:i for i,x in enumerate(TAX_CLASSES)}
 def map_labels(self,ds,labels):
  names=[mapped_name(ds,x) for x in np.asarray(labels).astype(str)]
  return np.asarray([index.get(x,-1) for x in names],dtype=np.int64)
 taxonomy.__class__.map_labels=map_labels

def setup_source_with_mapping():
 original=r.load_labels
 def wrapped(taxonomy):
  install_mapping(taxonomy)
  return original(taxonomy)
 r.load_labels=wrapped
 try:
  return ORIGINAL_SETUP_SOURCE()
 finally:
  r.load_labels=original

def synthetic_checks():
 cases=[("nsl_kdd"," saint ","Probe_Recon"),("nsl_kdd","normal","Normal"),
        ("nsl_kdd","odd_attack","UNSUPPORTED_odd_attack"),
        ("unsw_nb15"," Backdoors ","UNSW_Backdoor"),
        ("unsw_nb15","DoS","DoS"),("unsw_nb15","???","UNSUPPORTED_???")]
 for ds,raw,want in cases:
  got=mapped_name(ds,raw)
  if got!=want: raise AssertionError((ds,raw,got,want))
 return {"cases":len(cases),"result":"PASS"}


def print_multiclass_report(metrics, classes, model_name="NOVEL-IDS"):
    """Print metrics calculated from the actual evaluation predictions."""

    print("\n" + "=" * 75)
    print(f"{model_name} — MULTICLASS EVALUATION REPORT")
    print("=" * 75)

    print(f"Accuracy       : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Macro Precision: {metrics['precision']:.4f}")
    print(f"Macro Recall   : {metrics['recall']:.4f}")
    print(f"Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1    : {metrics['weighted_f1']:.4f}")
    print(f"FPR            : {metrics['fpr']:.4f}")
    print(f"FNR            : {metrics['fnr']:.4f}")

    print("\nPER-CLASS RESULTS")
    print("-" * 75)
    print(f"{'Class':<22}{'Precision':>11}{'Recall':>11}{'F1':>11}{'Support':>12}")
    print("-" * 75)

    for class_name in classes:
        values = metrics.get("per_class", {}).get(class_name)

        if values is None:
            continue

        print(
            f"{class_name:<22}"
            f"{values['precision']:>11.4f}"
            f"{values['recall']:>11.4f}"
            f"{values['f1']:>11.4f}"
            f"{values['support']:>12}"
        )

    print("\nCONFUSION MATRIX")
    print("-" * 75)
    print("Class order:", list(classes))
    print(np.asarray(metrics["confusion_matrix"]))

    print("=" * 75 + "\n")


def prediction_eval(v,helper,branches,artifact,ds,split,labels,aug_cfg,level,confidence,group_only,level3_mask,ood=None,cal=None):
 arr=artifact[split]
 # The shared source_batch helper reads split arrays from helper.datasets; register
 # the already-opened, frozen test partition explicitly before forwarding rows.
 helper.datasets[ds][split]=arr
 raw=np.asarray(arr["labels"]).astype(str)
 true_names=np.asarray([mapped_name(ds,x) for x in raw],dtype=object)
 classes=list(r.taxonomy.classes); c2i={x:i for i,x in enumerate(classes)}
 logits_all=[]; z_all=[]; gate_all=[]
 n=len(raw); t0=time.time()
 for st in range(0,n,512):
  if r.free_ram_gb()<2.0: raise RuntimeError("HARD STOP free RAM below 2 GiB during Phase C")
  if shutil.disk_usage("C:\\").free<10*1024**3: raise RuntimeError("HARD STOP free disk below 10 GiB during Phase C")
  if torch.cuda.max_memory_allocated()>6*1024**3: raise RuntimeError("HARD STOP GPU allocation above 6 GiB")
  ix=np.arange(st,min(n,st+512))
  lo,z,g=c.forward_rows(v,helper,branches,ds,split,ix,labels,aug_cfg,level,confidence,group_only,level3_mask)
  logits_all.append(lo.cpu()); z_all.append(z.cpu()); gate_all.append(g.cpu())
  if time.time()-t0>=60:
   c.log(f"TEST_PROGRESS model={v['name']} dataset={ds} split={split} rows={min(n,st+512)}/{n} free_ram_gb={r.free_ram_gb():.2f}")
   t0=time.time()
 lo=torch.cat(logits_all); z=torch.cat(z_all); gate=torch.cat(gate_all)
 probs0=torch.softmax(lo,dim=1).numpy()
 if v["head_mode"]=="native":
  native=list(labels[ds]["native_classes"])
  agg=np.zeros((n,len(classes)),dtype=np.float32)
  for j,name in enumerate(native):
   fam=mapped_name(ds,name)
   if fam in c2i: agg[:,c2i[fam]]+=probs0[:,j]
  probs=agg
 else:
  probs=probs0
 pred_idx=probs.argmax(1)
 pred_names=np.asarray(classes,dtype=object)[pred_idx]
 confidence_values=probs.max(1)
 ood_scores=np.full(n,np.nan,dtype=np.float64)
 unfamiliar=np.zeros(n,dtype=bool); uncertain=np.zeros(n,dtype=bool)
 if ood is not None and cal is not None:
  mahal,comb=ood
  score=comb.score(mahal.score(z),c.energy_score(lo,temperature=float(cal["temperature"]))).cpu().numpy()
  ood_scores=score
  unfamiliar=score>float(cal["tau_ood_validation_p95"])
  uncertain=(confidence_values<float(cal["tau_conf_validation_p10"]))&(~unfamiliar)
 decision=np.where(unfamiliar,"Unfamiliar",np.where(uncertain,"Uncertain",np.where(pred_names=="Normal","Normal","Known_attack")))
 outdir=RUN/"predictions"; outdir.mkdir(parents=True,exist_ok=True)
 outpath=outdir/f"{ds}_{split}_{v['name']}.csv"
 frame=pd.DataFrame({"row_index":np.arange(n),"true_raw":raw,"true_family":true_names,
                     "predicted_family":pred_names,"confidence":confidence_values,
                     "ood_score":ood_scores,"decision":decision})
 for j,name in enumerate(classes): frame[f"prob_{name}"]=probs[:,j]
 frame.to_csv(outpath,index=False)
 # Stage 2: reload only the saved prediction file and calculate metrics from it.
 saved=pd.read_csv(outpath)
 known=~saved["true_family"].astype(str).str.startswith("UNSUPPORTED_")
 supported=saved.loc[known]
 if len(supported)==0: raise RuntimeError(f"No supported labels in {ds}/{split}")
 yt=np.asarray([c2i[x] for x in supported["true_family"]],dtype=np.int64)
 yp=np.asarray([c2i[x] for x in supported["predicted_family"]],dtype=np.int64)
 pmat=supported[[f"prob_{x}" for x in classes]].to_numpy(dtype=np.float64)
 from novel_ids.evaluation.stage11_metrics import classification_metrics
 metrics=classification_metrics(yt,yp,classes,probabilities=pmat)
 print_multiclass_report(
    metrics,
    classes,
    model_name="Phase C Evaluation"
)
 metrics.update({"rows":int(n),"known_label_rows":int(known.sum()),
                 "unsupported_rows":int((~known).sum()),
                 "unsupported_labels":sorted(set(saved.loc[~known,"true_raw"].astype(str))),
                 "prediction_file":str(outpath),"prediction_sha256":sha(outpath),
                 "gate_mean":float(gate.mean()),"gate_per_dimension_mean":gate.mean(0).tolist()})
 if ood is not None and cal is not None:
  score=saved["ood_score"].to_numpy(dtype=float)
  train_labels=set(map(str,labels[ds]["native_train"]))
  test_only=np.asarray([(x not in train_labels and norm(x)!="normal") for x in raw],dtype=bool)
  ood_positive=(~known.to_numpy()) | (test_only if ds=="nsl_kdd" else False)
  oy=ood_positive.astype(int)
  from novel_ids.evaluation.stage11_metrics import _auc,_ap,_fpr95
  metrics["ood"]={"auroc":_auc(oy,score),"aupr":_ap(oy,score),"fpr_at_95_tpr":_fpr95(oy,score),
                  "positive_rows":int(oy.sum()),"test_only_attack_rows":int(test_only.sum()) if ds=="nsl_kdd" else 0}
  metrics["ood_score_summary"]={"mean":float(np.mean(score)),"median":float(np.median(score)),
                                "p95":float(np.quantile(score,.95)),"p99":float(np.quantile(score,.99))}
  metrics["thresholds"]={"temperature":float(cal["temperature"]),"tau_ood":float(cal["tau_ood_validation_p95"]),
                         "tau_conf":float(cal["tau_conf_validation_p10"])}
  metrics["decision_counts"]=saved["decision"].value_counts().to_dict()
  if ds=="nsl_kdd":
   sub=saved.loc[test_only,"decision"].value_counts().to_dict()
   metrics["test_only_attack_decisions"]=sub
   metrics["test_only_attack_family_accuracy"]=float(np.mean(
       saved.loc[test_only & known,"true_family"].to_numpy()==saved.loc[test_only & known,"predicted_family"].to_numpy()
   )) if np.any(test_only & known) else None
 return metrics

def preflight():
 global MAPPINGS
 _,MAPPINGS=load_maps()
 print("SYNTHETIC",json.dumps(synthetic_checks()),flush=True)
 audit=raw_label_audit()
 print("RAW_LABEL_AUDIT",json.dumps({k:{"rows":v["rows"],"unique":v["unique_label_count"],"unsupported":v["unsupported_labels"]} for k,v in audit["datasets"].items()}),flush=True)
 c.preflight()
 print("VALIDATION_AND_LABEL_PREFLIGHT_PASS",flush=True)

def execute():
 global MAPPINGS
 _,MAPPINGS=load_maps()
 if RUN.exists() and (RUN/"results.json").exists(): raise RuntimeError("Recovery results already exist; refusing rerun")
 RUN.mkdir(parents=True,exist_ok=True)
 # Preserve the original freeze manifest, adding this recovery evaluator and approved label map.
 m=json.loads((ORIGINAL_RUN/"freeze_manifest.json").read_text(encoding="utf-8"))
 # Refresh the Phase-C evaluator entry after the approved memory-lifecycle patch.
 for item in m["files"]:
  if item.get("name")=="phaseC_evaluator_script":
   item["sha256"]=sha(item["path"]); item["bytes"]=Path(item["path"]).stat().st_size
 m["files"].append({"path":str(Path(__file__).resolve()),"sha256":sha(__file__),"name":"phaseC_recovery_evaluator","bytes":Path(__file__).stat().st_size})
 m["files"].append({"path":str(MAP_PATH),"sha256":sha(MAP_PATH),"name":"phaseC_evaluation_label_map","bytes":MAP_PATH.stat().st_size})
 vocab_report=RUN/"raw_label_vocabulary_audit.json"
 if not vocab_report.is_file(): raise RuntimeError("Label-only raw-source vocabulary audit is missing")
 m["files"].append({"path":str(vocab_report),"sha256":sha(vocab_report),"name":"phaseC_raw_label_vocabulary_audit","bytes":vocab_report.stat().st_size})
 (RUN/"freeze_manifest.json").write_text(json.dumps(m,indent=2),encoding="utf-8")
 c.RUN=RUN; c.MANIFEST=RUN/"freeze_manifest.json"
 c.log=lambda msg: (print(time.strftime("%Y-%m-%d %H:%M:%S"),msg,flush=True),
                    (RUN/"phaseC_recovery_progress.log").open("a",encoding="utf-8").write(time.strftime("%Y-%m-%d %H:%M:%S")+" "+msg+"\n"))
 c.setup_source=setup_source_with_mapping
 c.eval_transformer=prediction_eval
 c.execute()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--preflight",action="store_true")
 ap.add_argument("--execute",action="store_true")
 a=ap.parse_args()
 if a.preflight==a.execute: raise SystemExit("Specify exactly one of --preflight or --execute")
 if a.preflight: preflight()
 else: execute()
if __name__=="__main__": main()
