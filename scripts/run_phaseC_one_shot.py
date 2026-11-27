from __future__ import annotations
import argparse, gc, hashlib, json, os, sys, time, shutil
from pathlib import Path
import joblib
import numpy as np
import torch
import yaml
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"src"))
import run_stage7_11_experiment as r
import run_baselines_stage7_11 as b
from novel_ids.fusion.stage7 import AdaptiveResidualFusion
from novel_ids.alignment.stage8 import Stage8SelectiveAlignment,NSL_KDD,UNSW_NB15
from novel_ids.fusion.stage9 import UnifiedRepresentation
from novel_ids.transformer.stage10 import Stage10Transformer
from novel_ids.classification.stage11 import Stage11Classifier,NativeAblationHeads,compute_class_weights
from novel_ids.taxonomy.stage11 import load_taxonomy
from novel_ids.ood.stage11 import ClassConditionalMahalanobis,OODScoreCombiner,energy_score
from novel_ids.evaluation.stage11_metrics import classification_metrics

RUN=Path(r"C:\NOVEL_IDS_runs\phaseC_one_shot_20261126")
MANIFEST=RUN/"freeze_manifest.json"
CAL=Path(r"C:\NOVEL_IDS_runs\phaseA_primary_seed43_final_retry2\phaseA_calibration.json")
BASE=Path(r"C:\NOVEL_IDS_runs\baselines_ablations\source_baselines_20261125_r1")
STAGE5=Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
STAGE6=Path(r"C:\NOVEL_IDS_runs\stage6\accepted_candidate09_lambda1_seed42\best.pt")
STAGE5_SHA="9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"
STAGE6_SHA="1c66f6f8d1a029266e5a637b5db4d6f49881988bf16462d2c3fbc09cd704ecbd"
PRIMARY=Path(r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43\best.pt")
VARIANTS=[
 ("primary",PRIMARY,"full","unified"),
 ("shared_only",Path(r"C:\NOVEL_IDS_runs\phaseB_final_shared_only_seed42\best.pt"),"shared_only","unified"),
 ("residual_fixed_gate1",Path(r"C:\NOVEL_IDS_runs\phaseB_final_residual_fixed_gate1_seed42\best.pt"),"residual_fixed_gate1","unified"),
 ("residual_adaptive_alignment_off",Path(r"C:\NOVEL_IDS_runs\phaseB_final_residual_adaptive_seed42\best.pt"),"residual_adaptive","unified"),
 ("common_only_reduced_1000",Path(r"C:\NOVEL_IDS_runs\phaseB_final_common_only_seed42_batch64_1000\best.pt"),"common_only","unified"),
 ("native_head_residual_adaptive",Path(r"C:\NOVEL_IDS_runs\phaseB_final_residual_adaptive_native_seed42\best.pt"),"residual_adaptive","native"),
]
MLP_SEEDS={"nsl_kdd":{"native":42,"unified":44},"unsw_nb15":{"native":44,"unified":43}}
DATA_ART={
 "nsl_kdd":ROOT/"data/processed/unified_representation/nsl_kdd_unified_v2_no_kddtest_feature_overlap.joblib",
 "unsw_nb15":ROOT/"data/processed/unified_representation/unsw_nb15_unified_v2_no_test_feature_overlap.joblib",
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
 return h.hexdigest()
def log(msg):
 line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
 print(line,flush=True)
 with (RUN/"phaseC_progress.log").open("a",encoding="utf-8") as f:f.write(line+"\n")
def verify_manifest():
 m=json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
 for item in m["files"]:
  p=Path(item["path"])
  if not p.is_file(): raise RuntimeError(f"HARD STOP frozen file missing: {item['name']}")
  actual=sha(p)
  if actual.lower()!=item["sha256"].lower(): raise RuntimeError(f"HARD STOP frozen hash mismatch: {item['name']} {actual}")
 if sha(STAGE5).lower()!=STAGE5_SHA or sha(STAGE6).lower()!=STAGE6_SHA:
  raise RuntimeError("HARD STOP frozen Stage5/6 snapshot hash mismatch")
 r.verify_protected()
 return m
def setup_source():
 if not torch.cuda.is_available(): raise RuntimeError("HARD STOP CUDA unavailable")
 torch.set_num_threads(2); torch.set_num_interop_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 r.RUN_DIR=RUN/("loader_"+str(os.getpid())); r.RUN_DIR.mkdir(parents=True,exist_ok=True)
 r.DEVICE=torch.device("cuda")
 r.taxonomy=load_taxonomy(ROOT/"config/taxonomy_unified.yaml",purpose="evaluation")
 helper,branches=r.read_data()
 r.helper=helper
 labels=r.load_labels(r.taxonomy)
 ck5=torch.load(STAGE5,map_location="cpu",weights_only=False)
 aug_cfg=ck5["config"]; del ck5
 level,confidence,group_only,level3_mask=r.make_metadata()
 r.COMMON_MASK=(level==1)
 return helper,branches,labels,aug_cfg,level,confidence,group_only,level3_mask
def init_variant(spec,helper,branches,labels,aug_cfg,level,confidence,group_only,level3_mask,fit_refs):
 name,path,ablation,head_mode=spec
 r.ABLATION_MODE=ablation; r.HEAD_MODE=head_mode
 stage7=AdaptiveResidualFusion().to(r.DEVICE).eval()
 stage8=Stage8SelectiveAlignment(coral_lambda=0.05,mode=("compatibility" if ablation=="full" else "no_alignment")).to(r.DEVICE).eval()
 stage9=UnifiedRepresentation()
 stage10=Stage10Transformer().to(r.DEVICE).eval()
 mapped=np.concatenate([labels[d]["train"] for d in r.DATASETS])
 cw,_=compute_class_weights([r.taxonomy.classes[int(i)] for i in mapped],r.taxonomy.classes,clip_ratio=15.0)
 if head_mode=="native":
  native={d:labels[d]["native_classes"] for d in r.DATASETS}
  classifier=NativeAblationHeads(native).to(r.DEVICE).eval()
 else:
  classifier=Stage11Classifier(len(r.taxonomy.classes),class_weights=cw).to(r.DEVICE).eval()
 ck=torch.load(path,map_location=r.DEVICE,weights_only=False)
 if ck.get("stage5_sha256")!=STAGE5_SHA or ck.get("stage6_sha256")!=STAGE6_SHA:
  raise RuntimeError(f"{name}: checkpoint provenance mismatch")
 stage7.load_state_dict(ck["stage7"],strict=True)
 stage8.load_state_dict(ck["stage8"],strict=True)
 stage10.load_state_dict(ck["stage10"],strict=True)
 classifier.load_state_dict(ck["classifier"],strict=True)
 del ck; gc.collect(); torch.cuda.empty_cache()
 if not bool(stage8.compatibility.references_fitted.item()):
  raise RuntimeError(f"{name}: frozen checkpoint lacks fitted Stage8 source-training references; refusing to refit")
 for m in (stage7,stage8,stage10,classifier): m.eval()
 return {"name":name,"path":path,"ablation":ablation,"head_mode":head_mode,
         "stage7":stage7,"stage8":stage8,"stage9":stage9,"stage10":stage10,"classifier":classifier}
def forward_rows(v,helper,branches,ds,split,indices,labels,aug_cfg,level,confidence,group_only,level3_mask):
 r.ABLATION_MODE=v["ablation"]; r.HEAD_MODE=v["head_mode"]
 idx=np.asarray(indices,dtype=np.int64)
 sidval=NSL_KDD if ds=="nsl_kdd" else UNSW_NB15
 bch=r.source_batch(helper,branches,ds,split,idx,False,
      torch.Generator(device=r.DEVICE).manual_seed(431+sidval),level3_mask,aug_cfg)
 sid=torch.full((len(idx),),sidval,device=r.DEVICE,dtype=torch.long)
 with torch.no_grad():
  logits,g,a,f=r.batch_forward(bch,v["stage7"],v["stage8"],v["stage9"],v["stage10"],
      v["classifier"],sid,level,confidence,group_only,1.0)
 if v["head_mode"]=="native": logits=logits[ds]
 return logits.float().detach(),f["Z_u"].float().detach(),g["G"].float().detach()
def fit_primary_ood(v,helper,branches,labels,aug_cfg,level,confidence,group_only,level3_mask,cal):
 zs=[]; ys=[]; ls=[]
 for di,ds in enumerate(r.DATASETS):
  n=len(helper.datasets[ds]["train"]["numeric_values"])
  rng=np.random.default_rng(4300+di)
  ix=np.sort(rng.choice(n,min(20000,n),replace=False))
  zz=[]; yy=[]; ll=[]
  for st in range(0,len(ix),256):
   batch=ix[st:st+256]
   lo,z,_=forward_rows(v,helper,branches,ds,"train",batch,labels,aug_cfg,level,confidence,group_only,level3_mask)
   zz.append(z.cpu()); ll.append(lo.cpu()); yy.append(labels[ds]["train"][batch])
  zs.append(torch.cat(zz)); ls.append(torch.cat(ll)); ys.append(np.concatenate(yy))
 train_z=torch.cat(zs); train_l=torch.cat(ls); train_y=torch.as_tensor(np.concatenate(ys),dtype=torch.long)
 o=cal["ood_combiner"]
 mahal=ClassConditionalMahalanobis(min_class_samples=20,shrinkage=0.1,ridge=1e-4).fit(train_z,train_y)
 comb=OODScoreCombiner(weights=(0.5,0.5))
 comb.mean_mahalanobis=float(o["train_mahal_mean"]); comb.std_mahalanobis=float(o["train_mahal_std"])
 comb.mean_energy=float(o["train_energy_mean"]); comb.std_energy=float(o["train_energy_std"])
 return mahal,comb
def basic_metrics(y,pred,classes):
 return classification_metrics(y,pred,classes)
def eval_transformer(v,helper,branches,artifact,ds,split,labels,aug_cfg,level,confidence,group_only,level3_mask,ood=None,cal=None):
 arr=artifact[split]
 helper.datasets[ds][split]=arr
 raw=np.asarray(arr["labels"]).astype(str)
 y=np.asarray(r.taxonomy.map_labels(ds,raw),dtype=np.int64)
 logits=[]; embeddings=[]; gates=[]
 n=len(y); t0=time.time()
 for st in range(0,n,512):
  if r.free_ram_gb()<2.0: raise RuntimeError("HARD STOP free RAM below 2 GiB during Phase C")
  if shutil.disk_usage("C:\\").free<10*1024**3: raise RuntimeError("HARD STOP free disk below 10 GiB during Phase C")
  if torch.cuda.max_memory_allocated()>6*1024**3: raise RuntimeError("HARD STOP GPU allocation above 6 GiB")
  ix=np.arange(st,min(n,st+512))
  lo,z,g=forward_rows(v,helper,branches,ds,split,ix,labels,aug_cfg,level,confidence,group_only,level3_mask)
  logits.append(lo.cpu()); embeddings.append(z.cpu()); gates.append(g.cpu())
  if time.time()-t0>=60:
   log(f"TEST_PROGRESS model={v['name']} dataset={ds} split={split} rows={min(n,st+512)}/{n} free_ram_gb={r.free_ram_gb():.2f} gpu_peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}")
   t0=time.time()
 lo=torch.cat(logits); z=torch.cat(embeddings); gate=torch.cat(gates)
 pred=lo.argmax(1).numpy()
 result=basic_metrics(y,pred,list(r.taxonomy.classes))
 result["rows"]=int(n); result["gate_mean"]=float(gate.mean()); result["gate_per_dimension_mean"]=gate.mean(0).tolist()
 if ood is not None and cal is not None:
  mahal,comb=ood; temp=float(cal["temperature"])
  probs=torch.softmax(lo/temp,dim=1)
  score=comb.score(mahal.score(z),energy_score(lo,temperature=temp))
  tau=float(cal["tau_ood_validation_p95"]); tc=float(cal["tau_conf_validation_p10"])
  conf=probs.max(1).values
  unfamiliar=score>tau; uncertain=(conf<tc)&(~unfamiliar); accepted=~(unfamiliar|uncertain)
  result.update({"temperature":temp,"tau_ood":tau,"tau_conf":tc,
   "ood_score":{"mean":float(score.mean()),"median":float(score.median()),"p95":float(torch.quantile(score,0.95)),"p99":float(torch.quantile(score,0.99))},
   "unfamiliar_rate":float(unfamiliar.float().mean()),"uncertain_rate":float(uncertain.float().mean()),
   "rejected_rate":float((~accepted).float().mean()),
   "accepted_accuracy":float(np.mean(y[accepted.numpy()]==pred[accepted.numpy()])) if accepted.any() else None,
   "status_counts":{}})
  pred_names=np.asarray(r.taxonomy.classes)[pred]
  un_np=unfamiliar.cpu().numpy(); uc_np=uncertain.cpu().numpy(); ac_np=accepted.cpu().numpy()
  result["status_counts"]={"Unfamiliar":int(un_np.sum()),"Uncertain":int(uc_np.sum()),
                    "Normal":int((ac_np & (pred_names=="Normal")).sum()),
                    "Known_attack":int((ac_np & (pred_names!="Normal")).sum())}
 return result
def baseline_matrix(arr,meta):
 ni=np.asarray(meta["numeric_positions"],dtype=int); ci=np.asarray(meta["categorical_positions"],dtype=int)
 num=np.asarray(arr["numeric_values"][:,ni],dtype=np.float32)
 cat=np.asarray(arr["categorical_values"][:,ci]).astype(str)
 miss=np.asarray(arr["M_miss"][:,ni],dtype=np.float32)
 avail=np.asarray(arr["M_avail"],dtype=np.float32)
 missall=np.asarray(arr["M_miss"],dtype=np.float32)
 med=np.asarray(meta["numeric_medians"],dtype=np.float32)
 num=np.where(np.isfinite(num),num,med).astype(np.float32)
 cats=[np.asarray(x,dtype=object) for x in meta["category_values"]]
 enc=OneHotEncoder(categories=cats,handle_unknown="ignore",sparse_output=False,dtype=np.float32)
 if len(cats): enc.fit(np.asarray([[x[0] for x in cats]],dtype=object)); catx=enc.transform(cat).astype(np.float32,copy=False)
 else: catx=np.empty((len(num),0),dtype=np.float32)
 X=np.concatenate([num,catx,miss,avail,missall],axis=1).astype(np.float32,copy=False)
 if not np.isfinite(X).all(): raise FloatingPointError("Non-finite baseline test features")
 return X
def eval_baselines(ds,artifact,split,baseline_results):
 arr=artifact[split]; raw=np.asarray(arr["labels"]).astype(str)
 meta=baseline_results["datasets"][ds]["feature_meta"]
 X=baseline_matrix(arr,meta)
 out={}
 for mode in ("native","unified"):
  info=baseline_results["datasets"][ds]["label_modes"][mode]
  classes=list(info["classes"]); c2i={x:i for i,x in enumerate(classes)}
  if mode=="native": names=raw
  else: names=np.asarray([r.taxonomy.classes[i] for i in r.taxonomy.map_labels(ds,raw)],dtype=object)
  y=np.asarray([c2i.get(str(x),-1) for x in names],dtype=np.int64)
  valid=y>=0
  ent={}
  hp=BASE/f"{ds}_{mode}_hgb_seed42.joblib"
  hm=joblib.load(hp); ph=hm.predict(X[valid])
  ent["HGB_seed42"]=basic_metrics(y[valid],ph,classes)
  mp=BASE/f"{ds}_{mode}_mlp_seed{MLP_SEEDS[ds][mode]}.pt"
  ck=torch.load(mp,map_location="cpu",weights_only=False)
  model=b.PlainMLP(int(ck["input_dim"]),len(ck["classes"]))
  model.load_state_dict(ck["state_dict"],strict=True); model.eval()
  with torch.no_grad(): pm=model(torch.as_tensor(X[valid],dtype=torch.float32)).argmax(1).numpy()
  ent[f"MLP_seed{MLP_SEEDS[ds][mode]}"]=basic_metrics(y[valid],pm,classes)
  ent["unsupported_test_labels"]=int((~valid).sum())
  ent["unsupported_label_names"]=sorted(set(names[~valid].tolist()))
  out[mode]=ent
 return out
def preflight():
 m=verify_manifest()
 helper,branches,labels,aug_cfg,level,confidence,group_only,level3_mask=setup_source()
 cal=json.loads(CAL.read_text(encoding="utf-8"))
 baseline_results=json.loads((BASE/"results.json").read_text(encoding="utf-8"))
 checks=[]
 for spec in VARIANTS:
  v=init_variant(spec,helper,branches,labels,aug_cfg,level,confidence,group_only,level3_mask,False)
  ds="nsl_kdd"; ix=np.arange(2)
  lo,z,g=forward_rows(v,helper,branches,ds,"validation",ix,labels,aug_cfg,level,confidence,group_only,level3_mask)
  if not torch.isfinite(lo).all() or not torch.isfinite(z).all(): raise FloatingPointError(f"Preflight non-finite {spec[0]}")
  checks.append(spec[0]); del v; gc.collect(); torch.cuda.empty_cache()
 # Validate baseline feature transform and selected saved model dimensions on validation only.
 for ds in r.DATASETS:
  obj=joblib.load(ROOT/"data/interim/stage6_trainval_only"/f"{ds}_inputs.joblib")
  X=baseline_matrix(obj["validation"],baseline_results["datasets"][ds]["feature_meta"])
  for mode in ("native","unified"):
   for typ in ("hgb","mlp"):
    if typ=="hgb":
     model=joblib.load(BASE/f"{ds}_{mode}_hgb_seed42.joblib"); model.predict(X[:2])
    else:
     ck=torch.load(BASE/f"{ds}_{mode}_mlp_seed{MLP_SEEDS[ds][mode]}.pt",map_location="cpu",weights_only=False)
     model=b.PlainMLP(int(ck["input_dim"]),len(ck["classes"])); model.load_state_dict(ck["state_dict"],strict=True)
     with torch.no_grad(): model(torch.as_tensor(X[:2],dtype=torch.float32))
  del obj,X; gc.collect()
 if not np.isfinite([cal["temperature"],cal["tau_ood_validation_p95"],cal["tau_conf_validation_p10"]]).all(): raise RuntimeError("Calibration values non-finite")
 log("PREFLIGHT_COMPLETE no_test_arrays_or_labels_loaded variants="+",".join(checks))
def execute():
 verify_manifest()
 if (RUN/"results.json").exists(): raise RuntimeError("Refusing to rerun Phase C; results.json already exists")
 if not torch.cuda.is_available(): raise RuntimeError("HARD STOP CUDA unavailable")
 helper,branches,labels,aug_cfg,level,confidence,group_only,level3_mask=setup_source()
 cal=json.loads(CAL.read_text(encoding="utf-8"))
 baseline_results=json.loads((BASE/"results.json").read_text(encoding="utf-8"))
 results={"phase":"C","one_shot":True,"freeze_manifest":str(MANIFEST),"freeze_manifest_sha256":sha(MANIFEST),
  "primary_checkpoint_sha256":sha(PRIMARY),"calibration_sha256":sha(CAL),
  "stage5_sha256":STAGE5_SHA,"stage6_sha256":STAGE6_SHA,"datasets":{}}
 # Primary OOD fit uses the same seeded 20k source-training sample and train-fitted score normalizers.
 models=[]
 for spec in VARIANTS:
  models.append(init_variant(spec,helper,branches,labels,aug_cfg,level,confidence,group_only,level3_mask,True))
 primary=models[0]
 ood=fit_primary_ood(primary,helper,branches,labels,aug_cfg,level,confidence,group_only,level3_mask,cal)
 # Memory lifecycle: source training/validation tensors and label arrays are no longer
 # needed after model construction and primary OOD fitting. Release them before opening
 # any test partition; preserve the frozen Stage5 encoder and Stage6 branches.
 for _ds in r.DATASETS:
  helper.datasets[_ds].clear()
 helper.datasets={_ds:{} for _ds in r.DATASETS}
 labels={_ds:{"native_train":np.unique(labels[_ds]["native_train"]),"native_classes":labels[_ds]["native_classes"]} for _ds in r.DATASETS}  # Keep only compact label vocabularies needed for native-head aggregation and NSL test-only analysis.
 gc.collect()
 if torch.cuda.is_available(): torch.cuda.empty_cache()
 log(f"MEMORY_RELEASED source_trainval_and_labels free_ram_gb={r.free_ram_gb():.2f} gpu_allocated_mib={torch.cuda.memory_allocated()/1024**2:.1f}")
 if r.free_ram_gb()<2.0 or shutil.disk_usage("C:\\").free<10*1024**3 or torch.cuda.max_memory_allocated()>6*1024**3:
  raise RuntimeError("HARD STOP resource threshold reached before test opening")
 log("PHASEC_ONE_SHOT_EVALUATION_STARTED frozen_hashes_verified; opening source-test artifacts exactly once")
 with (ROOT/"PROGRESS.md").open("a",encoding="utf-8") as f:
  f.write("\\n- 2026-11-26 Phase C actual one-shot evaluation STARTED: supplemental freeze manifest verified immediately before test-artifact loading; no test arrays/labels previously loaded by this evaluator. No rerun permitted.\\n")

 for ds in r.DATASETS:
  art=joblib.load(DATA_ART[ds])
  expected={"nsl_kdd":(22544,21880),"unsw_nb15":(82332,73791)}[ds]
  if len(art["test_full"]["labels"])!=expected[0] or len(art["test_independent"]["labels"])!=expected[1]:
   raise RuntimeError(f"HARD STOP {ds} test row-count mismatch")
  results["datasets"][ds]={}
  for split in ("test_independent","test_full"):
   results["datasets"][ds][split]={}
   for vi,v in enumerate(models):
    results["datasets"][ds][split][v["name"]]=eval_transformer(v,helper,branches,art,ds,split,labels,aug_cfg,level,confidence,group_only,level3_mask,ood if vi==0 else None,cal if vi==0 else None)
   results["datasets"][ds][split]["baselines"]=eval_baselines(ds,art,split,baseline_results)
   tmp=RUN/"results.partial.json"; tmp.write_text(json.dumps(results,indent=2,allow_nan=False),encoding="utf-8")
   log(f"DATASET_SPLIT_COMPLETE {ds} {split}")
  del art
  helper.datasets.pop(ds,None)
  gc.collect()
  log(f"DATASET_MEMORY_RELEASED dataset={ds} free_ram_gb={r.free_ram_gb():.2f}")
 (RUN/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False),encoding="utf-8")
 log("PHASEC_COMPLETE results.json written; source-test one-shot consumed")
 with (ROOT/"PROGRESS.md").open("a",encoding="utf-8") as f:
  f.write("\n- 2026-11-26 Phase C one-shot source-test evaluation COMPLETE. Full and training-independent partitions evaluated for primary, validation-selected ablations, HGB and MLP baselines. See C:\\NOVEL_IDS_runs\\phaseC_one_shot_20261126\\results.json. No rerun performed.\n")
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--preflight",action="store_true")
 ap.add_argument("--execute",action="store_true")
 a=ap.parse_args()
 if a.preflight==a.execute: raise SystemExit("Specify exactly one of --preflight or --execute")
 if a.preflight: preflight()
 else: execute()
if __name__=="__main__": main()
