from __future__ import annotations
import gc,json,sys,time,hashlib
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(ROOT/"src"))
import run_stage7_11_experiment as r
from novel_ids.fusion.stage7 import AdaptiveResidualFusion
from novel_ids.alignment.stage8 import Stage8SelectiveAlignment,NSL_KDD,UNSW_NB15,UNSEEN
from novel_ids.fusion.stage9 import UnifiedRepresentation
from novel_ids.transformer.stage10 import Stage10Transformer
from novel_ids.classification.stage11 import Stage11Classifier
from novel_ids.taxonomy.stage11 import load_taxonomy
from novel_ids.residual.stage6_training import Stage6TrainingConfig
from novel_ids.residual.stage6 import Stage6ResidualBranch
OUT=Path(r"C:\NOVEL_IDS_runs\phaseD_lodo_20261126")
INPUT=ROOT/"data/interim/stage6_trainval_only"
STAGE5=Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
STAGE6=Path(r"C:\NOVEL_IDS_runs\stage6\accepted_candidate09_lambda1_seed42\best.pt")
STAGE5_SHA="9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"
STAGE6_SHA="1c66f6f8d1a029266e5a637b5db4d6f49881988bf16462d2c3fbc09cd704ecbd"
def log(s):
 line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"; print(line,flush=True)
 with (OUT/"progress.log").open("a",encoding="utf-8") as f:f.write(line+"\n")
def metrics(y,pred,classes):
 cm=confusion_matrix(y,pred,labels=np.arange(len(classes)))
 pr,re,f1,s=precision_recall_fscore_support(y,pred,labels=np.arange(len(classes)),zero_division=0)
 present=s>0
 return {"rows":int(len(y)),"accuracy":float(accuracy_score(y,pred)),
  "macro_f1_supported_classes":float(np.mean(f1[present])) if present.any() else None,
  "weighted_f1":float(f1_score(y,pred,labels=np.arange(len(classes)),average="weighted",zero_division=0)),
  "per_class":{classes[i]:{"precision":float(pr[i]),"recall":float(re[i]),"f1":float(f1[i]),"support":int(s[i])} for i in range(len(classes))},
  "confusion_matrix":cm.tolist(),"normalized_confusion_matrix":np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(1,keepdims=True)>0).tolist()}
def run_direction(source,target,tax,helper,branches,aug_cfg,level,confidence,group_only,level3_mask):
 global_start=time.time()
 source_id=NSL_KDD if source=="nsl_kdd" else UNSW_NB15
 source_labels_obj=joblib.load(INPUT/f"{source}_labels.joblib")
 ytr_raw=np.asarray(source_labels_obj["train"]["labels"]).astype(str)
 yva_raw=np.asarray(source_labels_obj["validation"]["labels"]).astype(str)
 global_tr=np.asarray(tax.map_labels(source,ytr_raw),dtype=np.int64)
 global_va=np.asarray(tax.map_labels(source,yva_raw),dtype=np.int64)
 shared=[c for c in tax.classes if c in set(tax.mappings["nsl_kdd"].values()) and c in set(tax.mappings["unsw_nb15"].values())]
 local={tax.classes.index(c):i for i,c in enumerate(shared)}
 ytr=np.asarray([local.get(int(x),-1) for x in global_tr]); yva=np.asarray([local.get(int(x),-1) for x in global_va])
 train_ix=np.flatnonzero(ytr>=0); val_ix=np.flatnonzero(yva>=0)
 if len(train_ix)<256 or len(val_ix)==0: raise RuntimeError(f"{source}: insufficient shared-class rows")
 counts=np.bincount(ytr[train_ix],minlength=len(shared)).astype(np.float64)
 if np.any(counts==0): raise RuntimeError(f"{source}: missing shared class in training")
 w=np.power(counts,-0.5); w=w/w.mean(); w=np.maximum(w,w.max()/15.0); w=w/w.mean()
 torch.manual_seed(43); np.random.seed(43); torch.cuda.manual_seed_all(43)
 stage7=AdaptiveResidualFusion().to(r.DEVICE)
 stage8=Stage8SelectiveAlignment(coral_lambda=0.0,mode="compatibility").to(r.DEVICE)
 stage9=UnifiedRepresentation()
 stage10=Stage10Transformer().to(r.DEVICE)
 classifier=Stage11Classifier(len(shared),class_weights=torch.as_tensor(w,dtype=torch.float32,device=r.DEVICE)).to(r.DEVICE)
 # Single-source pseudo-unseen reference: duplicate the source-only Z_f reference into both slots.
 # This is a diagnostic adaptation for Section 11.5, not the final two-source Stage 8 fit.
 zref=[]; stage7.eval()
 for st in range(0,len(helper.datasets[source]["train"]["numeric_values"]),512):
  ix=np.arange(st,min(st+512,len(helper.datasets[source]["train"]["numeric_values"])))
  bch=r.source_batch(helper,branches,source,"train",ix,False,torch.Generator(device=r.DEVICE).manual_seed(430),level3_mask,aug_cfg)
  with torch.no_grad(): zref.append(stage7(bch["Z_s"],bch["R_proj"])["Z_f"].float())
 zref=torch.cat(zref,0)
 stage8.compatibility.fit_source_training_references(zref,zref)
 del zref; gc.collect()
 trainable=list(stage7.parameters())+list(stage8.W_a.parameters())+list(stage8.W_u.parameters())+list(stage10.parameters())+list(classifier.parameters())
 opt=torch.optim.AdamW(trainable,lr=5e-4,weight_decay=1e-4)
 scaler=torch.amp.GradScaler("cuda")
 rng=torch.Generator(device=r.DEVICE); rng.manual_seed(43043)
 rowrng=np.random.default_rng(43043); steps=1500; batch=256; interval=100
 best=-1; beststep=0; stale=0; history=[]; started=time.time()
 r.ABLATION_MODE="full"; r.HEAD_MODE="unified"; r.COMMON_MASK=(level==1)
 order=rowrng.permutation(train_ix); cursor=0
 for step in range(1,steps+1):
  if r.free_ram_gb()<2: raise RuntimeError("HARD STOP free RAM below 2 GiB during LODO")
  if torch.cuda.max_memory_allocated()>6*1024**3: raise RuntimeError("HARD STOP GPU allocation above 6 GiB during LODO")
  if cursor+batch>len(order): order=rowrng.permutation(train_ix); cursor=0
  ix=order[cursor:cursor+batch]; cursor+=batch
  bch=r.source_batch(helper,branches,source,"train",ix,True,rng,level3_mask,aug_cfg)
  sid=torch.full((len(ix),),source_id,device=r.DEVICE,dtype=torch.long)
  yy=torch.as_tensor(ytr[ix],device=r.DEVICE,dtype=torch.long)
  opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type="cuda",dtype=torch.float16):
   logits,g,a,f=r.batch_forward(bch,stage7,stage8,stage9,stage10,classifier,sid,level,confidence,group_only,1.0)
   loss=classifier.loss(logits.float(),yy)
  if not torch.isfinite(loss): raise FloatingPointError(f"{source}: non-finite LODO loss at {step}")
  scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trainable,1.0); scaler.step(opt); scaler.update()
  if step%interval==0 or step==steps:
   stage7.eval();stage8.eval();stage10.eval();classifier.eval()
   pp=[]
   with torch.no_grad():
    for st in range(0,len(val_ix),512):
     vi=val_ix[st:st+512]; vb=r.source_batch(helper,branches,source,"validation",vi,False,rng,level3_mask,aug_cfg)
     vsid=torch.full((len(vi),),source_id,device=r.DEVICE,dtype=torch.long)
     with torch.autocast(device_type="cuda",dtype=torch.float16):
      vl,_,_,_=r.batch_forward(vb,stage7,stage8,stage9,stage10,classifier,vsid,level,confidence,group_only,1.0)
     pp.extend(vl.float().argmax(1).cpu().numpy().tolist())
   score=float(f1_score(yva[val_ix],np.asarray(pp),labels=np.arange(len(shared)),average="macro",zero_division=0))
   history.append({"step":step,"source_validation_macro_f1":score})
   if score>best+1e-5:
    best=score;beststep=step;stale=0
    torch.save({"stage7":stage7.state_dict(),"stage8":stage8.state_dict(),"stage10":stage10.state_dict(),"classifier":classifier.state_dict(),
      "source":source,"target":target,"shared_classes":shared,"step":step,"seed":43,
      "stage5_sha256":STAGE5_SHA,"stage6_sha256":STAGE6_SHA},OUT/f"{source}_to_{target}_best.pt")
   else: stale+=1
   log(f"LODO_TRAIN source={source} target={target} step={step}/{steps} val_macro_f1={score:.6f} best={best:.6f} stale={stale} ram_gb={r.free_ram_gb():.2f} gpu_peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}")
   if stale>=5: break
   stage7.train();stage8.train();stage10.train();classifier.train()
 # Load best checkpoint, then open target validation labels only for evaluation.
 ck=torch.load(OUT/f"{source}_to_{target}_best.pt",map_location=r.DEVICE,weights_only=False)
 stage7.load_state_dict(ck["stage7"]);stage8.load_state_dict(ck["stage8"]);stage10.load_state_dict(ck["stage10"]);classifier.load_state_dict(ck["classifier"])
 del ck
 target_label_obj=joblib.load(INPUT/f"{target}_labels.joblib")
 ytarget_raw=np.asarray(target_label_obj["validation"]["labels"]).astype(str)
 global_target=np.asarray(tax.map_labels(target,ytarget_raw),dtype=np.int64)
 ytarget=np.asarray([local.get(int(x),-1) for x in global_target])
 target_ix=np.flatnonzero(ytarget>=0)
 if len(target_ix)==0: raise RuntimeError(f"{target}: no target validation rows in shared classes")
 stage7.eval();stage8.eval();stage10.eval();classifier.eval()
 preds=[]
 with torch.no_grad():
  for st in range(0,len(target_ix),512):
   ix=target_ix[st:st+512]
   tb=r.source_batch(helper,branches,target,"validation",ix,False,rng,level3_mask,aug_cfg)
   tsid=torch.full((len(ix),),UNSEEN,device=r.DEVICE,dtype=torch.long)
   with torch.autocast(device_type="cuda",dtype=torch.float16):
    tl,_,_,_=r.batch_forward(tb,stage7,stage8,stage9,stage10,classifier,tsid,level,confidence,group_only,1.0)
   preds.extend(tl.float().argmax(1).cpu().numpy().tolist())
 pred=np.asarray(preds,dtype=np.int64)
 result={"source":source,"target":target,"shared_classes":shared,
  "protocol":"train source only; target validation labels loaded only after source-only training; target rows restricted to shared approved taxonomy classes; Stage8 pseudo-unseen reference duplicates source-only training Z_f into both reference slots; alignment architecture retained; no source test/unseen data accessed",
  "source_train_rows_shared":int(len(train_ix)),"source_validation_rows_shared":int(len(val_ix)),
  "target_validation_rows_shared":int(len(target_ix)),"target_validation_rows_unsupported":int((ytarget<0).sum()),
  "best_source_validation_macro_f1":best,"best_step":beststep,"training_history":history,
  "target_validation_metrics":metrics(ytarget[target_ix],pred,shared),
  "elapsed_seconds":time.time()-global_start}
 log(f"LODO_COMPLETE source={source} target={target} target_macro_f1={result['target_validation_metrics']['macro_f1_supported_classes']:.6f} elapsed_s={result['elapsed_seconds']:.1f}")
 return result
def main():
 if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
 OUT.mkdir(parents=True)
 if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
 r.verify_protected()
 if r.digest(STAGE5)!=STAGE5_SHA or r.digest(STAGE6)!=STAGE6_SHA: raise RuntimeError("HARD STOP frozen Stage5/6 hash mismatch")
 r.RUN_DIR=OUT/("loader_"+str(__import__("os").getpid()));r.RUN_DIR.mkdir(parents=True)
 r.DEVICE=torch.device("cuda");r.taxonomy=load_taxonomy(ROOT/"config/taxonomy_unified.yaml",purpose="evaluation")
 helper,branches=r.read_data()
 ck5=torch.load(STAGE5,map_location="cpu",weights_only=False);aug_cfg=ck5["config"];del ck5
 level,confidence,group_only,level3_mask=r.make_metadata()
 results={"phase":"D_LODO","unseen_directory_exists":(ROOT/"data/unseen").exists(),"cic_ddos2019":"not supplied; unseen evaluation pending"}
 results["directions"]=[]
 results["directions"].append(run_direction("nsl_kdd","unsw_nb15",r.taxonomy,helper,branches,aug_cfg,level,confidence,group_only,level3_mask))
 results["directions"].append(run_direction("unsw_nb15","nsl_kdd",r.taxonomy,helper,branches,aug_cfg,level,confidence,group_only,level3_mask))
 (OUT/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False),encoding="utf-8")
 with (ROOT/"PROGRESS.md").open("a",encoding="utf-8") as f:
  f.write("\n- 2026-11-26 Phase D LODO diagnostics complete on source-train to opposite-source-validation only; see C:\\NOVEL_IDS_runs\\phaseD_lodo_20261126\\results.json. CIC-DDoS2019 absent; unseen evaluation pending. Diagnostic uses single-source Stage8 reference duplicated in both reference slots; explicitly not a production unseen evaluation.\n")
 log("PHASE_D_LODO_COMPLETE")
if __name__=="__main__":main()
