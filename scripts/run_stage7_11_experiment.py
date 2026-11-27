from __future__ import annotations
import argparse, gc, hashlib, json, math, os, random, time, traceback
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from novel_ids.residual.stage6_training import Stage6Trainer, Stage6TrainingConfig, STAGE5_PROTECTED_HASHES, sha256_file
from novel_ids.residual.stage6 import Stage6ResidualBranch
from novel_ids.fusion.stage7 import AdaptiveResidualFusion
from novel_ids.alignment.stage8 import Stage8SelectiveAlignment, NSL_KDD, UNSW_NB15
from novel_ids.fusion.stage9 import UnifiedRepresentation
from novel_ids.transformer.stage10 import Stage10Transformer
from novel_ids.classification.stage11 import Stage11Classifier, NativeAblationHeads, compute_class_weights
from novel_ids.taxonomy.stage11 import load_taxonomy

ROOT=Path(__file__).resolve().parents[1]
STAGE5=Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
STAGE6=Path(r"C:\NOVEL_IDS_runs\stage6\accepted_candidate09_lambda1_seed42\best.pt")
STAGE6_SHA="1c66f6f8d1a029266e5a637b5db4d6f49881988bf16462d2c3fbc09cd704ecbd"
STAGE5_SHA="9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"
TAXONOMY=ROOT/"config"/"taxonomy_unified.yaml"
INPUT=ROOT/"data"/"interim"/"stage6_trainval_only"
DATASETS=("nsl_kdd","unsw_nb15")
BATCH_PER_DATASET=256
VAL_BATCH=512
CONFIDENCE_VALUE={"high":1.0,"medium":0.8,"low":0.5,"unassessed":0.0}
ABLATION_MODE="full"
HEAD_MODE="unified"
NATIVE_CLASSES={}
NATIVE_WEIGHTS={}
COMMON_MASK=None
CACHE_ROOT=None
CACHE_ARRAYS={}
CACHE_MANIFEST_SHA=None
CACHE_SKIP_TOKENS=False

def digest(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def log(path:Path,msg:str):
    line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line,flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:f.write(line+"\n")

def atomic_json(path:Path,obj:Any):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str),encoding="utf-8")
    os.replace(tmp,path)

def atomic_torch(path:Path,obj:Any):
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp)
    os.replace(tmp,path)

def verify_protected():
    for rel,expected in STAGE5_PROTECTED_HASHES.items():
        actual=sha256_file(ROOT/rel)
        if actual.lower()!=expected.lower():
            raise RuntimeError(f"HARD STOP protected Stage5 hash mismatch: {rel}: {actual}")
    if digest(STAGE5)!=STAGE5_SHA: raise RuntimeError("HARD STOP Stage5 snapshot hash mismatch")
    if digest(STAGE6)!=STAGE6_SHA: raise RuntimeError("Frozen Stage6 checkpoint hash mismatch")

def free_ram_gb():
    import ctypes
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
          ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
          ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
          ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
          ("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
    m=MEMORYSTATUSEX(); m.dwLength=ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError("GlobalMemoryStatusEx failed")
    return m.ullAvailPhys/1024**3

def read_data():
    cfg=Stage6TrainingConfig(residual_dim=128,lambda_red=1.0,variance_floor=1e-4,
        projection_mode="identity",max_steps=1,validation_interval=1,
        early_stop_patience=1,device=DEVICE)
    # Stage6Trainer loader is used only to construct the frozen Stage5 model and
    # load train/validation feature inputs. Its Stage6 parameters are replaced
    # by the frozen, hash-verified accepted Stage6 checkpoint below.
    helper_dir=RUN_DIR/"frozen_input_loader"
    helper=Stage6Trainer.from_project(stage5_snapshot=STAGE5,
        expected_stage5_sha256=STAGE5_SHA,run_dir=helper_dir,config=cfg)
    ck=torch.load(STAGE6,map_location="cpu",weights_only=False)
    if ck.get("stage5_checkpoint_sha256")!=STAGE5_SHA: raise RuntimeError("Stage6 checkpoint provenance mismatch")
    branches=torch.nn.ModuleDict({d:Stage6ResidualBranch(128,128,128,128,128,128) for d in DATASETS})
    branches.load_state_dict(ck["branches"],strict=True)
    branches.to(DEVICE).eval()
    for p in branches.parameters():p.requires_grad_(False)
    helper.stage5.eval()
    for p in helper.stage5.parameters():p.requires_grad_(False)
    del ck
    # Release the helper's freshly initialized random Stage6 branches and optimizer;
    # only the accepted frozen branch checkpoint is retained.
    del helper.branches, helper.optimizer
    torch.cuda.empty_cache()
    return helper,branches

def memmap_dataset_inputs(helper,run_dir,log_path):
    """Move frozen train/validation feature arrays to read-only joblib memmaps to respect RAM gate."""
    for ds in DATASETS:
        for split in ("train","validation"):
            path=run_dir/f"{ds}_{split}_input_memmap.joblib"
            joblib.dump(helper.datasets[ds][split],path,compress=0)
            mapped=joblib.load(path,mmap_mode="r")
            helper.datasets[ds][split]=mapped
            del mapped
            gc.collect()
            log(log_path,f"INPUT_MEMMAP_READY {ds} {split} path={path.name} free_ram_gb={free_ram_gb():.2f}")
    gc.collect()

def load_labels(taxonomy):
    result={}
    for ds in DATASETS:
        obj=joblib.load(INPUT/f"{ds}_labels.joblib")
        ytr=np.asarray(obj["train"]["labels"]).astype(str)
        yva=np.asarray(obj["validation"]["labels"]).astype(str)
        native_classes=tuple(sorted(np.unique(ytr).tolist()))
        native_index={name:i for i,name in enumerate(native_classes)}
        result[ds]={"train":np.asarray(taxonomy.map_labels(ds,ytr),dtype=np.int64),
                    "validation":np.asarray(taxonomy.map_labels(ds,yva),dtype=np.int64),
                    "native_train":ytr,"native_validation":yva,
                    "native_classes":native_classes,
                    "native_train_idx":np.asarray([native_index[x] for x in ytr],dtype=np.int64),
                    "native_validation_idx":np.asarray([native_index.get(x,-1) for x in yva],dtype=np.int64)}
        if len(result[ds]["train"])!=len(helper.datasets[ds]["train"]["numeric_values"]):
            raise RuntimeError(f"{ds}: train feature/label row mismatch")
        if len(result[ds]["validation"])!=len(helper.datasets[ds]["validation"]["numeric_values"]):
            raise RuntimeError(f"{ds}: validation feature/label row mismatch")
    return result

def make_metadata():
    schema=yaml.safe_load((ROOT/"data"/"processed"/"unified_representation"/"unified_feature_schema.yaml").read_text(encoding="utf-8"))
    features=schema["features"]
    if len(features)!=78: raise RuntimeError(f"Expected 78 feature slots, got {len(features)}")
    levels=np.asarray([int(x["level"]) for x in features],dtype=np.int64)
    conf=np.asarray([CONFIDENCE_VALUE.get(str(x.get("confidence","unassessed")).lower(),0.0) for x in features],dtype=np.float32)
    group=np.asarray([str(x.get("relationship","")) in {"group_only","partial"} for x in features],dtype=bool)
    return (torch.as_tensor(levels,device=DEVICE,dtype=torch.long),
            torch.as_tensor(conf,device=DEVICE,dtype=torch.float32),
            torch.as_tensor(group,device=DEVICE,dtype=torch.bool),
            torch.as_tensor(levels==3,device=DEVICE,dtype=torch.bool))

def trim_working_set():
    if CACHE_ROOT is None: return
    try:
        import ctypes
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
    except Exception:
        pass

def source_batch(helper,branches,ds,split,indices,augment,rng,level3_mask,aug_cfg):
    idx=np.asarray(indices,dtype=np.int64)
    if CACHE_ROOT is not None:
        mode="common_only" if ABLATION_MODE=="common_only" else "full"
        cached=CACHE_ARRAYS[ds][split][mode]
        packed=np.asarray(cached["zszd"][idx],dtype=np.float32)
        zs=torch.as_tensor(packed[:,:128].copy(),device=DEVICE,dtype=torch.float32)
        zd=torch.as_tensor(packed[:,128:].copy(),device=DEVICE,dtype=torch.float32)
        residual=branches[ds](zs,zd)
        if CACHE_SKIP_TOKENS:
            tokens=None
        else:
            token_map=np.load(cached["tokens_path"],mmap_mode="r")
            token_np=np.asarray(token_map[idx],dtype=np.float32).copy()
            token_map._mmap.close()
            tokens=torch.as_tensor(token_np,device=DEVICE,dtype=torch.float32)
            del token_np
        available=torch.as_tensor(np.asarray(cached["available"][idx]).copy(),device=DEVICE,dtype=torch.bool)
        return {"Z_s":zs,"Z_d":zd,"R":residual["R"],"R_proj":residual["R_proj"],"tokens":tokens,"available":available}
    arr=helper.datasets[ds][split]
    numeric=torch.as_tensor(arr["numeric_values"][idx],device=DEVICE,dtype=torch.float32)
    categorical=torch.as_tensor(arr["categorical_values"][idx],device=DEVICE,dtype=torch.long)
    avail=torch.as_tensor(arr["M_avail"][idx],device=DEVICE,dtype=torch.float32).clone()
    miss=torch.as_tensor(arr["M_miss"][idx],device=DEVICE,dtype=torch.float32).clone()
    if ABLATION_MODE=="common_only":
        cm=COMMON_MASK.unsqueeze(0).to(device=DEVICE,dtype=avail.dtype)
        avail=avail*cm; miss=miss*cm
    num_mask=torch.as_tensor(arr["numeric_token_mask"],device=DEVICE,dtype=torch.bool)
    cat_mask=torch.as_tensor(arr["categorical_token_mask"],device=DEVICE,dtype=torch.bool)
    if augment:
        # Frozen Stage5 remains in eval/no_grad. The approved training-only
        # feature-set/value masking is applied to its input masks, not weights.
        rate=float(aug_cfg.get("feature_set_dropout_rate",0.1))
        all_prob=float(aug_cfg.get("feature_set_dropout_all_probability",0.05))
        value_rate=float(aug_cfg.get("value_masking_rate",0.05))
        candidates=(avail>0)&level3_mask.unsqueeze(0)
        drop=(torch.rand(avail.shape,device=DEVICE,generator=rng)<rate)&candidates
        drop_all=torch.rand((len(idx),1),device=DEVICE,generator=rng)<all_prob
        drop=torch.where(drop_all,candidates,drop)
        avail=torch.where(drop,torch.zeros_like(avail),avail)
        miss=torch.where(drop,torch.zeros_like(miss),miss)
        eligible=(avail>0)&(miss==0)&(num_mask|cat_mask).unsqueeze(0)
        value_mask=(torch.rand(miss.shape,device=DEVICE,generator=rng)<value_rate)&eligible
        miss=torch.where(value_mask,torch.ones_like(miss),miss)
    # Keep frozen Stage5/6 inference in float32: extreme but finite input
    # magnitudes can overflow fp16 before token validation. AMP is applied to
    # trainable Stage7-11 operations downstream.
    with torch.no_grad():
        out=helper.stage5(numeric_values=numeric,categorical_values=categorical,
            M_avail=avail,M_miss=miss,numeric_token_mask=num_mask,
            categorical_token_mask=cat_mask,dataset=ds,feature_descriptors=None)
        s6=branches[ds](out["Z_s"],out["Z_d"])
    return {"Z_s":out["Z_s"].detach(),"Z_d":out["Z_d"].detach(),
            "R":s6["R"].detach(),"R_proj":s6["R_proj"].detach(),
            "tokens":out["token_embeddings"].detach(),
            "available":out["M_avail"].bool().detach()}

def batch_forward(batch,stage7,stage8,stage9,stage10,classifier,source_ids,
                  level,confidence,group_only,warmup):
    zs=batch["Z_s"]; zd=batch["Z_d"]; r=batch["R"]; rp=batch["R_proj"]
    gate_out=stage7(zs,rp)
    stage8.warmup_fraction=float(warmup)
    align=stage8(gate_out["Z_f"],zd,source_ids,batch["available"],
                 level.expand(len(zs),-1),confidence.expand(len(zs),-1),
                 group_only.expand(len(zs),-1))
    gate=gate_out["G"]; zf=gate_out["Z_f"]; a=align["A"]; u=align["U"]; c=align["C"]
    if ABLATION_MODE in {"shared_only","common_only"}:
        gate=torch.zeros_like(gate); zf=zs; a=torch.zeros_like(a); u=torch.zeros_like(u); c=torch.zeros_like(c)
    elif ABLATION_MODE=="residual_fixed_gate1":
        gate=torch.ones_like(gate); zf=zs+rp; a=torch.zeros_like(a); u=torch.zeros_like(u); c=torch.zeros_like(c)
    elif ABLATION_MODE in {"residual_adaptive","alignment_off"}:
        a=torch.zeros_like(a); u=torch.zeros_like(u); c=torch.zeros_like(c)
    gate_out={**gate_out,"G":gate}
    fused=stage9(z_s=zs,r=r,r_proj=rp,gate=gate,a=a,u=u,c=c,z_f=zf)
    align={**align,"A":a,"U":u,"C":c}
    transformed=stage10(batch["tokens"],fused["Z_u"],batch["available"])
    if HEAD_MODE=="native":
        nsl_mask=source_ids==NSL_KDD; unsw_mask=source_ids==UNSW_NB15
        logits={"nsl_kdd":classifier(transformed["Z_t"][nsl_mask],"nsl_kdd"),
                "unsw_nb15":classifier(transformed["Z_t"][unsw_mask],"unsw_nb15")}
    else:
        logits=classifier(transformed["Z_t"])
    return logits,gate_out,align,fused

def fit_references(helper,branches,stage7,stage8,level3_mask,aug_cfg,log_path):
    global CACHE_SKIP_TOKENS
    CACHE_SKIP_TOKENS=True
    stage7.eval(); zs={}
    start=time.perf_counter()
    for ds in DATASETS:
        pieces=[]; n=len(helper.datasets[ds]["train"]["numeric_values"])
        rng=torch.Generator(device=DEVICE); rng.manual_seed(20261125+(0 if ds=="nsl_kdd" else 1))
        for st in range(0,n,VAL_BATCH):
            idx=np.arange(st,min(n,st+VAL_BATCH))
            b=source_batch(helper,branches,ds,"train",idx,False,rng,level3_mask,aug_cfg)
            with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.float16):
                g=stage7(b["Z_s"],b["R_proj"])
            pieces.append(g["Z_f"].float().cpu())
            if (st//VAL_BATCH)%20==0:
                trim_working_set()
                log(log_path,f"REFERENCE_FIT {ds} rows={min(n,st+VAL_BATCH)}/{n} elapsed_s={time.perf_counter()-start:.1f} free_ram_gb={free_ram_gb():.2f} gpu_peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}")
        zs[ds]=torch.cat(pieces,dim=0)
    stage8.compatibility.fit_source_training_references(zs["nsl_kdd"],zs["unsw_nb15"])
    CACHE_SKIP_TOKENS=False
    trim_working_set()
    stage7.train()
    log(log_path,f"REFERENCE_FIT_COMPLETE seconds={time.perf_counter()-start:.1f} tau={float(stage8.compatibility.tau.item()):.7f} source_train_only=True")
    del zs; gc.collect()

def evaluate(helper,branches,stage7,stage8,stage9,stage10,classifier,labels,
             level,confidence,group_only,level3_mask,aug_cfg,log_path,seed):
    metrics={}; preds={}; gate_means={}; gate_sat={}
    for di,ds in enumerate(DATASETS):
        stage7.eval(); stage8.eval(); stage10.eval(); classifier.eval()
        n=len(helper.datasets[ds]["validation"]["numeric_values"])
        pr=[]
        rng=torch.Generator(device=DEVICE); rng.manual_seed(seed+800+di)
        for st in range(0,n,VAL_BATCH):
            idx=np.arange(st,min(n,st+VAL_BATCH))
            b=source_batch(helper,branches,ds,"validation",idx,False,rng,level3_mask,aug_cfg)
            sid=torch.full((len(idx),),NSL_KDD if ds=="nsl_kdd" else UNSW_NB15,device=DEVICE,dtype=torch.long)
            with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.float16):
                logits,g,a,f=batch_forward(b,stage7,stage8,stage9,stage10,classifier,sid,
                    level,confidence,group_only,1.0)
            if HEAD_MODE=="native":
                pr.append(logits[ds].float().argmax(1).cpu().numpy())
            else:
                pr.append(logits.float().argmax(1).cpu().numpy())
            if CACHE_ROOT is not None and (st//VAL_BATCH)%10==0: trim_working_set()
        pred=np.concatenate(pr)
        if HEAD_MODE=="native":
            y=labels[ds]["native_validation_idx"]
            valid=y>=0; y=y[valid]; pred=pred[valid]
            class_names=labels[ds]["native_classes"]
        else:
            y=labels[ds]["validation"]
            class_names=taxonomy.classes
        present=np.unique(y)
        p,r,f1,s=precision_recall_fscore_support(y,pred,labels=present,zero_division=0)
        metrics[ds]={
            "accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,labels=present,average="macro",zero_division=0)),
            "weighted_f1":float(f1_score(y,pred,labels=present,average="weighted",zero_division=0)),
            "per_class":{class_names[int(c)]:{"precision":float(p[i]),"recall":float(r[i]),"f1":float(f1[i]),"support":int(s[i])} for i,c in enumerate(present)},
            "low_support_classes":[class_names[int(c)] for c in present if int(np.sum(y==c))<20],
            "validation_rows":int(len(y)),
        }
        preds[ds]=pred
    # Fixed 256-row validation batches per dataset for a comparable raw CORAL diagnostic.
    vals=[]
    for di,ds in enumerate(DATASETS):
        idx=np.arange(256)
        rng=torch.Generator(device=DEVICE); rng.manual_seed(seed+950+di)
        b=source_batch(helper,branches,ds,"validation",idx,False,rng,level3_mask,aug_cfg)
        sid=torch.full((256,),NSL_KDD if ds=="nsl_kdd" else UNSW_NB15,device=DEVICE,dtype=torch.long)
        with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.float16):
            zf=stage7(b["Z_s"],b["R_proj"])["Z_f"]
            vals.append((zf,b["Z_d"],sid,b["available"]))
    zf=torch.cat([v[0] for v in vals]); zd=torch.cat([v[1] for v in vals])
    sid=torch.cat([v[2] for v in vals]); av=torch.cat([v[3] for v in vals])
    with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.float16):
        ao=stage8(zf,zd,sid,av,level.expand(512,-1),confidence.expand(512,-1),group_only.expand(512,-1))
        coral=float(stage8.coral_loss(ao,sid)["L_coral"].float().item())
        gout=stage7(torch.cat([v[0] for v in vals])[:,:] if False else vals[0][0],vals[0][1]) if False else None
    # Gate means from the deterministic validation batches above.
    gate_mean={}
    for di,ds in enumerate(DATASETS):
        b=source_batch(helper,branches,ds,"validation",np.arange(256),False,
            torch.Generator(device=DEVICE).manual_seed(seed+1100+di),level3_mask,aug_cfg)
        with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.float16):
            gg=stage7(b["Z_s"],b["R_proj"])["G"].float()
        gate_mean[ds]=float(gg.mean().item())
    stage7.train(); stage8.train(); stage10.train(); classifier.train()
    return {"datasets":metrics,"mean_macro_f1":float(np.mean([metrics[d]["macro_f1"] for d in DATASETS])),
            "validation_L_coral":coral,"gate_mean":gate_mean,
            "gate_saturation":{d:bool(v<0.01 or v>0.99) for d,v in gate_mean.items()}}

def main(args):
    global RUN_DIR,DEVICE,taxonomy,helper,ABLATION_MODE,HEAD_MODE,NATIVE_CLASSES,NATIVE_WEIGHTS,COMMON_MASK,BATCH_PER_DATASET,CACHE_ROOT,CACHE_ARRAYS,CACHE_MANIFEST_SHA
    ABLATION_MODE=args.ablation
    HEAD_MODE=args.head_mode
    BATCH_PER_DATASET=int(args.batch_per_dataset)
    RUN_DIR=Path(args.run_dir)
    if RUN_DIR.exists(): raise FileExistsError(f"Refusing to overwrite run directory {RUN_DIR}")
    RUN_DIR.mkdir(parents=True,exist_ok=False)
    LOG=RUN_DIR/"progress.log"
    verify_protected()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    free,total=torch.cuda.mem_get_info()
    if free<5*1024**3: raise RuntimeError(f"GPU free memory below approved preflight 5 GiB: {free/1024**3:.2f}")
    if free_ram_gb()<3.0: raise RuntimeError(f"Free RAM below 3 GiB preflight: {free_ram_gb():.2f}")
    if shutil_free:=__import__("shutil").disk_usage("C:\\").free < 10*1024**3:
        raise RuntimeError("Disk free below 10 GiB")
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    DEVICE=torch.device("cuda")
    taxonomy=load_taxonomy(TAXONOMY,purpose="training")
    helper,branches=read_data()
    if args.cache_root:
        CACHE_ROOT=Path(args.cache_root)
        manifest_path=CACHE_ROOT/"cache_manifest.json"
        if not manifest_path.exists(): raise RuntimeError("Input cache manifest missing; refusing cached training")
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("stage5_sha256")!=STAGE5_SHA: raise RuntimeError("Cached Stage5 hash differs from protected snapshot")
        CACHE_MANIFEST_SHA=__import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
        CACHE_ARRAYS={}
        for ds in DATASETS:
            CACHE_ARRAYS[ds]={}
            for split in ("train","validation"):
                CACHE_ARRAYS[ds][split]={}
                for mode in ("full","common_only"):
                    sd=CACHE_ROOT/ds/split
                    CACHE_ARRAYS[ds][split][mode]={
                      "zszd":np.load(sd/f"{mode}_zszd.npy",mmap_mode="r"),
                      "tokens_path":str(sd/f"{mode}_tokens.npy"),
                      "available":np.load(sd/f"{mode}_available.npy",mmap_mode="r")}
        # Drop the full raw arrays and frozen Stage5 instance; the immutable cache
        # supplies Stage5 outputs, while the accepted Stage6 branches remain frozen.
        if hasattr(helper,"stage5"): helper.stage5=None
        helper.datasets={ds:{sp:{"numeric_values":CACHE_ARRAYS[ds][sp]["full"]["zszd"]} for sp in ("train","validation")} for ds in DATASETS}
        gc.collect(); torch.cuda.empty_cache()
        log(LOG,f"CACHED_INPUTS_READY manifest_sha256={CACHE_MANIFEST_SHA} free_ram_gb={free_ram_gb():.2f}")
    else:
        memmap_dataset_inputs(helper,RUN_DIR,LOG)
    labels=load_labels(taxonomy)
    ck5=torch.load(STAGE5,map_location="cpu",weights_only=False)
    aug_cfg=ck5["config"]; del ck5
    level,confidence,group_only,level3_mask=make_metadata()
    COMMON_MASK=(level==1)
    torch.cuda.reset_peak_memory_stats()
    stage7=AdaptiveResidualFusion().to(DEVICE)
    stage8=Stage8SelectiveAlignment(coral_lambda=float(args.lambda_coral),mode=("compatibility" if args.ablation=="full" else "no_alignment")).to(DEVICE)
    stage9=UnifiedRepresentation()
    stage10=Stage10Transformer().to(DEVICE)
    pooled_labels=np.concatenate([labels[d]["native_train"] for d in DATASETS]).astype(str)
    pooled_mapped=np.concatenate([labels[d]["train"] for d in DATASETS])
    class_weights,counts=compute_class_weights([taxonomy.classes[int(i)] for i in pooled_mapped],taxonomy.classes,clip_ratio=15.0)
    if HEAD_MODE=="native":
        NATIVE_CLASSES={d:labels[d]["native_classes"] for d in DATASETS}
        NATIVE_WEIGHTS={}
        for d in DATASETS:
            w,_=compute_class_weights(labels[d]["native_train"],NATIVE_CLASSES[d],clip_ratio=15.0)
            NATIVE_WEIGHTS[d]=w.to(DEVICE)
        classifier=NativeAblationHeads(NATIVE_CLASSES).to(DEVICE)
    else:
        classifier=Stage11Classifier(len(taxonomy.classes),class_weights=class_weights).to(DEVICE)
    fit_references(helper,branches,stage7,stage8,level3_mask,aug_cfg,LOG)
    trainable=list(stage7.parameters())+list(stage8.W_a.parameters())+list(stage8.W_u.parameters())+list(stage10.parameters())+list(classifier.parameters())
    optimizer=torch.optim.AdamW(trainable,lr=5e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda")
    rng=torch.Generator(device=DEVICE); rng.manual_seed(args.seed+19001)
    row_rng={d:np.random.default_rng(args.seed+2000+i) for i,d in enumerate(DATASETS)}
    permutations={d:row_rng[d].permutation(len(labels[d]["train"])) for d in DATASETS}
    cursors={d:0 for d in DATASETS}
    warmup_steps=max(1,int(args.max_steps*0.1))
    history=[]; best=-1.0; best_step=0; stale=0; eval_count=0
    config={"seed":args.seed,"ablation":args.ablation,"lambda_coral":float(args.lambda_coral if args.ablation=="full" else 0.0),"lambda_gate":0.0,
      "head_mode":HEAD_MODE,"max_steps":args.max_steps,"validation_interval":args.validation_interval,
      "early_stop_patience":6,"min_relative_improvement":0.005,"warmup_steps":warmup_steps,
      "batch_per_dataset":BATCH_PER_DATASET,"dataset_weighting":"0.5/0.5",
      "optimizer":"AdamW","learning_rate":5e-4,"weight_decay":1e-4,"gradient_clip":1.0,
      "mixed_precision":"CUDA float16 autocast + GradScaler","stage5_eval_no_grad":True,
      "stage6_eval_no_grad":True,"stage5_6_feature_mask_augmentation":"training only, applied to input masks",
      "feature_confidence_encoding":CONFIDENCE_VALUE,"taxonomy":str(TAXONOMY),
      "stage5_sha256":STAGE5_SHA,"stage6_sha256":STAGE6_SHA,"taxonomy_sha256":digest(TAXONOMY),
      "class_weight_counts":counts,"class_weights":class_weights.tolist(),
      "native_class_names":({d:list(NATIVE_CLASSES[d]) for d in DATASETS} if HEAD_MODE=="native" else None),
      "native_class_weights":({d:NATIVE_WEIGHTS[d].detach().cpu().tolist() for d in DATASETS} if HEAD_MODE=="native" else None),
      "validation_support_under20":{d:metrics for d,metrics in [(ds,{taxonomy.classes[int(c)]:int(np.sum(labels[ds]['validation']==c)) for c in np.unique(labels[ds]['validation']) if int(np.sum(labels[ds]['validation']==c))<20}) for ds in DATASETS]}}
    atomic_json(RUN_DIR/"run_config.json",config)
    log(LOG,f"RUN_START seed={args.seed} lambda_coral={args.lambda_coral} steps={args.max_steps} val_interval={args.validation_interval} free_ram_gb={free_ram_gb():.2f} gpu_free_mib={free/1024**2:.0f}")
    start=time.perf_counter(); last_log=start; last_checkpoint=start
    for step in range(1,args.max_steps+1):
        if free_ram_gb()<2.0: raise RuntimeError(f"HARD STOP free RAM below 2 GiB at step {step}: {free_ram_gb():.2f}")
        optimizer.zero_grad(set_to_none=True)
        batches=[]
        for di,ds in enumerate(DATASETS):
            n=len(labels[ds]["train"])
            if cursors[ds]+BATCH_PER_DATASET>n:
                permutations[ds]=row_rng[ds].permutation(n); cursors[ds]=0
            idx=permutations[ds][cursors[ds]:cursors[ds]+BATCH_PER_DATASET]
            cursors[ds]+=BATCH_PER_DATASET
            batches.append(source_batch(helper,branches,ds,"train",idx,True,rng,level3_mask,aug_cfg))
        batch={k:torch.cat([b[k] for b in batches],dim=0) for k in batches[0]}
        sid=torch.cat([torch.full((BATCH_PER_DATASET,),NSL_KDD,device=DEVICE,dtype=torch.long),
                       torch.full((BATCH_PER_DATASET,),UNSW_NB15,device=DEVICE,dtype=torch.long)])
        idx_nsl=permutations["nsl_kdd"][cursors["nsl_kdd"]-BATCH_PER_DATASET:cursors["nsl_kdd"]]
        idx_unsw=permutations["unsw_nb15"][cursors["unsw_nb15"]-BATCH_PER_DATASET:cursors["unsw_nb15"]]
        y_nsl=torch.as_tensor(labels["nsl_kdd"]["native_train_idx"][idx_nsl] if HEAD_MODE=="native" else labels["nsl_kdd"]["train"][idx_nsl],device=DEVICE,dtype=torch.long)
        y_unsw=torch.as_tensor(labels["unsw_nb15"]["native_train_idx"][idx_unsw] if HEAD_MODE=="native" else labels["unsw_nb15"]["train"][idx_unsw],device=DEVICE,dtype=torch.long)
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            logits,gout,aout,fused=batch_forward(batch,stage7,stage8,stage9,stage10,classifier,sid,
                level,confidence,group_only,min(1.0,step/warmup_steps))
            if HEAD_MODE=="native":
                loss_nsl=torch.nn.functional.cross_entropy(logits["nsl_kdd"].float(),y_nsl,weight=NATIVE_WEIGHTS["nsl_kdd"])
                loss_unsw=torch.nn.functional.cross_entropy(logits["unsw_nb15"].float(),y_unsw,weight=NATIVE_WEIGHTS["unsw_nb15"])
            else:
                loss_nsl=classifier.loss(logits[:BATCH_PER_DATASET].float(),y_nsl)
                loss_unsw=classifier.loss(logits[BATCH_PER_DATASET:].float(),y_unsw)
            l_cls=0.5*(loss_nsl+loss_unsw)
            align=stage8.coral_loss(aout,sid)
            loss=l_cls+align["L_align"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Stage7-11 loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable,1.0)
        scaler.step(optimizer); scaler.update()
        now=time.perf_counter()
        if now-last_log>=60 or step==1 or step==args.max_steps:
            elapsed=now-start; sec=elapsed/step; eta=sec*(args.max_steps-step)
            log(LOG,f"TRAIN_PROGRESS step={step}/{args.max_steps} L={float(loss.item()):.7f} L_cls={float(l_cls.item()):.7f} L_coral={float(align['L_coral'].item()):.7f} L_align={float(align['L_align'].item()):.7f} gate_mean={float(gout['G'].float().mean().item()):.5f} elapsed_s={elapsed:.1f} eta_s={eta:.1f} free_ram_gb={free_ram_gb():.2f} gpu_alloc_mib={torch.cuda.memory_allocated()/1024**2:.1f} gpu_peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}")
            last_log=now
        if torch.cuda.max_memory_allocated()>6*1024**3:
            raise RuntimeError("HARD STOP GPU memory allocation exceeded 6 GiB")
        if step%1000==0 or now-last_checkpoint>=60:
            state={"stage7":stage7.state_dict(),"stage8":stage8.state_dict(),"stage10":stage10.state_dict(),
                   "classifier":classifier.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                   "step":step,"seed":args.seed,"lambda_coral":args.lambda_coral,"config":config,
                   "stage5_sha256":STAGE5_SHA,"stage6_sha256":STAGE6_SHA,"taxonomy_sha256":digest(TAXONOMY)}
            atomic_torch(RUN_DIR/"last.pt",state); last_checkpoint=now
        # Release the previous step's autograd graph before allocating the next one.
        # Without this, Python locals keep the old graph alive across loop iterations.
        del loss,logits,gout,aout,fused,batch,batches,sid,y_nsl,y_unsw,align
        if step%10==0:
            gc.collect(); trim_working_set()
        if step%args.validation_interval==0 or step==args.max_steps:
            val=evaluate(helper,branches,stage7,stage8,stage9,stage10,classifier,labels,
                         level,confidence,group_only,level3_mask,aug_cfg,LOG,args.seed)
            val["step"]=step; history.append(val); eval_count+=1
            metric=val["mean_macro_f1"]
            improved=metric>=best*(1.0+0.005) if best>0 else metric>best
            if improved:
                best=metric; best_step=step; stale=0
                state={"stage7":stage7.state_dict(),"stage8":stage8.state_dict(),"stage10":stage10.state_dict(),
                       "classifier":classifier.state_dict(),"step":step,"seed":args.seed,"lambda_coral":args.lambda_coral,
                       "best_validation":val,"validation_history":history,"config":config,
                       "stage5_sha256":STAGE5_SHA,"stage6_sha256":STAGE6_SHA,"taxonomy_sha256":digest(TAXONOMY)}
                atomic_torch(RUN_DIR/"best.pt",state)
            else: stale+=1
            atomic_json(RUN_DIR/"validation_history.json",history)
            atomic_json(RUN_DIR/"latest_validation.json",val)
            log(LOG,f"VALIDATION step={step} mean_macro_f1={metric:.6f} best={best:.6f} best_step={best_step} stale={stale}/6 nsl={val['datasets']['nsl_kdd']['macro_f1']:.6f} unsw={val['datasets']['unsw_nb15']['macro_f1']:.6f} coral={val['validation_L_coral']:.7f} gates={val['gate_mean']}")
            if eval_count>=8 and stale>=6:
                log(LOG,f"EARLY_STOP step={step} after_six_nonimproving_validations"); break
    elapsed=time.perf_counter()-start
    final_state={"stage7":stage7.state_dict(),"stage8":stage8.state_dict(),"stage10":stage10.state_dict(),
       "classifier":classifier.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
       "step":step,"seed":args.seed,"lambda_coral":args.lambda_coral,"config":config,
       "validation_history":history,"stage5_sha256":STAGE5_SHA,"stage6_sha256":STAGE6_SHA,
       "taxonomy_sha256":digest(TAXONOMY)}
    atomic_torch(RUN_DIR/"last.pt",final_state)
    final={"seed":args.seed,"lambda_coral":args.lambda_coral,"steps_completed":step,
       "best_step":best_step,"best_validation_macro_f1":best,"elapsed_seconds":elapsed,
       "seconds_per_step":elapsed/max(step,1),"gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),
       "validation_history":history,"best_checkpoint_sha256":digest(RUN_DIR/"best.pt") if (RUN_DIR/"best.pt").exists() else None,
       "last_checkpoint_sha256":digest(RUN_DIR/"last.pt") if (RUN_DIR/"last.pt").exists() else None,
       "stage5_sha256":STAGE5_SHA,"stage6_sha256":STAGE6_SHA}
    atomic_json(RUN_DIR/"result.json",final)
    log(LOG,f"RUN_COMPLETE {json.dumps({k:v for k,v in final.items() if k!='validation_history'},sort_keys=True)}")
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--lambda-coral",type=float,default=0.0)
    ap.add_argument("--max-steps",type=int,default=100)
    ap.add_argument("--validation-interval",type=int,default=100)
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--batch-per-dataset",type=int,default=256)
    ap.add_argument("--cache-root",default=None)
    ap.add_argument("--ablation",choices=["full","shared_only","residual_fixed_gate1","residual_adaptive","alignment_off","common_only"],default="full")
    ap.add_argument("--head-mode",choices=["unified","native"],default="unified")
    args=ap.parse_args()
    try:
        raise SystemExit(main(args))
    except Exception:
        print(traceback.format_exc(),flush=True)
        raise







