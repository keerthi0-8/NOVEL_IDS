from __future__ import annotations
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_stage7_11_experiment as r
from novel_ids.fusion.stage7 import AdaptiveResidualFusion
from novel_ids.alignment.stage8 import Stage8SelectiveAlignment,NSL_KDD,UNSW_NB15
from novel_ids.fusion.stage9 import UnifiedRepresentation
from novel_ids.transformer.stage10 import Stage10Transformer
from novel_ids.classification.stage11 import Stage11Classifier,compute_class_weights
from novel_ids.taxonomy.stage11 import load_taxonomy
from novel_ids.calibration.temperature import TemperatureScaler,expected_calibration_error
from novel_ids.ood.stage11 import ClassConditionalMahalanobis,OODScoreCombiner,energy_score

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(log,obj):
    line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {obj}"
    print(line,flush=True)
    with open(log,"a",encoding="utf-8") as f:f.write(line+"\n")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--smoke-batches",type=int,default=0)
    args=ap.parse_args()
    out=Path(args.run_dir)
    out.mkdir(parents=True,exist_ok=False)
    log=out/"progress.log"
    r.RUN_DIR=out
    r.DEVICE=torch.device("cuda")
    r.verify_protected()
    free,total=torch.cuda.mem_get_info()
    if free < 5*1024**3: raise RuntimeError(f"Preflight GPU free <5 GiB: {free/1024**3:.2f}")
    if r.free_ram_gb()<2.0: raise RuntimeError(f"HARD STOP free RAM <2 GiB: {r.free_ram_gb():.2f}")
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(43); np.random.seed(43)
    r.taxonomy=load_taxonomy(r.TAXONOMY,purpose="training")
    helper,branches=r.read_data()
    r.helper=helper
    labels=r.load_labels(r.taxonomy)
    ck5=torch.load(r.STAGE5,map_location="cpu",weights_only=False); aug_cfg=ck5["config"]; del ck5
    level,confidence,group_only,level3_mask=r.make_metadata()
    stage7=AdaptiveResidualFusion().to(r.DEVICE).eval()
    stage8=Stage8SelectiveAlignment(coral_lambda=0.05,mode="compatibility").to(r.DEVICE).eval()
    stage9=UnifiedRepresentation()
    stage10=Stage10Transformer().to(r.DEVICE).eval()
    mapped=np.concatenate([labels[d]["train"] for d in r.DATASETS])
    cw,_=compute_class_weights([r.taxonomy.classes[int(i)] for i in mapped],r.taxonomy.classes,clip_ratio=15.0)
    classifier=Stage11Classifier(len(r.taxonomy.classes),class_weights=cw).to(r.DEVICE).eval()
    ck=torch.load(args.checkpoint,map_location=r.DEVICE,weights_only=False)
    expected={"stage5_sha256":r.STAGE5_SHA,"stage6_sha256":r.STAGE6_SHA}
    for k,v in expected.items():
        if ck.get(k)!=v: raise RuntimeError(f"Checkpoint provenance mismatch: {k}")
    stage7.load_state_dict(ck["stage7"],strict=True)
    stage8.load_state_dict(ck["stage8"],strict=True)
    stage10.load_state_dict(ck["stage10"],strict=True)
    classifier.load_state_dict(ck["classifier"],strict=True)
    step=int(ck["step"]); seed=int(ck["seed"])
    del ck; gc.collect(); torch.cuda.empty_cache()
    started=time.time()
    emit(log,f"PHASEA_CALIBRATION_START checkpoint={args.checkpoint} seed={seed} step={step} gpu_free_mib={free/1024**2:.0f} ram_gb={r.free_ram_gb():.2f}")
    feats={}; logits={}; zs={}; ys={}; ids={}; gate_parts={}
    calibration_indices={}; probe_indices={}
    for di,ds in enumerate(r.DATASETS):
        ntrain=len(helper.datasets[ds]["train"]["numeric_values"])
        rg=np.random.default_rng(4300+di)
        calibration_indices[ds]=np.sort(rg.choice(ntrain,min(20000,ntrain),replace=False))
        probe_indices[ds]=np.sort(rg.choice(ntrain,min(10000,ntrain),replace=False))
        for split in ("train","validation"):
            n=len(helper.datasets[ds][split]["numeric_values"])
            fz=[]; fl=[]; fs=[]; fy=[]
            stage7.eval(); stage8.eval(); stage10.eval(); classifier.eval()
            for st in range(0,n,256):
                if r.free_ram_gb()<2.0: raise RuntimeError(f"HARD STOP free RAM <2 GiB during embedding: {r.free_ram_gb():.2f}")
                ix=np.arange(st,min(n,st+256))
                b=r.source_batch(helper,branches,ds,split,ix,False,
                    torch.Generator(device=r.DEVICE).manual_seed(4300+di),level3_mask,aug_cfg)
                sid=torch.full((len(ix),),NSL_KDD if ds=="nsl_kdd" else UNSW_NB15,device=r.DEVICE,dtype=torch.long)
                with torch.no_grad():
                    lo,gout,_,fu=r.batch_forward(b,stage7,stage8,stage9,stage10,classifier,sid,level,confidence,group_only,1.0)
                    if split=="validation":
                        fz.append(fu["Z_u"].detach().float().cpu()); fl.append(lo.detach().float().cpu())
                        fs.append(b["Z_s"].detach().float().cpu()); fy.append(labels[ds][split][ix])
                        gate_parts.setdefault(ds,[]).append(gout["G"].detach().float().cpu())
                    else:
                        cm=np.isin(ix,calibration_indices[ds]); pm=np.isin(ix,probe_indices[ds])
                        if cm.any():
                            fz.append(fu["Z_u"][torch.as_tensor(cm,device=fu["Z_u"].device)].detach().float().cpu()); fl.append(lo[torch.as_tensor(cm,device=lo.device)].detach().float().cpu()); fy.append(labels[ds][split][ix[cm]])
                        if pm.any(): fs.append(b["Z_s"][torch.as_tensor(pm,device=b["Z_s"].device)].detach().float().cpu())
                if st==0 or (st//256)%100==0:
                    emit(log,f"EMBED {ds} {split} rows={min(n,st+256)}/{n} ram_gb={r.free_ram_gb():.2f} gpu_peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}")
                if args.smoke_batches and (st//256+1)>=args.smoke_batches:
                    emit(log,f"SMOKE_TEST_COMPLETE batches={args.smoke_batches} dataset={ds} split={split} elapsed_s={time.time()-started:.1f}")
                    return
            key=f"{ds}_{split}"
            feats[key]=torch.cat(fz); logits[key]=torch.cat(fl); zs[key]=torch.cat(fs)
            ys[key]=np.concatenate(fy); ids[key]=np.full(len(fy[0]) if False else len(ys[key]),di,dtype=np.int64)
            if not torch.isfinite(feats[key]).all() or not torch.isfinite(logits[key]).all() or not torch.isfinite(zs[key]).all():
                raise FloatingPointError(f"Non-finite embedding/logit in {key}")
            emit(log,f"EMBED_COMPLETE {key} n={len(ys[key])}")
    # One global temperature fitted only on pooled source validation logits.
    val_logits=torch.cat([logits[f"{d}_validation"] for d in r.DATASETS])
    val_y=torch.as_tensor(np.concatenate([ys[f"{d}_validation"] for d in r.DATASETS]),dtype=torch.long)
    scaler=TemperatureScaler().fit(val_logits,val_y)
    t=float(scaler.temperature)
    # Class-conditional Mahalanobis and energy normalization are fit on source TRAIN only.
    train_z=torch.cat([feats[f"{d}_train"] for d in r.DATASETS])
    train_y=torch.as_tensor(np.concatenate([ys[f"{d}_train"] for d in r.DATASETS]),dtype=torch.long)
    train_logits=torch.cat([logits[f"{d}_train"] for d in r.DATASETS])
    # Classes with fewer than 20 training examples are excluded and explicitly reported.
    mahal=ClassConditionalMahalanobis(min_class_samples=20,shrinkage=0.1,ridge=1e-4).fit(train_z,train_y)
    train_m=mahal.score(train_z)
    train_e=energy_score(train_logits,temperature=t)
    comb=OODScoreCombiner(weights=(0.5,0.5)).fit(train_m,train_e)
    train_comb=comb.score(train_m,train_e)
    val_m=mahal.score(torch.cat([feats[f"{d}_validation"] for d in r.DATASETS]))
    val_e=energy_score(val_logits,temperature=t)
    val_comb=comb.score(val_m,val_e)
    tau_ood=float(torch.quantile(val_comb,0.95).item())
    val_probs=torch.softmax(val_logits/t,dim=1)
    val_conf=val_probs.max(1).values
    tau_conf=float(torch.quantile(val_conf,0.10).item())
    # Dataset-identity probe on Z_s; balanced deterministic training subsample, validation-only score.
    rng=np.random.default_rng(43)
    Xtr=[]; ytr=[]; Xva=[]; yva=[]
    for di,d in enumerate(r.DATASETS):
        x=zs[f"{d}_train"].numpy(); take=len(x)
        Xtr.append(x); ytr.append(np.full(take,di))
        Xva.append(zs[f"{d}_validation"].numpy()); yva.append(np.full(len(Xva[-1]),di))
    probe=LogisticRegression(max_iter=1000,solver="lbfgs",n_jobs=2,random_state=43)
    probe.fit(np.concatenate(Xtr),np.concatenate(ytr))
    probe_pred=probe.predict(np.concatenate(Xva)); probe_prob=probe.predict_proba(np.concatenate(Xva))[:,1]
    probe_acc=float(accuracy_score(np.concatenate(yva),probe_pred))
    probe_bal=float(balanced_accuracy_score(np.concatenate(yva),probe_pred))
    probe_auc=float(roc_auc_score(np.concatenate(yva),probe_prob))
    # Per-dataset validation calibration and OOD diagnostics.
    per={}
    for d in r.DATASETS:
        k=f"{d}_validation"; y=ys[k]; lo=logits[k]; z=feats[k]
        p=torch.softmax(lo/t,dim=1); pred=p.argmax(1).numpy()
        m=mahal.score(z); e=energy_score(lo,temperature=t); score=comb.score(m,e)
        per[d]={"rows":int(len(y)),"accuracy":float(np.mean(pred==y)),
          "macro_f1":float(__import__("sklearn.metrics").metrics.f1_score(y,pred,labels=np.unique(y),average="macro",zero_division=0)),
          "weighted_f1":float(__import__("sklearn.metrics").metrics.f1_score(y,pred,average="weighted",zero_division=0)),
          "ece_15_bins":expected_calibration_error(lo/t,torch.as_tensor(y)),
          "ood_score_mean":float(score.mean()),"ood_score_p95":float(torch.quantile(score,0.95)),
          "uncertain_rate":float((p.max(1).values<tau_conf).float().mean()),
          "unfamiliar_rate":float((score>tau_ood).float().mean())}
    distributions={}
    for d in r.DATASETS:
        k=f"{d}_validation"
        probs=torch.softmax(logits[k]/t,dim=1)
        score=comb.score(mahal.score(feats[k]),energy_score(logits[k],temperature=t))
        distributions[d+"_labels"]=ys[k]
        distributions[d+"_probabilities"]=probs.numpy()
        distributions[d+"_ood_scores"]=score.numpy()
        distributions[d+"_gate_mean"]=torch.cat(gate_parts[d],dim=0).mean(0).numpy()
        distributions[d+"_predictions"]=probs.argmax(1).numpy()
    np.savez_compressed(out/"validation_distributions.npz",**distributions)
    names={int(k):r.taxonomy.classes[int(k)] for k in range(len(r.taxonomy.classes))}
    # Exploratory held-out-class OOD proxy: remove one class from the OOD reference
    # and mask its output logit at inference. Upstream Stage7-11 weights were trained
    # with these classes, so this is explicitly NOT a strict leave-class-out retrain.
    heldout_targets={"nsl_kdd":"NSL_R2L","unsw_nb15":"UNSW_Shellcode"}
    heldout_dev={}
    for d, cname in heldout_targets.items():
        cid=r.taxonomy.classes.index(cname)
        tk=f"{d}_train"; vk=f"{d}_validation"
        ty=ys[tk]; vy=ys[vk]
        known_tr=torch.as_tensor(ty!=cid,dtype=torch.bool)
        hmahal=ClassConditionalMahalanobis(min_class_samples=20,shrinkage=0.1,ridge=1e-4).fit(feats[tk][known_tr],torch.as_tensor(ty[known_tr]))
        tlog=logits[tk].clone(); tlog[:,cid]=-1e4
        tm=hmahal.score(feats[tk][known_tr]); te=energy_score(tlog[known_tr],temperature=t)
        hcomb=OODScoreCombiner(weights=(0.5,0.5)).fit(tm,te)
        vlog=logits[vk].clone(); vlog[:,cid]=-1e4
        vm=hmahal.score(feats[vk]); ve=energy_score(vlog,temperature=t); vs=hcomb.score(vm,ve)
        is_held=torch.as_tensor(vy==cid,dtype=torch.bool); is_known=~is_held
        if int(is_held.sum())==0 or int(is_known.sum())==0: raise RuntimeError(f"Held-out class {cname} has no validation support or no known comparison rows")
        threshold=float(torch.quantile(vs[is_known],0.95).item())
        truth=is_held.numpy().astype(int); score=vs.numpy()

        heldout_dev[d]={"held_out_class":cname,"validation_support":int(is_held.sum()),
          "known_validation_support":int(is_known.sum()),"threshold_known_validation_p95":threshold,
          "heldout_detection_rate_at_5pct_known_validation_fpr":float((vs[is_held]>threshold).float().mean()),
          "known_validation_fpr":float((vs[is_known]>threshold).float().mean()),
          "auroc_heldout_vs_known_validation":float(roc_auc_score(truth,score)),
          "protocol":"OOD reference fit excludes held-out class; held-out class logit masked; threshold from known validation 95th percentile",
          "limitation":"post-hoc OOD proxy only: the frozen Stage7-11 checkpoint was originally trained with this class; not a strict leave-class-out retraining result"}
    payload={"phase":"A","seed_selected":seed,"checkpoint":str(Path(args.checkpoint)),
      "checkpoint_sha256":sha(args.checkpoint),"stage5_sha256":r.STAGE5_SHA,"stage6_sha256":r.STAGE6_SHA,
      "taxonomy_sha256":sha(r.TAXONOMY),"temperature":t,"tau_ood_validation_p95":tau_ood,
      "tau_conf_validation_p10":tau_conf,"ood_combiner":{"weights":[0.5,0.5],
      "train_mahal_mean":comb.mean_mahalanobis,"train_mahal_std":comb.std_mahalanobis,
      "train_energy_mean":comb.mean_energy,"train_energy_std":comb.std_energy,
      "mahal_min_class_samples":20,"mahal_shrinkage":0.1,"mahal_ridge":1e-4,
      "excluded_classes":{names[k]:v for k,v in (mahal.excluded_classes or {}).items()},
      "included_counts":{names[k]:v for k,v in (mahal.included_counts or {}).items()}},
      "validation_metrics":per,"heldout_class_ood_development_proxy":heldout_dev,"dataset_identity_probe_Z_s":{"balanced_train_rows_per_dataset":10000,
      "validation_accuracy":probe_acc,"validation_balanced_accuracy":probe_bal,"validation_roc_auc":probe_auc,
      "random_baseline_accuracy":0.5},
      "data_access":"source train and validation only; no source test or unseen data accessed",
      "memory_reduction":"deterministic seeded uniform random sample without replacement: 20,000 source-training rows per dataset for Mahalanobis/energy fitting; 10,000 per dataset for Z_s identity probe; all validation rows retained. Samples drawn from training only.",
      "checkpoint_config":str(Path(args.checkpoint).parent/"run_config.json")}
    (out/"phaseA_calibration.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    # Persist exact configuration and hash manifest for the frozen primary.
    cfg=json.loads((Path(args.checkpoint).parent/"run_config.json").read_text(encoding="utf-8"))
    cfg.update({"selected_checkpoint_sha256":sha(args.checkpoint),"selected_seed":seed,"selected_step":step,
      "lambda_coral":0.05,"lambda_gate":0.0,"temperature":t,"tau_ood":tau_ood,"tau_conf":tau_conf,
      "ood_definition":"mean z-scored class-conditional Mahalanobis and energy; source-training score normalization; thresholds from pooled validation",
      "phaseA_calibration_sha256":"pending"})
    cfg_path=out/"primary_frozen_config.json"
    cal_for_hash=json.dumps(payload,indent=2,allow_nan=False).encode("utf-8")
    cal_hash=hashlib.sha256(cal_for_hash).hexdigest()
    cfg["phaseA_calibration_sha256"]=cal_hash
    cfg["phaseA_calibration_hash_note"]="SHA256 of calibration JSON payload excluding primary_config_sha256 back-reference to avoid circular hashing"
    cfg_path.write_text(json.dumps(cfg,indent=2,allow_nan=False),encoding="utf-8")
    payload["primary_config_sha256"]=sha(cfg_path)
    (out/"phaseA_calibration.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    emit(log,f"PHASEA_COMPLETE seed={seed} T={t:.8f} tau_ood={tau_ood:.8f} tau_conf={tau_conf:.8f} probe_acc={probe_acc:.6f} probe_auc={probe_auc:.6f} config_sha256={sha(cfg_path)}")
    print(json.dumps(payload,indent=2),flush=True)
if __name__=="__main__": main()




