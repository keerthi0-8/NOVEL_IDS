from __future__ import annotations
import copy, gc, json, time
from pathlib import Path
import numpy as np, torch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from novel_ids.residual.stage6 import Stage6ResidualBranch, ResidualEncoder
from novel_ids.residual.stage6_data import load_stage6_label_artifact
from novel_ids.residual.stage6_fastpath import fit_logistic_probe, macro_f1_supported, sha256_file

OUT=Path(r"C:\NOVEL_IDS_runs\stage6\phase2_fastpath_20261125")
CACHE=Path(r"C:\NOVEL_IDS_runs\stage6\phase2_fastcache_20261125\cache")
INPUT=Path(r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project\data\interim\stage6_trainval_only")
DATASETS=("nsl_kdd","unsw_nb15"); DEVICE="cuda"; BATCH=512; BOOTSTRAPS=2000
def log(s):
 line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"; print(line,flush=True)
 with (OUT/"probe_evaluation_progress.log").open("a",encoding="utf-8") as f:f.write(line+"\n")
def write_json(p,obj):
 tmp=p.with_suffix(p.suffix+".tmp"); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str),encoding="utf-8"); tmp.replace(p)
def fit_probe(name,tr,va,yt,yv,standard,seed=42):
 log(f"PROBE_START {name} standardize={standard} train_rows={len(yt)} validation_rows={len(yv)}")
 r=fit_logistic_probe(tr,va,yt,yv,seed=seed,standardize_blocks=standard)
 pred=r.pop("prediction"); r.pop("scaler_stats",None)
 log(f"PROBE_COMPLETE {name} standardize={standard} macro_f1={r['macro_f1']:.6f} fit_s={r['fit_seconds']:.2f} n_iter={r['n_iter']}")
 return r,pred
def probe_codes(y,pred,labels):
 c=len(labels); yt=np.searchsorted(labels,y).astype(np.int32); pi=np.searchsorted(labels,pred)
 valid=pi<c; valid &= labels[np.minimum(pi,c-1)]==pred
 out=np.full(len(y),-1,dtype=np.int32); out[valid]=yt[valid]*c+pi[valid].astype(np.int32)
 return out
def score_sampled(codes,idx,c):
 b,n=idx.shape; samp=codes[idx]; valid=samp>=0
 offsets=np.arange(b,dtype=np.int64)[:,None]*(c*c)
 keys=samp[valid].astype(np.int64)+np.broadcast_to(offsets,samp.shape)[valid]
 cm=np.bincount(keys,minlength=b*c*c).reshape(b,c,c).astype(np.float64)
 tp=np.diagonal(cm,axis1=1,axis2=2); fp=cm.sum(axis=1)-tp; fn=cm.sum(axis=2)-tp
 den=2*tp+fp+fn; f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0)
 return f1.mean(axis=1)
def bootstrap_gain(y,p0,p1,seed):
 y=np.asarray(y).astype(str); p0=np.asarray(p0).astype(str); p1=np.asarray(p1).astype(str)
 labels=np.unique(y); c=len(labels); rng=np.random.default_rng(seed)
 observed=macro_f1_supported(y,p1,labels)-macro_f1_supported(y,p0,labels)
 b0=probe_codes(y,p0,labels); b1=probe_codes(y,p1,labels); draws=np.empty(BOOTSTRAPS,dtype=np.float64)
 for start in range(0,BOOTSTRAPS,50):
  stop=min(BOOTSTRAPS,start+50); idx=rng.integers(0,len(y),size=(stop-start,len(y)),dtype=np.int32)
  draws[start:stop]=score_sampled(b1,idx,c)-score_sampled(b0,idx,c)
 return float(observed),draws
def combined_bootstrap(y_by_ds,p0_by_ds,p1_by_ds):
 gains=[]; observed=[]
 for i,ds in enumerate(DATASETS):
  obs,draw=bootstrap_gain(y_by_ds[ds],p0_by_ds[ds],p1_by_ds[ds],42+i*100003)
  observed.append(obs); gains.append(draw)
 combined=(gains[0]+gains[1])/2
 return {"mean_gain":float(np.mean(observed)),"ci95_lower":float(np.quantile(combined,.025)),"ci95_upper":float(np.quantile(combined,.975)),"bootstrap_replicates":BOOTSTRAPS,"seed":42,"per_dataset_observed_gain":dict(zip(DATASETS,observed))}
def branch_features(branch,train_packed,val_packed):
 result={}
 for split,packed in (("train",train_packed),("validation",val_packed)):
  es=[]; rs=[]
  with torch.no_grad():
   for start in range(0,len(packed),BATCH):
    b=np.asarray(packed[start:start+BATCH],dtype=np.float32)
    zs=torch.as_tensor(b[:,:128],device=DEVICE); zd=torch.as_tensor(b[:,128:],device=DEVICE)
    o=branch(zs,zd); es.append(o["E_d"].float().cpu().numpy()); rs.append(o["R"].float().cpu().numpy())
  result[split]={"E":np.concatenate(es),"R":np.concatenate(rs)}
 return result
def random_null_encoder(e_train,e_val,seed):
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed); enc=ResidualEncoder(128,128,128).to(DEVICE).eval()
 def apply(x):
  parts=[]
  with torch.no_grad():
   for start in range(0,len(x),BATCH):
    parts.append(enc(torch.as_tensor(x[start:start+BATCH],dtype=torch.float32,device=DEVICE)).cpu().numpy())
  return np.concatenate(parts)
 return apply(e_train),apply(e_val)
def nonlinear_mlp_r2(zs_validation, residual_validation, seed=42):
 x=np.asarray(zs_validation,dtype=np.float32); y=np.asarray(residual_validation,dtype=np.float32)
 rng=np.random.default_rng(seed); order=rng.permutation(len(x)); cut=int(0.8*len(x))
 tr,te=order[:cut],order[cut:]
 xs=StandardScaler(); ys=StandardScaler()
 xtr=xs.fit_transform(x[tr]); xte=xs.transform(x[te])
 ytr=ys.fit_transform(y[tr]); yte=ys.transform(y[te])
 model=MLPRegressor(hidden_layer_sizes=(128,),activation="relu",solver="adam",alpha=1e-4,batch_size=256,learning_rate_init=1e-3,max_iter=100,early_stopping=True,validation_fraction=0.1,n_iter_no_change=10,random_state=seed)
 start=time.perf_counter()
 with threadpool_limits(limits=2):
  model.fit(xtr,ytr); pred=model.predict(xte)
 fit_s=time.perf_counter()-start
 per_dim=r2_score(yte,pred,multioutput="raw_values")
 return {"heldout_R2":float(r2_score(yte,pred,multioutput="uniform_average")),"per_dimension_R2_mean":float(np.mean(per_dim)),"per_dimension_R2_median":float(np.median(per_dim)),"per_dimension_R2_min":float(np.min(per_dim)),"train_rows":int(len(tr)),"heldout_rows":int(len(te)),"split_seed":int(seed),"hidden_layer_sizes":[128],"max_iter":100,"n_iter":int(model.n_iter_),"fit_seconds":float(fit_s),"diagnostic_only":True}
def main():
 manifest=json.loads((Path(r"C:\NOVEL_IDS_runs\stage6\phase2_fastcache_20261125")/"cache_manifest.json").read_text(encoding="utf-8"))
 if manifest.get("status")!="CACHE_EQUIVALENCE_PASSED": raise RuntimeError("Cache equivalence not passed")
 training=json.loads((OUT/"phase2_training_results.json").read_text(encoding="utf-8"))
 arms=training["seed42_arms"]
 if len(arms)!=6: raise RuntimeError(f"Expected six completed seed42 arms, got {len(arms)}")
 torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 baseline={}; y_by_ds={}; p0={}; p1={}; base_results={}; gaussian={}
 # Cache P0 and P1 once per dataset; reuse these exact validation predictions across all six arms.
 for di,ds in enumerate(DATASETS):
  train=np.load(CACHE/ds/"train_zszd.npy",mmap_mode="r"); val=np.load(CACHE/ds/"validation_zszd.npy",mmap_mode="r")
  tr=np.array(train[:,0,:],dtype=np.float32,copy=True); va=np.array(val[:,:],dtype=np.float32,copy=True)
  labels=load_stage6_label_artifact(INPUT/f"{ds}_labels.joblib",ds)
  yt=np.asarray(labels["train"]["labels"]).astype(str); yv=np.asarray(labels["validation"]["labels"]).astype(str)
  if len(yt)!=len(tr) or len(yv)!=len(va): raise RuntimeError(f"{ds}: label/cache row count mismatch")
  y_by_ds[ds]=yv; baseline[ds]={"train":{"Z_s":tr[:,:128],"Z_d":tr[:,128:]},"validation":{"Z_s":va[:,:128],"Z_d":va[:,128:]},"y_train":yt}
  p0[ds]={}; p1[ds]={}; base_results[ds]={}
  for standard in (True,False):
   tag="standardized" if standard else "unstandardized"
   r0,pred0=fit_probe(f"{ds}/P0",{"Z_s":tr[:,:128]},{"Z_s":va[:,:128]},yt,yv,standard)
   r1,pred1=fit_probe(f"{ds}/P1",{"Z_s":tr[:,:128],"Z_d":tr[:,128:]},{"Z_s":va[:,:128],"Z_d":va[:,128:]},yt,yv,standard)
   p0[ds][standard]=pred0; p1[ds][standard]=pred1
   base_results[ds][tag]={"P0":r0,"P1":r1}
  rng=np.random.default_rng(4200+di); gaussian[ds]={"train":rng.standard_normal((len(yt),128),dtype=np.float32),"validation":rng.standard_normal((len(yv),128),dtype=np.float32)}
  del train,val,tr,va; gc.collect()
 ceiling=combined_bootstrap(y_by_ds,{d:p0[d][True] for d in DATASETS},{d:p1[d][True] for d in DATASETS})
 log(f"P1_CEILING {json.dumps(ceiling,sort_keys=True)}")
 ceiling_ok=ceiling["mean_gain"]>0 and ceiling["ci95_lower"]>0
 results=[]; pred_archive={}
 for arm_meta in arms:
  arm=arm_meta["arm"]; run_dir=OUT/arm; ckpt=torch.load(run_dir/"best.pt",map_location="cpu",weights_only=False)
  branches=torch.nn.ModuleDict({d:Stage6ResidualBranch(128,128,128,128,128,128) for d in DATASETS})
  branches.load_state_dict(ckpt["branches"],strict=True); branches.to(DEVICE).eval()
  per_ds={}; p3_preds={}; nulln_preds={}; nullg_preds={}; p2_preds={}; var_ratios={}; r_archive={}
  for di,ds in enumerate(DATASETS):
   train=np.load(CACHE/ds/"train_zszd.npy",mmap_mode="r"); val=np.load(CACHE/ds/"validation_zszd.npy",mmap_mode="r")
   tr=np.array(train[:,0,:],dtype=np.float32,copy=True); va=np.array(val[:,:],dtype=np.float32,copy=True)
   f=branch_features(branches[ds],tr,va)
   r_archive[ds]=f["validation"]["R"].copy()
   var_e=float(np.var(f["validation"]["E"],axis=0,dtype=np.float64).mean())
   var_z=float(np.var(va[:,128:],axis=0,dtype=np.float64).mean())
   var_ratios[ds]=var_e/max(var_z,1e-12)
   nonlinear=nonlinear_mlp_r2(va[:,:128],f["validation"]["R"],seed=42+di)
   ntr,nva=random_null_encoder(f["train"]["E"],f["validation"]["E"],4242+di)
   gtr=gaussian[ds]["train"]; gva=gaussian[ds]["validation"]
   per_ds[ds]={}; pred_archive[f"{arm}_{ds}_P0_std"]=p0[ds][True]; pred_archive[f"{arm}_{ds}_P1_std"]=p1[ds][True]
   for standard in (True,False):
    tag="standardized" if standard else "unstandardized"
    probes={
     "P2":({"Z_s":tr[:,:128],"E_d":f["train"]["E"]},{"Z_s":va[:,:128],"E_d":f["validation"]["E"]}),
     "P3":({"Z_s":tr[:,:128],"R":f["train"]["R"]},{"Z_s":va[:,:128],"R":f["validation"]["R"]}),
     "NullN":({"Z_s":tr[:,:128],"R_N":ntr},{"Z_s":va[:,:128],"R_N":nva}),
     "NullG":({"Z_s":tr[:,:128],"G":gtr},{"Z_s":va[:,:128],"G":gva}),
    }
    per_ds[ds][tag]={}
    for pname,(tb,vb) in probes.items():
     rr,pp=fit_probe(f"{arm}/{ds}/{pname}",tb,vb,baseline[ds]["y_train"],y_by_ds[ds],standard)
     per_ds[ds][tag][pname]=rr
     if standard:
      if pname=="P3": p3_preds[ds]=pp
      elif pname=="P2": p2_preds[ds]=pp
      elif pname=="NullN": nulln_preds[ds]=pp
      elif pname=="NullG": nullg_preds[ds]=pp
     pred_archive[f"{arm}_{ds}_{pname}_{tag}"]=pp
   per_ds[ds]["nonlinear_MLP_R_from_Zs"]=nonlinear
   del train,val,tr,va,f,ntr,nva; gc.collect()
  np.savez_compressed(OUT/f"{arm}_validation_R.npz",**{f"{d}_R":r_archive[d] for d in DATASETS})
  p3_ci=combined_bootstrap(y_by_ds,{d:p0[d][True] for d in DATASETS},p3_preds)
  n_ci=combined_bootstrap(y_by_ds,{d:p0[d][True] for d in DATASETS},nulln_preds)
  g_ci=combined_bootstrap(y_by_ds,{d:p0[d][True] for d in DATASETS},nullg_preds)
  p2_gain=float(np.mean([macro_f1_supported(y_by_ds[d],p2_preds[d],np.unique(y_by_ds[d]))-macro_f1_supported(y_by_ds[d],p0[d][True],np.unique(y_by_ds[d])) for d in DATASETS]))
  val=arm_meta["final_best_checkpoint_validation"]
  mean_lpred_info=float(np.mean([val[d]["validation_L_pred"]+val[d]["validation_L_info"] for d in DATASETS]))
  rec={d:float(val[d]["reconstruction_skill"]) for d in DATASETS}
  floors={d:int(val[d]["near_zero_dimensions"]) for d in DATASETS}
  ranks={d:float(val[d]["effective_rank_participation"]) for d in DATASETS}
  finite=bool(np.isfinite([*rec.values(),p3_ci["mean_gain"],p3_ci["ci95_lower"],p3_ci["ci95_upper"],n_ci["mean_gain"],g_ci["mean_gain"]]).all())
  recon=all(v>0 for v in rec.values()); floor_ok=all(v<=32 for v in floors.values()); var_ratio_ok=all(v>=0.001 for v in var_ratios.values())
  above_null=p3_ci["ci95_lower"]>max(n_ci["mean_gain"],g_ci["mean_gain"])
  qualifies=finite and ceiling_ok and recon and floor_ok and var_ratio_ok and p3_ci["mean_gain"]>=0.01 and p3_ci["ci95_lower"]>0 and above_null
  summary={"arm":arm,"candidate":arm_meta["candidate"],"lambda_red":arm_meta["candidate_parameters"]["lambda_red"],"lambda_decorr":arm_meta["lambda_decorr"],"seed":arm_meta["seed"],"probe_standardized":per_ds,"p3_gain_bootstrap":p3_ci,"nullN_gain_bootstrap":n_ci,"nullG_gain_bootstrap":g_ci,"p2_mean_gain":p2_gain,"mean_validation_Lpred_plus_Linfo":mean_lpred_info,"reconstruction_skill":rec,"variance_floor_dimensions":floors,"var_E_over_var_Zd":var_ratios,"effective_rank_participation_reported_nonblocking":ranks,"gates":{"finite":finite,"P1_ceiling_passed":ceiling_ok,"positive_reconstruction_skill_both":recon,"variance_floor_at_most25pct":floor_ok,"Var_E_over_Var_Zd_at_least_0.001":var_ratio_ok,"p3_mean_gain_at_least_0.01":p3_ci["mean_gain"]>=0.01,"p3_ci_excludes_zero":p3_ci["ci95_lower"]>0,"p3_ci_lower_above_both_null_mean_gains":above_null},"qualifies_for_user_approved_selection":qualifies,"seed_sd_assessed":False}
  # Keep unstandardized probe metrics too, in a separate field.
  summary["probe_unstandardized"]={d:per_ds[d]["unstandardized"] for d in DATASETS}
  results.append(summary); np.savez_compressed(OUT/f"{arm}_probe_predictions.npz",**{k:v for k,v in pred_archive.items() if k.startswith(arm+"_")})
  write_json(OUT/"phase2_probe_partial_results.json",{"p0_p1_reused_baselines":base_results,"p1_ceiling":ceiling,"arms":results})
  log(f"ARM_PROBE_COMPLETE {arm} P3_gain={p3_ci['mean_gain']:.6f} CI=[{p3_ci['ci95_lower']:.6f},{p3_ci['ci95_upper']:.6f}] nullN={n_ci['mean_gain']:.6f} nullG={g_ci['mean_gain']:.6f} qualifies={qualifies}")
  del branches,ckpt; gc.collect(); torch.cuda.empty_cache()
 old_path=Path(r"C:\NOVEL_IDS_runs\stage6\phase2_decorr_20261125\candidate09_lambda0.1_seed42\best.pt")
 old_ck=torch.load(old_path,map_location="cpu",weights_only=False)
 if old_ck.get("stage5_checkpoint_sha256")!=manifest["stage5_snapshot_sha256"]: raise RuntimeError("Legacy arm-1 checkpoint Stage5 hash mismatch")
 old_branches=torch.nn.ModuleDict({d:Stage6ResidualBranch(128,128,128,128,128,128) for d in DATASETS})
 old_branches.load_state_dict(old_ck["branches"],strict=True); old_branches.to(DEVICE).eval()
 old_p3={}
 for ds in DATASETS:
  train=np.load(CACHE/ds/"train_zszd.npy",mmap_mode="r"); val=np.load(CACHE/ds/"validation_zszd.npy",mmap_mode="r")
  tr=np.array(train[:,0,:],dtype=np.float32,copy=True); va=np.array(val[:,:],dtype=np.float32,copy=True)
  fo=branch_features(old_branches[ds],tr,va)
  ro,po=fit_probe(f"legacy_arm1/{ds}/P3",{"Z_s":tr[:,:128],"R":fo["train"]["R"]},{"Z_s":va[:,:128],"R":fo["validation"]["R"]},baseline[ds]["y_train"],y_by_ds[ds],True)
  old_p3[ds]=po
  del train,val,tr,va,fo; gc.collect()
 new_arm="candidate09_lambda0.1_seed42"
 new_p3={ds:pred_archive[f"{new_arm}_{ds}_P3_standardized"] for ds in DATASETS}
 old_gain=combined_bootstrap(y_by_ds,{d:p0[d][True] for d in DATASETS},old_p3)
 new_gain=combined_bootstrap(y_by_ds,{d:p0[d][True] for d in DATASETS},new_p3)
 diff_gain=combined_bootstrap(y_by_ds,old_p3,new_p3)
 overlap=max(old_gain["ci95_lower"],new_gain["ci95_lower"])<=min(old_gain["ci95_upper"],new_gain["ci95_upper"])
 old_comparison={"legacy_checkpoint":str(old_path),"legacy_checkpoint_sha256":sha256_file(str(old_path)),"legacy_global_step":int(old_ck["global_step"]),"legacy_P3_gain_vs_P0":old_gain,"rerun_P3_gain_vs_P0":new_gain,"paired_rerun_minus_legacy_gain":diff_gain,"marginal_CIs_overlap":bool(overlap),"paired_difference_CI_excludes_zero":bool(diff_gain["ci95_lower"]>0 or diff_gain["ci95_upper"]<0),"interpretation":"paired CI determines whether the rerun's P3 gain differs from legacy arm 1; marginal CI overlap is descriptive only"}
 write_json(OUT/"old_arm1_rerun_comparison.json",old_comparison)
 log(f"OLD_ARM1_RERUN_COMPARISON {json.dumps(old_comparison,sort_keys=True)}")
 del old_branches,old_ck; gc.collect(); torch.cuda.empty_cache()
 eligible=sorted([x for x in results if x["qualifies_for_user_approved_selection"]],key=lambda x:(-x["p3_gain_bootstrap"]["mean_gain"],x["mean_validation_Lpred_plus_Linfo"],x["arm"]))
 unique_groups=[]
 for item in eligible:
  cur=np.load(OUT/f"{item['arm']}_validation_R.npz")
  duplicate=None; maxdiff=None
  for group in unique_groups:
   rep=np.load(OUT/f"{group['representative']}_validation_R.npz")
   diff=max(float(np.max(np.abs(cur[f"{d}_R"]-rep[f"{d}_R"]))) for d in DATASETS)
   if diff<=1e-6:
    duplicate=group; maxdiff=diff; break
  if duplicate is None:
   unique_groups.append({"representative":item["arm"],"duplicate_arms":[],"max_abs_diff_to_representative":0.0})
  else:
   duplicate["duplicate_arms"].append(item["arm"])
   duplicate["max_abs_diff_to_representative"]=max(duplicate["max_abs_diff_to_representative"],float(maxdiff))
 finalists=unique_groups[:2]
 selection={"p1_ceiling":ceiling,"p1_ceiling_passed":ceiling_ok,"eligible_arms":[x["arm"] for x in eligible],"unique_R_groups":unique_groups,"finalist_arms":[x["representative"] for x in finalists],"selected_arm":finalists[0]["representative"] if finalists else None,"selection_tie_breaker":"highest mean P3 gain; ties use lower mean validation L_pred+L_info","duplicate_R_tolerance_max_abs":1e-6,"seed_sd_assessed":False,"rank_gate_policy":"reported, non-blocking per latest user instruction","selection_status":"TWO_UNIQUE_FINALISTS_SELECTED_PENDING_SEED_SD" if len(finalists)>=2 else ("ONE_UNIQUE_FINALIST" if finalists else "NO_QUALIFYING_STAGE6_ARM")}
 write_json(OUT/"phase2_probe_results.json",{"p0_p1_reused_baselines":base_results,"p1_ceiling":ceiling,"arms":results,"selection":selection})
 write_json(OUT/"phase2_selection.json",selection)
 print("PROBE_EVALUATION_COMPLETE",json.dumps(selection,indent=2),flush=True)
 return 0
if __name__=="__main__": raise SystemExit(main())






