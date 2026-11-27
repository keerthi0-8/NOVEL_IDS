from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import evaluate_stage6_fastpath_probes as ev
from novel_ids.residual.stage6 import Stage6ResidualBranch
from novel_ids.residual.stage6_data import load_stage6_label_artifact
OUT=ev.OUT; CACHE=ev.CACHE; INPUT=ev.INPUT
def main():
 sel=json.loads((OUT/"phase2_selection.json").read_text(encoding="utf-8"))
 selected=sel.get("selected_arm")
 if not selected: raise RuntimeError("No selected seed42 arm for seed-repeat assessment.")
 seed_runs=json.loads((OUT/"seed_repeat_results.json").read_text(encoding="utf-8"))
 if not seed_runs.get("seed_sd_assessed"): raise RuntimeError("Seed43/44 runs were not completed.")
 training=json.loads((OUT/"phase2_training_results.json").read_text(encoding="utf-8"))
 meta=next(x for x in training["seed42_arms"] if x["arm"]==selected)
 name=meta["candidate"]; lam=float(meta["lambda_decorr"])
 results=[]; p3_gains=[]; nulln_means=[]; nullg_means=[]
 for seed in (43,44):
  arm=f"{name}_lambda{lam:g}_seed{seed}"; ck=torch.load(OUT/arm/"best.pt",map_location="cpu",weights_only=False)
  branches=torch.nn.ModuleDict({d:Stage6ResidualBranch(128,128,128,128,128,128) for d in ev.DATASETS})
  branches.load_state_dict(ck["branches"],strict=True); branches.to(ev.DEVICE).eval()
  y_by_ds={}; p0={}; p3={}; nnul={}; ngul={}; ds_summary={}
  for di,ds in enumerate(ev.DATASETS):
   trm=np.load(CACHE/ds/"train_zszd.npy",mmap_mode="r"); vam=np.load(CACHE/ds/"validation_zszd.npy",mmap_mode="r")
   tr=np.array(trm[:,0,:],dtype=np.float32,copy=True); va=np.array(vam[:,:],dtype=np.float32,copy=True)
   lab=load_stage6_label_artifact(INPUT/f"{ds}_labels.joblib",ds); yt=np.asarray(lab["train"]["labels"]).astype(str); yv=np.asarray(lab["validation"]["labels"]).astype(str); y_by_ds[ds]=yv
   f=ev.branch_features(branches[ds],tr,va)
   rp,pp=ev.fit_probe(f"{arm}/{ds}/P0",{"Z_s":tr[:,:128]},{"Z_s":va[:,:128]},yt,yv,True,seed=seed)
   rr,pr=ev.fit_probe(f"{arm}/{ds}/P3",{"Z_s":tr[:,:128],"R":f["train"]["R"]},{"Z_s":va[:,:128],"R":f["validation"]["R"]},yt,yv,True,seed=seed)
   rntr,rnva=ev.random_null_encoder(f["train"]["E"],f["validation"]["E"],4242+di)
   rn,nnp=ev.fit_probe(f"{arm}/{ds}/NullN",{"Z_s":tr[:,:128],"R_N":rntr},{"Z_s":va[:,:128],"R_N":rnva},yt,yv,True,seed=seed)
   rng=np.random.default_rng(4200+di); gtr=rng.standard_normal((len(yt),128),dtype=np.float32); gva=rng.standard_normal((len(yv),128),dtype=np.float32)
   rg,gnp=ev.fit_probe(f"{arm}/{ds}/NullG",{"Z_s":tr[:,:128],"G":gtr},{"Z_s":va[:,:128],"G":gva},yt,yv,True,seed=seed)
   p0[ds]=pp; p3[ds]=pr; nnul[ds]=nnp; ngul[ds]=gnp
   var_ratio=float(np.var(f["validation"]["E"],axis=0,dtype=np.float64).mean()/max(np.var(va[:,128:],axis=0,dtype=np.float64).mean(),1e-12))
   ds_summary[ds]={"P0":rp,"P3":rr,"NullN":rn,"NullG":rg,"var_E_over_var_Zd":var_ratio}
   del trm,vam,tr,va,lab,f,rntr,rnva,gtr,gva; gc.collect()
  p3ci=ev.combined_bootstrap(y_by_ds,p0,p3); nci=ev.combined_bootstrap(y_by_ds,p0,nnul); gci=ev.combined_bootstrap(y_by_ds,p0,ngul)
  p3_gains.append(p3ci["mean_gain"]); nulln_means.append(nci["mean_gain"]); nullg_means.append(gci["mean_gain"])
  results.append({"arm":arm,"seed":seed,"per_dataset":ds_summary,"P3_gain_bootstrap":p3ci,"NullN_gain_bootstrap":nci,"NullG_gain_bootstrap":gci})
  del branches,ck; gc.collect(); torch.cuda.empty_cache()
 seed42=next(x for x in json.loads((OUT/"phase2_probe_results.json").read_text(encoding="utf-8"))["arms"] if x["arm"]==selected)
 gains=[float(seed42["p3_gain_bootstrap"]["mean_gain"]),*p3_gains]
 seed_sd=float(np.std(gains,ddof=1)); mean_gain=float(np.mean(gains))
 nulln_mean=float(np.mean([seed42["nullN_gain_bootstrap"]["mean_gain"],*nulln_means]))
 nullg_mean=float(np.mean([seed42["nullG_gain_bootstrap"]["mean_gain"],*nullg_means]))
 p3_lower=float(seed42["p3_gain_bootstrap"]["ci95_lower"])
 passes_seed_sd=mean_gain>seed_sd
 passes_null=p3_lower>max(nulln_mean,nullg_mean)
 decision={"selected_seed42_arm":selected,"seed43_44_results":results,"P3_mean_gains_seed42_43_44":gains,"P3_mean_gain_across_seeds":mean_gain,"P3_seed_sample_SD":seed_sd,"passes_mean_gain_at_least_0.01":mean_gain>=0.01,"seed42_bootstrap_CI_excludes_zero":p3_lower>0,"mean_nullN_gain_across_seeds":nulln_mean,"mean_nullG_gain_across_seeds":nullg_mean,"seed42_P3_CI_lower_above_both_null_means":passes_null,"P3_mean_gain_exceeds_seed_SD":passes_seed_sd,"seed_sd_assessed":True,"status":"STAGE6_GATES_PASS" if mean_gain>=0.01 and p3_lower>0 and passes_null and passes_seed_sd else "STAGE6_GATES_NOT_ALL_PASS"}
 ev.write_json(OUT/"seed_repeat_probe_results.json",decision)
 selection["seed_sd_assessed"]=True; selection["seed_repeat_probe_decision"]=decision; selection["stage6_final_status"]=decision["status"]
 ev.write_json(OUT/"phase2_selection.json",selection)
 ev.log(f"SEED_REPEAT_PROBES_COMPLETE {json.dumps(decision,sort_keys=True)}")
 print("SEED_REPEAT_PROBES_COMPLETE",json.dumps(decision,indent=2),flush=True)
 return 0
if __name__=="__main__": raise SystemExit(main())
