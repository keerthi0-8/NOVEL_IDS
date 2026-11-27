from __future__ import annotations
import ctypes, gc, json, math, re, shutil, time, traceback
from pathlib import Path
import numpy as np, torch, yaml
from novel_ids.residual.stage6_training import Stage6Trainer, Stage6TrainingConfig, STAGE5_PROTECTED_HASHES, sha256_file
from novel_ids.residual.stage6_cached import attach_stage6_cache, release_stage6_source_arrays
ROOT=Path(__file__).resolve().parents[1]
SNAP=Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
CACHE=Path(r"C:\NOVEL_IDS_runs\stage6\phase2_fastcache_20261125")
OUT=Path(r"C:\NOVEL_IDS_runs\stage6\phase2_fastpath_20261125")
SMOKE_LOG=Path(r"C:\NOVEL_IDS_runs\stage6\fastpath_smoke_memory_released_20261125\smoke_progress.log")
LOG=OUT/"fastpath_progress.log"
CANDIDATES={"candidate09":{"residual_dim":128,"projection_mode":"identity","lambda_red":1.0},"candidate07":{"residual_dim":128,"projection_mode":"identity","lambda_red":0.01}}
LAMBDAS=(0.0,0.1,1.0)
def free_ram_gb():
 class M(ctypes.Structure):
  _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
 m=M(); m.dwLength=ctypes.sizeof(M)
 if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)): raise OSError("GlobalMemoryStatusEx failed")
 return m.ullAvailPhys/1024**3
def log(msg):
 line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
 print(line,flush=True); LOG.parent.mkdir(parents=True,exist_ok=True)
 with LOG.open("a",encoding="utf-8") as f:f.write(line+"\n")
def write_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str),encoding="utf-8"); tmp.replace(path)
def verify_protected():
 bad=[]
 for rel,expected in STAGE5_PROTECTED_HASHES.items():
  actual=sha256_file(ROOT/rel)
  if actual.lower()!=expected.lower(): bad.append((rel,expected,actual))
 if bad: raise RuntimeError(f"Protected Stage5 hash mismatch: {bad}")
 if sha256_file(SNAP)!= "9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b": raise RuntimeError("Stage5 snapshot hash mismatch")
def read_plan():
 lines=SMOKE_LOG.read_text(encoding="utf-8").splitlines()
 entries=[x.split("SMOKE_COMPLETE ",1)[1] for x in lines if "SMOKE_COMPLETE " in x]
 if not entries: raise RuntimeError("No completed 100-step smoke result found.")
 result=json.loads(entries[-1])
 step_s=float(result["seconds_per_step"]); val_s=float(result["validation_seconds"])
 projected=step_s*10000+val_s*20
 if projected<=7200:
  return {"max_steps":10000,"validation_interval":500,"projected_10000_step_seconds":projected,"reduced":False,"seconds_per_step":step_s,"validation_seconds":val_s}
 interval=1000
 max_steps=min(10000,int(6900/max(step_s+val_s/interval,1e-9)))
 if max_steps<1000: raise RuntimeError(f"Projected step time cannot fit even 1000 steps in 2 hours: {step_s}")
 return {"max_steps":max_steps,"validation_interval":interval,"projected_10000_step_seconds":projected,"reduced":True,"seconds_per_step":step_s,"validation_seconds":val_s}
def make_config(raw,candidate,seed,lam,plan):
 t=raw["training"]; loss=t["loss_coefficients"]
 return Stage6TrainingConfig(residual_dim=candidate["residual_dim"],lambda_red=candidate["lambda_red"],variance_floor=float(t["variance_floor"]),projection_mode=candidate["projection_mode"],
  sub_batch_size=256,validation_batch_size=512,max_steps=plan["max_steps"],validation_interval=plan["validation_interval"],early_stop_patience=int(t["early_stop_patience"]),
  min_relative_improvement=float(t["min_relative_improvement"]),warmup_fraction=float(t["warmup_fraction"]),lambda_pred=float(loss["lambda_pred"]),lambda_info=float(loss["lambda_info"]),
  lambda_mag=float(loss["lambda_mag"]),lambda_proj=float(t.get("projection_experiment",{}).get("coefficient",1.0)) if isinstance(t.get("projection_experiment"),dict) else 1.0,
  decorr_enabled=True,lambda_decorr=float(lam),seed=int(seed),device="cuda")
def run_arm(raw,name,candidate,lam,seed,plan):
 arm=f"{name}_lambda{lam:g}_seed{seed}"; run_dir=OUT/arm
 if run_dir.exists(): raise FileExistsError(run_dir)
 verify_protected()
 while free_ram_gb()<3.0:
  log(f"RESOURCE_WAIT arm={arm} free_ram_gb={free_ram_gb():.2f}; waiting 60s before next sequential arm")
  time.sleep(60)
 torch.cuda.reset_peak_memory_stats()
 cfg=make_config(raw,candidate,seed,lam,plan)
 trainer=Stage6Trainer.from_project(stage5_snapshot=SNAP,expected_stage5_sha256="9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b",run_dir=run_dir,config=cfg)
 attach_stage6_cache(trainer,CACHE); release_stage6_source_arrays(trainer)
 trainer.stage5.eval()
 start=time.perf_counter(); last_log=start; warmup=max(1,int(round(cfg.max_steps*cfg.warmup_fraction)))
 best=float("inf"); best_step=None; nonimproving=0; validations=[]; last_metrics={}
 log(f"ARM_START {arm} max_steps={cfg.max_steps} validation_interval={cfg.validation_interval} warmup={warmup} lambda_red={cfg.lambda_red} lambda_decorr={lam} free_ram_gb={free_ram_gb():.2f} gpu_free_mib={torch.cuda.mem_get_info()[0]/1024**2:.0f}")
 for _ in range(cfg.max_steps):
  metrics=trainer.train_step(); last_metrics=metrics
  step=trainer.global_step; now=time.perf_counter()
  if now-last_log>=60:
   sec=(now-start)/max(step,1); eta=sec*max(cfg.max_steps-step,0)
   log(f"ARM_PROGRESS {arm} step={step}/{cfg.max_steps} elapsed_s={now-start:.1f} sec_per_step={sec:.4f} eta_s={eta:.1f} loss={metrics['combined_total']:.7f} free_ram_gb={free_ram_gb():.2f} gpu_alloc_mib={torch.cuda.memory_allocated()/1024**2:.1f} gpu_peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}")
   last_log=now
  if step%100==0:
   if torch.cuda.max_memory_allocated()>6*1024**3: raise RuntimeError("HARD STOP: GPU allocation exceeded 6 GiB")
  if step%cfg.validation_interval==0 and step>=warmup:
   trainer._stage6_cache_force_clean=True
   try: val=trainer.validate()
   finally: trainer._stage6_cache_force_clean=False
   val["step"]=step; validations.append(val); trainer.validation_history.append(val)
   objective=float(val["mean_validation_objective"])
   improved=(not np.isfinite(best)) or objective<=best*(1.0-cfg.min_relative_improvement)
   if improved:
    best=objective; best_step=step; nonimproving=0; trainer.best_validation_objective=best; trainer.best_step=step; trainer.save_checkpoint(best=True)
   else: nonimproving+=1
   trainer.non_improving_evaluations=nonimproving
   trainer.save_checkpoint(best=False)
   log(f"ARM_VALIDATION {arm} step={step} objective={objective:.8f} best={best:.8f} best_step={best_step} nonimproving={nonimproving}/{cfg.early_stop_patience} nsl_skill={val['nsl_kdd']['reconstruction_skill']:.6f} unsw_skill={val['unsw_nb15']['reconstruction_skill']:.6f} free_ram_gb={free_ram_gb():.2f} gpu_peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}")
   if nonimproving>=cfg.early_stop_patience: break
 elapsed=time.perf_counter()-start
 if not (run_dir/"best.pt").is_file(): raise RuntimeError(f"{arm}: no best checkpoint was saved")
 trainer.save_checkpoint(best=False)
 ck=torch.load(run_dir/"best.pt",map_location="cpu",weights_only=False)
 trainer.branches.load_state_dict(ck["branches"],strict=True); trainer.branches.eval()
 trainer._stage6_cache_force_clean=True
 try: final_val=trainer.validate()
 finally: trainer._stage6_cache_force_clean=False
 result={"arm":arm,"candidate":name,"candidate_parameters":candidate,"lambda_decorr":lam,"seed":seed,"steps_completed":trainer.global_step,"best_step":best_step,"best_validation_objective":best,"validation_history":validations,"final_best_checkpoint_validation":final_val,"checkpoint_sha256":sha256_file(run_dir/"best.pt"),"stage5_snapshot_sha256":sha256_file(SNAP),"max_steps":cfg.max_steps,"validation_interval":cfg.validation_interval,"early_stop_patience":cfg.early_stop_patience,"seconds":elapsed,"seconds_per_step":elapsed/max(trainer.global_step,1),"gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),"free_ram_gb_after":free_ram_gb(),"cache_policy":trainer._stage6_cache_draw_policy}
 write_json(run_dir/"fastpath_arm_summary.json",result)
 del trainer,ck; gc.collect(); torch.cuda.empty_cache(); verify_protected()
 log(f"ARM_COMPLETE {arm} steps={result['steps_completed']} best_step={best_step} seconds={elapsed:.1f} checkpoint_sha256={result['checkpoint_sha256']} free_ram_gb={free_ram_gb():.2f}")
 return result
def main():
 if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
 manifest=json.loads((CACHE/"cache_manifest.json").read_text(encoding="utf-8"))
 if manifest.get("status")!="CACHE_EQUIVALENCE_PASSED": raise RuntimeError("Cache equivalence manifest not passed.")
 verify_protected()
 if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable.")
 free,total=torch.cuda.mem_get_info()
 if free<5*1024**3: raise RuntimeError(f"GPU free memory below 5 GiB: {free}")
 if free_ram_gb()<4: raise RuntimeError(f"Free RAM below 4 GiB after imports: {free_ram_gb():.2f}")
 if shutil.disk_usage("C:\\").free<10*1024**3: raise RuntimeError("Disk free below 10 GiB.")
 torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 plan=read_plan(); raw=yaml.safe_load((ROOT/"config"/"stage6_projection.yaml").read_text(encoding="utf-8"))
 OUT.mkdir(parents=True,exist_ok=False)
 write_json(OUT/"phase2_fastpath_config.json",{"candidates":CANDIDATES,"lambda_decorr":LAMBDAS,"seeds":[42,43,44],"plan":plan,"cache_manifest":str(CACHE/"cache_manifest.json"),"stage5_snapshot_sha256":sha256_file(SNAP),"source_config":raw})
 log(f"FASTPATH_PHASE2_START plan={json.dumps(plan,sort_keys=True)} arms=6 sequential=True")
 results=[]
 for name,candidate in CANDIDATES.items():
  for lam in LAMBDAS:
   r=run_arm(raw,name,candidate,lam,42,plan); results.append(r)
   write_json(OUT/"phase2_partial_results.json",{"seed42_arms":results,"plan":plan})
 log("SEED42_SIX_ARMS_COMPLETE")
 write_json(OUT/"phase2_training_results.json",{"seed42_arms":results,"plan":plan,"probe_evaluation_pending":True})
 print("FASTPATH_STAGE6_TRAINING_COMPLETE",json.dumps({"arms":len(results),"output":str(OUT),"plan":plan},indent=2),flush=True)
 return 0
if __name__=="__main__":
 try: raise SystemExit(main())
 except Exception:
  log("FASTPATH_PHASE2_STOPPED "+traceback.format_exc()); raise






