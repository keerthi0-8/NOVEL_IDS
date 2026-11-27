from __future__ import annotations
import gc, json, time, traceback
from pathlib import Path
import numpy as np, torch, yaml
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import run_stage6_fastpath_training as fast
from novel_ids.residual.stage6_training import Stage6Trainer
from novel_ids.residual.stage6_cached import attach_stage6_cache, release_stage6_source_arrays
OUT=fast.OUT
def resume_first_arm(raw,plan):
 name="candidate09"; lam=0.0; seed=42; candidate=fast.CANDIDATES[name]
 arm=f"{name}_lambda{lam:g}_seed{seed}"; run_dir=OUT/arm
 last=run_dir/"last.pt"
 if not last.is_file(): raise FileNotFoundError(last)
 ck=torch.load(last,map_location="cpu",weights_only=False)
 start_step=int(ck["global_step"])
 if start_step!=1000: raise RuntimeError(f"Expected checkpoint at step 1000, got {start_step}")
 if ck["config"]["lambda_decorr"]!=lam or ck["config"]["seed"]!=seed: raise RuntimeError("Resume checkpoint config mismatch")
 fast.verify_protected()
 cfg=fast.make_config(raw,candidate,seed,lam,plan)
 bootstrap_dir=run_dir/"resume_bootstrap"
 trainer=Stage6Trainer.from_project(stage5_snapshot=fast.SNAP,expected_stage5_sha256="9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b",run_dir=bootstrap_dir,config=cfg)
 attach_stage6_cache(trainer,fast.CACHE); counts=release_stage6_source_arrays(trainer)
 trainer.run_dir=run_dir
 trainer.branches.load_state_dict(ck["branches"],strict=True)
 trainer.optimizer.load_state_dict(ck["optimizer"])
 trainer.global_step=start_step
 trainer.loss_history=ck["loss_history"]
 trainer.validation_history=ck["validation_history"]
 trainer.best_validation_objective=float(ck["best_validation_objective"])
 trainer.best_step=int(ck["best_step"])
 trainer.non_improving_evaluations=0
 # Replay the exact row-index and per-row cache-draw RNG consumption of the
 # first 1000 train steps; RNG state was not part of the original checkpoint.
 for _ in range(start_step):
  for ds in ("nsl_kdd","unsw_nb15"):
   n=counts[ds]["train"]
   trainer.rng.choice(n,size=cfg.sub_batch_size,replace=False)
   trainer._stage6_cache_rng[ds].integers(0,8,size=cfg.sub_batch_size,dtype=np.int64)
 trainer._stage6_cache_force_clean=False
 trainer.stage5.eval(); torch.cuda.reset_peak_memory_stats()
 warmup=max(1,int(round(cfg.max_steps*cfg.warmup_fraction)))
 best=trainer.best_validation_objective; best_step=trainer.best_step
 nonimproving=0; validations=list(trainer.validation_history); start=time.perf_counter(); last_log=start
 fast.log(f"ARM_RESUME {arm} from_step={start_step} rng_replay_steps={start_step} max_steps={cfg.max_steps} validation_interval={cfg.validation_interval} best_objective={best:.8f} free_ram_gb={fast.free_ram_gb():.2f}")
 for _ in range(start_step,cfg.max_steps):
  metrics=trainer.train_step(); step=trainer.global_step; now=time.perf_counter()
  if now-last_log>=60:
   sec=(now-start)/(max(step-start_step,1)); eta=sec*max(cfg.max_steps-step,0)
   fast.log(f"ARM_PROGRESS {arm} step={step}/{cfg.max_steps} elapsed_resume_s={now-start:.1f} sec_per_step={sec:.4f} eta_s={eta:.1f} loss={metrics['combined_total']:.7f} free_ram_gb={fast.free_ram_gb():.2f} gpu_alloc_mib={torch.cuda.memory_allocated()/1024**2:.1f} gpu_peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}")
   last_log=now
  if torch.cuda.max_memory_allocated()>6*1024**3: raise RuntimeError("HARD STOP: GPU allocation exceeded 6 GiB")
  if step%cfg.validation_interval==0 and step>=warmup:
   trainer._stage6_cache_force_clean=True
   try: val=trainer.validate()
   finally: trainer._stage6_cache_force_clean=False
   val["step"]=step; validations.append(val); trainer.validation_history.append(val)
   objective=float(val["mean_validation_objective"])
   improved=objective<=best*(1.0-cfg.min_relative_improvement)
   if improved:
    best=objective; best_step=step; nonimproving=0; trainer.best_validation_objective=best; trainer.best_step=step; trainer.save_checkpoint(best=True)
   else: nonimproving+=1
   trainer.non_improving_evaluations=nonimproving
   trainer.save_checkpoint(best=False)
   fast.log(f"ARM_VALIDATION {arm} step={step} objective={objective:.8f} best={best:.8f} best_step={best_step} nonimproving={nonimproving}/{cfg.early_stop_patience} nsl_skill={val['nsl_kdd']['reconstruction_skill']:.6f} unsw_skill={val['unsw_nb15']['reconstruction_skill']:.6f} free_ram_gb={fast.free_ram_gb():.2f} gpu_peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}")
   if nonimproving>=cfg.early_stop_patience: break
 trainer.save_checkpoint(best=False)
 best_ck=torch.load(run_dir/"best.pt",map_location="cpu",weights_only=False)
 trainer.branches.load_state_dict(best_ck["branches"],strict=True); trainer.branches.eval()
 trainer._stage6_cache_force_clean=True
 try: final_val=trainer.validate()
 finally: trainer._stage6_cache_force_clean=False
 elapsed=time.perf_counter()-start
 result={"arm":arm,"candidate":name,"candidate_parameters":candidate,"lambda_decorr":lam,"seed":seed,"resumed_from_step":start_step,"rng_replay_steps":start_step,"steps_completed":trainer.global_step,"best_step":best_step,"best_validation_objective":best,"validation_history":validations,"final_best_checkpoint_validation":final_val,"checkpoint_sha256":fast.sha256_file(run_dir/"best.pt"),"stage5_snapshot_sha256":fast.sha256_file(fast.SNAP),"max_steps":cfg.max_steps,"validation_interval":cfg.validation_interval,"early_stop_patience":cfg.early_stop_patience,"resume_seconds":elapsed,"gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),"free_ram_gb_after":fast.free_ram_gb(),"cache_policy":trainer._stage6_cache_draw_policy}
 fast.write_json(run_dir/"fastpath_arm_summary.json",result)
 del trainer,ck,best_ck; gc.collect(); torch.cuda.empty_cache(); fast.verify_protected()
 fast.log(f"ARM_COMPLETE {arm} steps={result['steps_completed']} best_step={best_step} resume_seconds={elapsed:.1f} checkpoint_sha256={result['checkpoint_sha256']} free_ram_gb={fast.free_ram_gb():.2f}")
 return result
def main():
 if not OUT.is_dir(): raise FileNotFoundError(OUT)
 manifest=json.loads((fast.CACHE/"cache_manifest.json").read_text(encoding="utf-8"))
 if manifest.get("status")!="CACHE_EQUIVALENCE_PASSED": raise RuntimeError("Cache manifest not passed")
 fast.verify_protected()
 if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
 if torch.cuda.mem_get_info()[0]<5*1024**3: raise RuntimeError("GPU free memory below 5 GiB")
 if fast.free_ram_gb()<3: raise RuntimeError(f"Sequential resume requires at least 3 GB free RAM: {fast.free_ram_gb():.2f}")
 torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 plan=fast.read_plan(); raw=yaml.safe_load((ROOT/"config"/"stage6_projection.yaml").read_text(encoding="utf-8"))
 results=[resume_first_arm(raw,plan)]
 fast.write_json(OUT/"phase2_partial_results.json",{"seed42_arms":results,"plan":plan,"resume_note":"candidate09 lambda0 resumed from step1000 after exact NumPy RNG-consumption replay"})
 remaining=[("candidate09",0.1),("candidate09",1.0),("candidate07",0.0),("candidate07",0.1),("candidate07",1.0)]
 for name,lam in remaining:
  result=fast.run_arm(raw,name,fast.CANDIDATES[name],lam,42,plan); results.append(result)
  fast.write_json(OUT/"phase2_partial_results.json",{"seed42_arms":results,"plan":plan})
 fast.write_json(OUT/"phase2_training_results.json",{"seed42_arms":results,"plan":plan,"probe_evaluation_pending":True})
 fast.log("SEED42_SIX_ARMS_COMPLETE")
 print("FASTPATH_STAGE6_TRAINING_COMPLETE",json.dumps({"arms":len(results),"output":str(OUT),"plan":plan},indent=2),flush=True)
 return 0
if __name__=="__main__":
 try: raise SystemExit(main())
 except Exception:
  fast.log("RESUME_FASTPATH_STOPPED "+traceback.format_exc()); raise
