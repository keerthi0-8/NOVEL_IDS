from __future__ import annotations
import json, os, time, shutil, ctypes
from pathlib import Path
import torch, yaml
from novel_ids.residual.stage6_training import Stage6Trainer, Stage6TrainingConfig, STAGE5_PROTECTED_HASHES, sha256_file
from novel_ids.residual.stage6_cached import attach_stage6_cache, release_stage6_source_arrays
ROOT=Path(__file__).resolve().parents[1]
SNAP=Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
CACHE=Path(r"C:\NOVEL_IDS_runs\stage6\phase2_fastcache_20261125")
RUN=Path(r"C:\NOVEL_IDS_runs\stage6\fastpath_smoke_memory_released_20261125\candidate09_lambda0.1_seed42")
LOG=RUN.parent/"smoke_progress.log"
def free_ram():
 class M(ctypes.Structure):
  _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
 m=M(); m.dwLength=ctypes.sizeof(M)
 if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)): raise OSError("GlobalMemoryStatusEx failed")
 return m.ullAvailPhys/(1024**3)
def log(msg):
 line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
 print(line,flush=True); LOG.parent.mkdir(parents=True,exist_ok=True)
 with LOG.open("a",encoding="utf-8") as f:f.write(line+"\n")
def main():
 if RUN.exists(): raise FileExistsError(RUN)
 manifest=json.loads((CACHE/"cache_manifest.json").read_text(encoding="utf-8"))
 if manifest.get("status")!="CACHE_EQUIVALENCE_PASSED": raise RuntimeError("Cache manifest is not approved.")
 if sha256_file(SNAP)!=manifest["stage5_snapshot_sha256"]: raise RuntimeError("Stage5 snapshot hash differs from cache manifest.")
 for rel,expected in STAGE5_PROTECTED_HASHES.items():
  actual=sha256_file(ROOT/rel)
  if actual.lower()!=expected.lower(): raise RuntimeError(f"Protected Stage5 hash mismatch: {rel}")
 if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable.")
 free,total=torch.cuda.mem_get_info()
 if free<5*1024**3: raise RuntimeError(f"GPU free memory below 5 GiB: {free}")
 if free_ram()<3: raise RuntimeError(f"Free RAM below 3 GiB for sequential cached smoke: {free_ram():.2f}")
 torch.set_num_threads(2); torch.set_num_interop_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
 raw=yaml.safe_load((ROOT/"config"/"stage6_projection.yaml").read_text(encoding="utf-8"))
 t=raw["training"]; loss=t["loss_coefficients"]
 cfg=Stage6TrainingConfig(residual_dim=128,lambda_red=1.0,variance_floor=float(t["variance_floor"]),projection_mode="identity",
  sub_batch_size=256,validation_batch_size=512,max_steps=10000,validation_interval=500,early_stop_patience=5,
  min_relative_improvement=0.005,warmup_fraction=0.10,lambda_pred=float(loss["lambda_pred"]),lambda_info=float(loss["lambda_info"]),
  lambda_mag=float(loss["lambda_mag"]),lambda_proj=float(loss["lambda_proj"]),decorr_enabled=True,lambda_decorr=0.1,seed=42,device="cuda")
 log("SMOKE_START steps=100 seed=42 candidate09 lambda_red=1.0 lambda_decorr=0.1")
 trainer=Stage6Trainer.from_project(stage5_snapshot=SNAP,expected_stage5_sha256=manifest["stage5_snapshot_sha256"],run_dir=RUN,config=cfg)
 attach_stage6_cache(trainer,CACHE/""); counts=release_stage6_source_arrays(trainer); log(f"SOURCE_ARRAYS_RELEASED counts={counts}")
 trainer.stage5.eval()
 log(f"TRAINER_READY free_ram_gb={free_ram():.2f} gpu_free_mib={torch.cuda.mem_get_info()[0]/1024**2:.0f} cache_policy={trainer._stage6_cache_draw_policy}")
 torch.cuda.reset_peak_memory_stats(); start=time.perf_counter(); last=start
 for step in range(1,101):
  metrics=trainer.train_step()
  now=time.perf_counter()
  if step%20==0 or now-last>=60:
   log(f"SMOKE_PROGRESS step={step}/100 elapsed_s={now-start:.2f} step_s={(now-start)/step:.4f} combined_loss={metrics['combined_total']:.7f} free_ram_gb={free_ram():.2f} gpu_alloc_mib={torch.cuda.memory_allocated()/1024**2:.1f} gpu_peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}")
   last=now
  if free_ram()<2: raise RuntimeError(f"HARD STOP: free RAM below 2 GB ({free_ram():.2f})")
  if torch.cuda.max_memory_allocated()>6*1024**3: raise RuntimeError("HARD STOP: GPU allocation exceeded 6 GiB")
 train_s=time.perf_counter()-start
 trainer._stage6_cache_force_clean=True
 vstart=time.perf_counter()
 try: validation=trainer.validate()
 finally: trainer._stage6_cache_force_clean=False
 val_s=time.perf_counter()-vstart
 step_s=train_s/100
 projected=step_s*10000+val_s*20
 result={"steps":100,"training_seconds":train_s,"seconds_per_step":step_s,"validation_seconds":val_s,"projected_10000_step_training_plus_20_validations_seconds":projected,"projected_hours":projected/3600,"validation_objective":validation["mean_validation_objective"],"gpu_peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),"free_ram_gb_after":free_ram(),"gpu_free_mib_after":torch.cuda.mem_get_info()[0]/1024**2,"cache_manifest":str(CACHE/"cache_manifest.json"),"cache_draw_policy":trainer._stage6_cache_draw_policy}
 log("SMOKE_COMPLETE "+json.dumps(result,sort_keys=True))
 print("SMOKE_RESULT",json.dumps(result,indent=2),flush=True)
 return 0
if __name__=="__main__": raise SystemExit(main())




