from __future__ import annotations
import json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable
CACHE=Path(r"C:\NOVEL_IDS_runs\phaseB_input_cache_20261126")
RUNNER=ROOT/"scripts"/"run_stage7_11_experiment.py"
PILOT_LOG=Path(r"C:\NOVEL_IDS_runs\phaseB_training_orchestration.log")
LOG=Path(r"C:\NOVEL_IDS_runs\phaseB_final_training.log")
OUT=Path(r"C:\NOVEL_IDS_runs")
VARIANTS=("shared_only","residual_fixed_gate1","residual_adaptive","common_only")
SEEDS=(42,43)
def log(msg):
 line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
 print(line,flush=True)
 with LOG.open("a",encoding="utf-8") as f:f.write(line+"\n")
def resources():
 import ctypes,torch
 class M(ctypes.Structure):
  _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
 m=M();m.dwLength=ctypes.sizeof(M)
 if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError("GlobalMemoryStatusEx failed")
 ram=m.ullAvailPhys/1024**3
 disk=shutil.disk_usage("C:/").free/1024**3
 free,total=torch.cuda.mem_get_info()
 if ram<2 or disk<10 or free<5*1024**3:
  raise RuntimeError(f"HARD STOP resources RAM={ram:.2f}GB disk={disk:.2f}GB GPUfree={free/1024**3:.2f}GB")
 return ram,disk,free/1024**3
def completed(run_dir):
 p=run_dir/"progress.log"
 return p.exists() and "RUN_COMPLETE" in p.read_text(encoding="utf-8",errors="ignore")
def execute(cmd,run_dir,timeout=40*60):
 t=time.time()
 log("COMMAND_START "+" ".join(map(str,cmd)))
 p=subprocess.run(cmd,cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT/"src")},timeout=timeout)
 log(f"COMMAND_END returncode={p.returncode} elapsed_s={time.time()-t:.1f}")
 if p.returncode: raise RuntimeError(f"Run failed rc={p.returncode}: {cmd}")
 if not completed(run_dir): raise RuntimeError(f"Run has no completion marker: {run_dir}")
def run_with_retry(cmd,run_dir,variant,seed,steps,interval,batch):
 target=run_dir
 if completed(target):
  log(f"REUSE_COMPLETED_FINAL path={target}"); return
 if target.exists():
  for attempt in range(1,4):
   candidate=Path(str(run_dir)+f"_retry{attempt}")
   if completed(candidate):
    log(f"REUSE_COMPLETED_RETRY path={candidate}"); return
   if not candidate.exists():
    target=candidate; cmd[cmd.index("--run-dir")+1]=str(target); break
  else: raise RuntimeError(f"Three incomplete attempts already exist: {run_dir}")
 try:
  execute(cmd,target)
 except Exception as e:
  log(f"RETRY_ONCE cause={type(e).__name__}:{e}")
  retry_cmd=cmd.copy(); retry_dir=None
  for attempt in range(1,4):
   candidate=Path(str(run_dir)+f"_retry{attempt}")
   if completed(candidate):
    log(f"REUSE_COMPLETED_RETRY path={candidate}"); return
   if not candidate.exists(): retry_dir=candidate; break
  if retry_dir is None: raise RuntimeError(f"No free retry directory for {run_dir}")
  retry_cmd[retry_cmd.index("--run-dir")+1]=str(retry_dir)
  if "OutOfMemory" in str(e) or "out of memory" in str(e).lower():
   retry_cmd[retry_cmd.index("--batch-per-dataset")+1]=str(max(32,batch//2))
   log(f"OOM_RETRY_BATCH_REDUCED from={batch} to={max(32,batch//2)}")
  execute(retry_cmd,retry_dir)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 start=time.time()
 # Wait for the pilot matrix to finish; no overlapping training jobs.
 while not PILOT_LOG.exists() or "PHASEB_TRAINING_COMPLETE" not in PILOT_LOG.read_text(encoding="utf-8",errors="ignore"):
  log("WAITING_FOR_1000_STEP_PILOTS_TO_FINISH")
  time.sleep(60)
 if not (CACHE/"cache_manifest.json").exists(): raise RuntimeError("Verified input cache manifest missing")
 manifest=json.loads((CACHE/"cache_manifest.json").read_text(encoding="utf-8"))
 if manifest.get("status")!="VERIFIED": raise RuntimeError("Input cache not verified")
 cache_hash=__import__("hashlib").sha256((CACHE/"cache_manifest.json").read_bytes()).hexdigest()
 log(f"PHASEB_FINAL_START cache_sha256={cache_hash} steps=4000 validation_interval=500 seeds={SEEDS}")
 for variant in VARIANTS:
  for seed in SEEDS:
   ram,disk,gpu=resources()
   smoke=OUT/f"phaseB_{variant}_seed{seed}_smoke"
   if not completed(smoke):
    smcmd=[PY,str(RUNNER),"--seed",str(seed),"--lambda-coral","0.05","--batch-per-dataset","256",
      "--ablation",variant,"--cache-root",str(CACHE),"--max-steps","100","--validation-interval","100","--run-dir",str(smoke)]
    run_with_retry(smcmd,smoke,variant,seed,100,100,256)
   final=OUT/f"phaseB_final_{variant}_seed{seed}"
   if completed(final):
    log(f"REUSE_EXISTING_FINAL variant={variant} seed={seed} path={final}")
    continue
   cmd=[PY,str(RUNNER),"--seed",str(seed),"--lambda-coral","0.05","--batch-per-dataset","256",
      "--ablation",variant,"--cache-root",str(CACHE),"--max-steps","4000","--validation-interval","500","--run-dir",str(final)]
   run_with_retry(cmd,final,variant,seed,5000,1000,256)
   log(f"FINAL_ABLATION_COMPLETE variant={variant} seed={seed} ram_start={ram:.2f} disk_start={disk:.2f} gpu_free_start={gpu:.2f} path={final}")
 # Paired native-head ablation on the same adaptive-residual/no-alignment path;
 # compare against the already completed unified-head residual_adaptive runs.
 for seed in SEEDS:
  ram,disk,gpu=resources()
  smoke=OUT/f"phaseB_residual_adaptive_native_seed{seed}_smoke"
  if not completed(smoke):
   smcmd=[PY,str(RUNNER),"--seed",str(seed),"--lambda-coral","0.05","--batch-per-dataset","256",
     "--ablation","residual_adaptive","--head-mode","native","--cache-root",str(CACHE),
     "--max-steps","100","--validation-interval","100","--run-dir",str(smoke)]
   run_with_retry(smcmd,smoke,"residual_adaptive_native",seed,100,100,256)
  final=OUT/f"phaseB_final_residual_adaptive_native_seed{seed}"
  if completed(final):
   log(f"REUSE_EXISTING_NATIVE_HEAD variant=residual_adaptive seed={seed} path={final}")
   continue
  cmd=[PY,str(RUNNER),"--seed",str(seed),"--lambda-coral","0.05","--batch-per-dataset","256",
    "--ablation","residual_adaptive","--head-mode","native","--cache-root",str(CACHE),
    "--max-steps","4000","--validation-interval","500","--run-dir",str(final)]
  run_with_retry(cmd,final,"residual_adaptive_native",seed,5000,1000,256)
  log(f"NATIVE_HEAD_ABLATION_COMPLETE seed={seed} ram_start={ram:.2f} disk_start={disk:.2f} gpu_free_start={gpu:.2f} path={final}")
 log("ALIGNMENT_OFF=final_residual_adaptive; ALIGNMENT_ON=primary_seed43_lambda_coral_0p05")
 log("NATIVE_VS_UNIFIED_HEAD=paired residual_adaptive Transformer runs; native head runs use per-dataset native labels; unified head runs use approved unified taxonomy.")
 log(f"PHASEB_FINAL_COMPLETE elapsed_min={(time.time()-start)/60:.1f}")
if __name__=="__main__":main()


