from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable
CACHE=Path(r"C:\NOVEL_IDS_runs\phaseB_input_cache_20261126")
RUNNER=ROOT/"scripts"/"run_stage7_11_experiment.py"
FINALIZER=ROOT/"scripts"/"finalize_stage7_11_input_cache.py"
OUT=Path(r"C:\NOVEL_IDS_runs")
LOG=OUT/"phaseB_training_orchestration.log"
VARIANTS=("shared_only","residual_fixed_gate1","residual_adaptive","common_only")
SEEDS=(42,43)
def log(msg):
 line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
 print(line,flush=True)
 with LOG.open("a",encoding="utf-8") as f:f.write(line+"\n")
def run(cmd,timeout):
 run_arg=cmd.index("--run-dir")+1
 requested=Path(cmd[run_arg])
 if requested.exists():
  marker=requested/"progress.log"
  if marker.exists() and "RUN_COMPLETE" in marker.read_text(encoding="utf-8",errors="ignore"):
   log(f"REUSE_COMPLETED_RUN path={requested}"); return
  for attempt in range(1,4):
   candidate=Path(str(requested)+f"_retry{attempt}")
   if not candidate.exists(): requested=candidate; cmd[run_arg]=str(candidate); break
  else: raise RuntimeError(f"Three prior invocation directories exist for {requested}")
 log("COMMAND_START "+" ".join(map(str,cmd)))
 t=time.time()
 p=subprocess.run(cmd,cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT/"src")},timeout=timeout)
 log(f"COMMAND_END returncode={p.returncode} elapsed_s={time.time()-t:.1f}")
 if p.returncode: raise RuntimeError(f"Command failed: {cmd}")
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if not (CACHE/"cache_manifest.json").exists():
  log("FINALIZE_CACHE_START")
  run([PY,str(FINALIZER)],timeout=3600)
 manifest=json.loads((CACHE/"cache_manifest.json").read_text(encoding="utf-8"))
 if manifest.get("status")!="VERIFIED": raise RuntimeError("Cache manifest is not verified")
 cache_hash=__import__("hashlib").sha256((CACHE/"cache_manifest.json").read_bytes()).hexdigest()
 log(f"PHASEB_START cache_manifest_sha256={cache_hash} variants={VARIANTS} seeds={SEEDS}")
 for variant in VARIANTS:
  for seed in SEEDS:
   smoke=OUT/f"phaseB_{variant}_seed{seed}_smoke"
   full=OUT/f"phaseB_{variant}_seed{seed}"
   common=[PY,str(RUNNER),"--seed",str(seed),"--lambda-coral","0.05",
     "--batch-per-dataset","256","--ablation",variant,"--cache-root",str(CACHE)]
   run(common+["--max-steps","100","--validation-interval","100","--run-dir",str(smoke)],timeout=40*60)
   run(common+["--max-steps","1000","--validation-interval","1000","--run-dir",str(full)],timeout=40*60)
   log(f"ABLATION_COMPLETE variant={variant} seed={seed} run_dir={full}")
 log("ALIGNMENT_OFF_REUSE residual_adaptive; ALIGNMENT_ON_REUSE primary seed43 lambda_coral0.05")
 log("NATIVE_VS_UNIFIED_HEAD evidence: completed HGB/MLP native/unified baseline modes; no same-Transformer native-head run implemented in this phase.")
 log("PHASEB_TRAINING_COMPLETE")
if __name__=="__main__":main()
