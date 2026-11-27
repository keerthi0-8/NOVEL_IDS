from __future__ import annotations
import os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable
RUNNER=ROOT/"scripts"/"run_stage7_11_experiment.py"
CACHE=Path(r"C:\NOVEL_IDS_runs\phaseB_input_cache_20261126")
FINAL_LOG=Path(r"C:\NOVEL_IDS_runs\phaseB_final_training.log")
OUT=Path(r"C:\NOVEL_IDS_runs")
LOG=OUT/"phaseB_native_heads.log"
SEEDS=(42,43)
def log(msg):
 line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
 print(line,flush=True)
 with LOG.open("a",encoding="utf-8") as f:f.write(line+"\n")
def completed(path):
 p=path/"progress.log"
 return p.exists() and "RUN_COMPLETE" in p.read_text(encoding="utf-8",errors="ignore")
def run(cmd,path):
 target=path
 if completed(target):log(f"REUSE_COMPLETED {target}");return
 if target.exists():
  for i in range(1,4):
   q=Path(str(path)+f"_retry{i}")
   if completed(q):log(f"REUSE_COMPLETED_RETRY {q}");return
   if not q.exists():target=q;cmd[cmd.index("--run-dir")+1]=str(q);break
  else:raise RuntimeError(f"Three incomplete native attempts: {path}")
 t=time.time();log("COMMAND_START "+" ".join(map(str,cmd)))
 p=subprocess.run(cmd,cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT/"src")},timeout=40*60)
 log(f"COMMAND_END rc={p.returncode} elapsed_s={time.time()-t:.1f}")
 if p.returncode or not completed(target):
  retry=None
  for i in range(1,4):
   q=Path(str(path)+f"_retry{i}")
   if completed(q):log(f"REUSE_COMPLETED_RETRY {q}");return
   if not q.exists():retry=q;break
  if retry is None:raise RuntimeError(f"No retry directory available: {path}")
  retry_cmd=cmd.copy();retry_cmd[retry_cmd.index("--run-dir")+1]=str(retry)
  if p.returncode:
   retry_cmd[retry_cmd.index("--batch-per-dataset")+1]="128"
  log("RETRY_ONCE "+" ".join(map(str,retry_cmd)))
  q=subprocess.run(retry_cmd,cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT/"src")},timeout=40*60)
  if q.returncode or not completed(retry):raise RuntimeError(f"Native head retry failed: {retry}")

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 while not FINAL_LOG.exists() or "PHASEB_FINAL_COMPLETE" not in FINAL_LOG.read_text(encoding="utf-8",errors="ignore"):
  log("WAITING_FOR_PRIMARY_ABLATION_MATRIX")
  time.sleep(60)
 if not (CACHE/"cache_manifest.json").exists():raise RuntimeError("Verified cache missing")
 for seed in SEEDS:
  smoke=OUT/f"phaseB_residual_adaptive_native_seed{seed}_smoke"
  if not completed(smoke):
   run([PY,str(RUNNER),"--seed",str(seed),"--lambda-coral","0.05","--batch-per-dataset","256",
     "--ablation","residual_adaptive","--head-mode","native","--cache-root",str(CACHE),
     "--max-steps","100","--validation-interval","100","--run-dir",str(smoke)],smoke)
  final=OUT/f"phaseB_final_residual_adaptive_native_seed{seed}"
  if not completed(final):
   run([PY,str(RUNNER),"--seed",str(seed),"--lambda-coral","0.05","--batch-per-dataset","256",
     "--ablation","residual_adaptive","--head-mode","native","--cache-root",str(CACHE),
     "--max-steps","2000","--validation-interval","500","--run-dir",str(final)],final)
  log(f"NATIVE_HEAD_COMPLETE seed={seed} path={final}")
 log("NATIVE_HEAD_MATRIX_COMPLETE")
if __name__=="__main__":main()


