from __future__ import annotations
import json,hashlib
from pathlib import Path
OUT=Path(r"C:\NOVEL_IDS_runs")
ROOT=Path(r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project")
RUNS={
 "full_alignment_on": [OUT/"stage7_11"/"coral_screen_lambda0p05_seed43"],
 "shared_only": [OUT/f"phaseB_final_shared_only_seed{s}" for s in (42,43)],
 "residual_fixed_gate1": [OUT/f"phaseB_final_residual_fixed_gate1_seed{s}" for s in (42,43)],
 "residual_adaptive_alignment_off": [OUT/f"phaseB_final_residual_adaptive_seed{s}" for s in (42,43)],
 "common_features_only": [OUT/f"phaseB_final_common_only_seed{s}" for s in (42,43)],
 "native_heads_residual_adaptive": [OUT/f"phaseB_final_residual_adaptive_native_seed{s}" for s in (42,43)],
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 f=p/"result.json"
 if not f.exists():return None
 d=json.loads(f.read_text(encoding="utf-8"))
 hist=d.get("validation_history",[])
 best=next((x for x in hist if int(x.get("step",-1))==int(d.get("best_step",-2))),None)
 if best is None and hist:best=max(hist,key=lambda x:x.get("mean_macro_f1",-1))
 return {"path":str(p),"seed":d.get("seed"),"best_step":d.get("best_step"),
  "best_validation_mean_macro_f1":d.get("best_validation_macro_f1"),
  "best_validation":best,"checkpoint_sha256":d.get("best_checkpoint_sha256"),
  "last_checkpoint_sha256":d.get("last_checkpoint_sha256"),
  "stage5_sha256":d.get("stage5_sha256"),"stage6_sha256":d.get("stage6_sha256"),
  "steps_completed":d.get("steps_completed"),"elapsed_seconds":d.get("elapsed_seconds"),
  "config":json.loads((p/"run_config.json").read_text(encoding="utf-8")) if (p/"run_config.json").exists() else None}
def main():
 report={}
 for name,paths in RUNS.items():
  runs=[x for p in paths if (x:=load(p)) is not None]
  if not runs:
   report[name]={"status":"pending","expected_paths":[str(p) for p in paths]};continue
  report[name]={"status":"complete" if len(runs)==len(paths) else "partial","runs":runs,
   "selected":max(runs,key=lambda x:x["best_validation_mean_macro_f1"] or -1)}
 p=OUT/"phaseB_ablation_summary.json"
 p.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
 lines=["# Phase B validation ablation summary","",
  "Validation only. No source-test or unseen data used for model selection.","",
  "| Variant | Seed | Best step | Mean macro-F1 | NSL-KDD macro-F1 | UNSW-NB15 macro-F1 | Checkpoint SHA-256 |",
  "|---|---:|---:|---:|---:|---:|---|"]
 for name,item in report.items():
  for run in item.get("runs",[]):
   v=run.get("best_validation") or {}; ds=v.get("datasets",{})
   lines.append(f"| {name} | {run.get('seed')} | {run.get('best_step')} | {run.get('best_validation_mean_macro_f1')} | {ds.get('nsl_kdd',{}).get('macro_f1')} | {ds.get('unsw_nb15',{}).get('macro_f1')} | {run.get('checkpoint_sha256')} |")
 md=OUT/"phaseB_ablation_summary.md";md.write_text("\n".join(lines)+"\n",encoding="utf-8")
 print("SUMMARY_JSON",p,"SHA256",sha(p))
 print("SUMMARY_MD",md,"SHA256",sha(md))
 for k,v in report.items():
  sel=v.get("selected")
  print(k,v.get("status"),None if sel is None else (sel.get("seed"),sel.get("best_validation_mean_macro_f1"),sel.get("checkpoint_sha256")))
if __name__=="__main__":main()
