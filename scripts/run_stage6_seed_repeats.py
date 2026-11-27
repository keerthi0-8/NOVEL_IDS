from __future__ import annotations
import json, time, traceback, sys
from pathlib import Path
import torch, yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"scripts"))
import run_stage6_fastpath_training as fast
OUT=fast.OUT
def main():
 selection_path=OUT/"phase2_selection.json"
 if not selection_path.is_file(): raise FileNotFoundError("Run six-arm probe evaluation and selection first.")
 selection=json.loads(selection_path.read_text(encoding="utf-8"))
 selected=selection.get("selected_arm")
 if not selected:
  fast.write_json(OUT/"seed_repeat_results.json",{"seed_sd_assessed":False,"reason":"No qualifying Stage6 arm selected","seeds":[]})
  return 0
 training=json.loads((OUT/"phase2_training_results.json").read_text(encoding="utf-8"))
 meta=next(x for x in training["seed42_arms"] if x["arm"]==selected)
 name=meta["candidate"]; candidate=fast.CANDIDATES[name]; lam=float(meta["lambda_decorr"])
 plan=fast.read_plan(); projected_total=2*float(plan["projected_10000_step_seconds"])
 if projected_total>7200:
  result={"seed_sd_assessed":False,"reason":"Two additional 10,000-step seed runs projected above 2 hours","projected_two_seed_seconds":projected_total,"seeds":[],"selected_seed42_arm":selected}
  fast.write_json(OUT/"seed_repeat_results.json",result); return 0
 raw=yaml.safe_load((ROOT/"config"/"stage6_projection.yaml").read_text(encoding="utf-8"))
 results=[]
 for seed in (43,44):
  r=fast.run_arm(raw,name,candidate,lam,seed,plan); results.append(r)
  fast.write_json(OUT/"seed_repeat_partial_results.json",{"selected_seed42_arm":selected,"candidate":name,"lambda_decorr":lam,"seed_runs":results,"projected_two_seed_seconds":projected_total})
 result={"seed_sd_assessed":True,"selected_seed42_arm":selected,"candidate":name,"lambda_red":candidate["lambda_red"],"lambda_decorr":lam,"seed_runs":results,"projected_two_seed_seconds":projected_total}
 fast.write_json(OUT/"seed_repeat_results.json",result)
 fast.log(f"SEED43_44_COMPLETE selected={selected} arms={[x['arm'] for x in results]}")
 return 0
if __name__=="__main__":
 try: raise SystemExit(main())
 except Exception:
  fast.log("SEED_REPEAT_STOPPED "+traceback.format_exc()); raise
