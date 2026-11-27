from pathlib import Path
import argparse, json, time, os, sys
import pandas as pd
from novel_ids.inference.api import load_model, onboard_csv, predict_csv
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--mode",choices=["option_a_target","option_b_strict_source_level1"],required=True)
 a=p.parse_args()
 root=Path.cwd(); run=Path(r"C:\NOVEL_IDS_runs\phaseD_cic_second_run_20261127")
 src=run/"CIC_DDoS2019_second_run_features.csv"
 out=run/("option_a" if a.mode=="option_a_target" else "option_b")
 out.mkdir(parents=True,exist_ok=True)
 print("STAGE 1 import complete; mode="+a.mode,flush=True)
 t=time.monotonic()
 model=load_model(r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43",device="cpu")
 print(f"STAGE 2 frozen model loaded elapsed={time.monotonic()-t:.1f}s",flush=True)
 report=onboard_csv(src,model=model,label_column=None,chunksize=8192,calibration_rows=50000,
                    feature_metadata_path=root/"config/cic_feature_metadata.yaml",normalization_mode=a.mode)
 print("STAGE 3 onboarding complete rows="+str(report.rows)+" compatibility="+json.dumps(report.compatibility,sort_keys=True),flush=True)
 t=time.monotonic()
 pred,summary=predict_csv(src,model,onboarding=report,label_column=None,batch_size=256,chunk_size=8192)
 print(f"STAGE 4 predictions complete rows={len(pred)} elapsed={time.monotonic()-t:.1f}s",flush=True)
 pred_path=out/"predictions.csv"; summary_path=out/"summary.json"
 pred.to_csv(pred_path,index=False)
 summary_path.write_text(json.dumps(summary,indent=2,allow_nan=True),encoding="utf-8")
 print("OUTPUT",pred_path,flush=True); print("SUMMARY",summary_path,flush=True)
 print("DECISIONS",json.dumps(summary["decision_counts"],sort_keys=True),flush=True)
 print("COMPATIBILITY",json.dumps(summary["compatibility"],sort_keys=True),flush=True)
 print("TRUST",summary["trust_level"],"NORMALIZATION",summary["normalization_mode"],flush=True)
if __name__=="__main__": main()

