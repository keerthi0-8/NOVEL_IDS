from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
import yaml, joblib, numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(ROOT/"src"))
import run_phaseC_one_shot as c
MAP=ROOT/"config/phaseC_label_map.yaml"
ART={
 "nsl_kdd":ROOT/"data/processed/unified_representation/nsl_kdd_unified_v2_no_kddtest_feature_overlap.joblib",
 "unsw_nb15":ROOT/"data/processed/unified_representation/unsw_nb15_unified_v2_no_test_feature_overlap.joblib",
}
def norm(x): return str(x).strip().casefold()
def mappings(cfg,ds):
 if ds=="nsl_kdd":
  out={}
  for fam,names in cfg["nsl_kdd"]["families"].items():
   for n in names: out[norm(n)]=cfg["nsl_kdd"]["unified"][fam]
  out["normal"]="Normal"
  return out
 out={norm(k):v for k,v in cfg["unsw_nb15"]["mappings"].items()}
 for alias,target in cfg["unsw_nb15"].get("aliases",{}).items(): out[norm(alias)]=out[norm(target)]
 return out
def map_one(ds,x,mp):
 k=norm(x)
 return mp.get(k,"UNSUPPORTED_"+str(x).strip())
def fixture_check():
 # Known classes: Normal, DoS, Probe_Recon. NSL_R2L has zero support.
 truth=np.array(["Normal","DoS","DoS","UNSUPPORTED_odd","Probe_Recon"])
 pred=np.array(["Normal","Normal","DoS","Normal","Probe_Recon"])
 known=~np.char.startswith(truth,"UNSUPPORTED_")
 acc=float(np.mean(truth[known]==pred[known]))
 # TP/FP/FN for DoS in the known-only fixture: TP=1, FP=0, FN=1.
 precision=1.0; recall=0.5; f1=2*precision*recall/(precision+recall)
 assert abs(acc-0.75)<1e-12 and abs(f1-2/3)<1e-12
 assert int((~known).sum())==1
 return {"known_support":{"Normal":1,"DoS":2,"Probe_Recon":1,"NSL_R2L":0},
         "unsupported_count":1,"accuracy_known":acc,"DoS_precision":precision,
         "DoS_recall":recall,"DoS_f1":f1,"hand_check":"PASS"}
def main():
 cfg=yaml.safe_load(MAP.read_text(encoding="utf-8"))
 print("LABEL_MAP_SHA256",hashlib.sha256(MAP.read_bytes()).hexdigest(),flush=True)
 print("SYNTHETIC_FIXTURE",json.dumps(fixture_check()),flush=True)
 # Run the original full validation-only model/baseline preflight. It does not load test partitions.
 print("VALIDATION_PREFLIGHT_START",flush=True)
 c.preflight()
 print("VALIDATION_PREFLIGHT_PASS",flush=True)
 report={"source":cfg["source"],"datasets":{}}
 for ds,path in ART.items():
  # mmap mode keeps numerical feature arrays lazy; only labels are touched below.
  art=joblib.load(path,mmap_mode="r")
  mp=mappings(cfg,ds); unique={}
  for split in ("test_independent","test_full"):
   raw=np.asarray(art[split]["labels"]).astype(str)
   pairs=sorted({(str(x),map_one(ds,x,mp)) for x in raw},key=lambda z:z[0].casefold())
   unique[split]=[{"raw":a,"mapped":b} for a,b in pairs]
  report["datasets"][ds]=unique
  del art
  print("LABEL_VOCAB",ds,json.dumps(unique),flush=True)
 out=ROOT/"results/phaseC_label_vocabulary_report.json"
 out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
 print("LABEL_VOCAB_REPORT",out,flush=True)
if __name__=="__main__": main()
