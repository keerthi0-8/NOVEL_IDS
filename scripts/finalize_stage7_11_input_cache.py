from __future__ import annotations
import hashlib,json,shutil,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(r"C:\NOVEL_IDS_runs\phaseB_input_cache_20261126")
STAGE5=Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
STAGE6=Path(r"C:\NOVEL_IDS_runs\stage6\accepted_candidate09_lambda1_seed42\best.pt")
INPUT=ROOT/"data"/"interim"/"stage6_trainval_only"
EXPECTED_STAGE5="9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"
EXPECTED_STAGE6="1c66f6f8d1a029266e5a637b5db4d6f49881988bf16462d2c3fbc09cd704ecbd"
ROWS={"nsl_kdd":{"train":100778,"validation":25074},"unsw_nb15":{"train":140273,"validation":32999}}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 t0=time.time()
 if sha(STAGE5)!=EXPECTED_STAGE5:raise RuntimeError("Stage5 snapshot provenance mismatch")
 if sha(STAGE6)!=EXPECTED_STAGE6:raise RuntimeError("Stage6 checkpoint provenance mismatch")
 chunks=OUT/"chunks"
 manifest={"status":"VERIFIED","stage5_sha256":EXPECTED_STAGE5,"stage6_sha256":EXPECTED_STAGE6,
  "token_dtype":"float32","zszd_dtype":"float32","availability_dtype":"uint8",
  "augmentation":{"training_draws_per_row":1,"seed_formula":"20261125 + dataset_index*10000 + 1*101 + chunk_start*13","chunk_rows":5000,"batch_rows":128},
  "common_only_definition":"Stage5 input restricted to Level-1 exact common feature slots; non-common slots excluded from pooling and attention",
  "validation":"clean, no augmentation","datasets":{}}
 expected_total=0
 for ds,splits in ROWS.items():
  manifest["datasets"][ds]={}
  for split,n in splits.items():
   manifest["datasets"][ds][split]={"rows":n,"modes":{}}
   for mode in ("full","common_only"):
    ms=[]
    for f in chunks.glob(f"{ds}_{split}_{mode}_*.json"):
     d=json.loads(f.read_text());ms.append((int(d["start"]),int(d["stop"]),f))
    ms.sort(key=lambda x:x[0])
    cursor=0
    for a,b,f in ms:
     if a!=cursor or b<=a:raise RuntimeError(f"Chunk gap/overlap {ds}/{split}/{mode}: expected {cursor}, got {a}:{b}")
     cursor=b
    if cursor!=n:raise RuntimeError(f"Incomplete cache {ds}/{split}/{mode}: {cursor}/{n}")
    sd=OUT/ds/split
    files={"zszd":sd/f"{mode}_zszd.npy","tokens":sd/f"{mode}_tokens.npy","available":sd/f"{mode}_available.npy"}
    arrays={k:np.load(v,mmap_mode="r") for k,v in files.items()}
    if arrays["zszd"].shape!=(n,256) or arrays["tokens"].shape!=(n,78,128) or arrays["available"].shape!=(n,78):
     raise RuntimeError(f"Shape mismatch {ds}/{split}/{mode}")
    if arrays["tokens"].dtype!=np.float32 or arrays["zszd"].dtype!=np.float32:
     raise RuntimeError(f"Unexpected dtype {ds}/{split}/{mode}")
    # Stream finite-value and binary-mask checks without loading full cache into RAM.
    for st in range(0,n,5000):
     en=min(n,st+5000)
     if not np.isfinite(arrays["zszd"][st:en]).all() or not np.isfinite(arrays["tokens"][st:en]).all():
      raise FloatingPointError(f"Non-finite cache values {ds}/{split}/{mode} rows {st}:{en}")
     av=arrays["available"][st:en]
     if not np.isin(av,[0,1]).all():raise RuntimeError(f"Availability mask not binary {ds}/{split}/{mode}")
    file_info={}
    for k,v in files.items():
     file_info[k]={"path":str(v),"bytes":v.stat().st_size,"sha256":sha(v)}
    manifest["datasets"][ds][split]["modes"][mode]={"rows":n,"chunks":len(ms),"files":file_info}
    expected_total+=len(ms)
    print(f"VERIFIED {ds}/{split}/{mode} rows={n} chunks={len(ms)}",flush=True)
  manifest["datasets"][ds]["input_artifact_sha256"]=sha(INPUT/f"{ds}_inputs.joblib")
 manifest["chunk_count"]=expected_total
 manifest["elapsed_seconds"]=time.time()-t0
 manifest["cache_build_log_sha256"]=sha(OUT/"orchestration.log")
 tmp=OUT/"cache_manifest.json.tmp";tmp.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8");tmp.replace(OUT/"cache_manifest.json")
 print("CACHE_FINALIZED",OUT/"cache_manifest.json","sha256",sha(OUT/"cache_manifest.json"),"elapsed_s",round(manifest["elapsed_seconds"],1),flush=True)
if __name__=="__main__":main()
