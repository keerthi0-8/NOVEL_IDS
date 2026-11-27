from __future__ import annotations
import argparse, gc, hashlib, json, time
from pathlib import Path
import joblib, numpy as np, torch, yaml
from novel_ids.representation.dual_encoder import DualEncoder
from novel_ids.residual.stage6_data import load_stage6_input_artifact
from novel_ids.residual.stage6_fastpath import make_augmentation_masks, sha256_file

ROOT=Path(__file__).resolve().parents[1]
SNAP=Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
INPUT=ROOT/"data"/"interim"/"stage6_trainval_only"
SCHEMA=ROOT/"data"/"processed"/"unified_representation"/"unified_feature_schema.yaml"
OUT=Path(r"C:\NOVEL_IDS_runs\phaseB_input_cache_20261126")
DATASETS=("nsl_kdd","unsw_nb15")
BATCH=128
SEED_BASE=20261125
DRAW=1
EXPECTED_STAGE5="9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"

def free_ram_gb():
    import ctypes
    class M(ctypes.Structure):
        _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
    m=M(); m.dwLength=ctypes.sizeof(M)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)): raise OSError("GlobalMemoryStatusEx failed")
    return m.ullAvailPhys/1024**3

def digest(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def log(msg):
    line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line,flush=True)
    with (OUT/"cache_build.log").open("a",encoding="utf-8") as f:f.write(line+"\n")

def make_stage5(device):
    ck=torch.load(SNAP,map_location="cpu",weights_only=False)
    cfg=ck["config"]
    ids=np.load(ROOT/"data"/"processed"/"unified_representation"/"feature_identity_vectors.npy",allow_pickle=False)
    model=DualEncoder(ids,dropout=float(cfg["dropout"]),value_masking_rate=float(cfg["value_masking_rate"]),
        feature_set_dropout_rate=float(cfg["feature_set_dropout_rate"]),
        feature_set_dropout_all_probability=float(cfg["feature_set_dropout_all_probability"]))
    model.load_state_dict(ck["dual_encoder"],strict=True); del ck
    model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,cfg

@torch.no_grad()
def encode_batch(model,arr,ds,idx,mode,common_mask,rng,cfg,train):
    avail=np.asarray(arr["M_avail"][idx],dtype=np.float32).copy()
    miss=np.asarray(arr["M_miss"][idx],dtype=np.float32).copy()
    dropped=np.zeros_like(avail,dtype=bool)
    if train:
        avail,miss,dropped,value_mask=make_augmentation_masks(
            avail,miss,arr["numeric_token_mask"],arr["categorical_token_mask"],
            [int(x["index"]) for x in schema["features"] if x.get("dataset")==ds and int(x.get("level",0))==3],
            rng,float(cfg["feature_set_dropout_rate"]),float(cfg["feature_set_dropout_all_probability"]),float(cfg["value_masking_rate"]))
    if mode=="common_only":
        avail*=common_mask[None,:].astype(np.float32)
        miss*=common_mask[None,:].astype(np.float32)
        dropped|=(~common_mask)[None,:]
    out=model(numeric_values=arr["numeric_values"][idx],categorical_values=arr["categorical_values"][idx],
        M_avail=avail,M_miss=miss,numeric_token_mask=np.array(arr["numeric_token_mask"],copy=True),
        categorical_token_mask=np.array(arr["categorical_token_mask"],copy=True),dataset=ds,feature_descriptors=None)
    tokens=out["token_embeddings"].float()
    pool_valid=out["M_avail"].bool() | (~torch.as_tensor(dropped,device=tokens.device,dtype=torch.bool))
    weights=pool_valid.unsqueeze(-1).to(tokens.dtype)
    pooled=(tokens*weights).sum(1)/weights.sum(1).clamp_min(1.0)
    zs=model.shared_encoder(pooled)
    zd=model.shared_context_adapter(pooled,out["feature_context"])
    return torch.cat((zs.float(),zd.float()),dim=1).cpu().numpy(),tokens.cpu().numpy(),out["M_avail"].bool().cpu().numpy().astype(np.uint8)

def main():
    global schema,OUT
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-root",default=str(OUT))
    ap.add_argument("--smoke-batches",type=int,default=0)
    args=ap.parse_args(); OUT=Path(args.output_root)
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite {OUT}")
    if digest(SNAP)!=EXPECTED_STAGE5: raise RuntimeError("Stage5 snapshot hash mismatch")
    if free_ram_gb()<2.0: raise RuntimeError(f"HARD STOP RAM {free_ram_gb():.2f} GiB")
    if torch.cuda.mem_get_info()[0]<5*1024**3: raise RuntimeError("GPU free memory below 5 GiB")
    if __import__("shutil").disk_usage("C:\\").free<10*1024**3: raise RuntimeError("Disk free below 10 GiB")
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    OUT.mkdir(parents=True)
    schema=yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    common_mask=np.asarray([int(x.get("level",0))==1 for x in schema["features"]],dtype=bool)
    if len(common_mask)!=78: raise RuntimeError("Expected 78 feature slots")
    model,cfg=make_stage5(torch.device("cuda"))
    start=time.time(); manifest={"stage5_sha256":digest(SNAP),"draw":DRAW,"seed_base":SEED_BASE,
        "batch_size":BATCH,"common_mask":"Level-1 exact/common slots only","datasets":{}}
    total_batches=0
    for di,ds in enumerate(DATASETS):
        if free_ram_gb()<2.0: raise RuntimeError(f"HARD STOP RAM before dataset {ds}: {free_ram_gb():.2f}")
        inp=INPUT/f"{ds}_inputs.joblib"; input_hash=digest(inp)
        data=load_stage6_input_artifact(inp,ds)
        # Re-serialize uncompressed and reopen read-only as memmaps to release the
        # compressed joblib arrays from RAM before Stage5 batched inference.
        mem_dir=OUT/"input_memmap"; mem_dir.mkdir(exist_ok=True)
        for split_name in ("train","validation"):
            mp=mem_dir/f"{ds}_{split_name}.joblib"
            joblib.dump(data[split_name],mp,compress=0)
            data[split_name]=joblib.load(mp,mmap_mode="r")
            gc.collect()
            log(f"INPUT_MEMMAP_READY {ds}/{split_name} free_ram_gb={free_ram_gb():.2f}")
        dm={}
        for split in ("train","validation"):
            arr=data[split]; n=len(arr["numeric_values"]); train=(split=="train")
            split_dir=OUT/ds/split; split_dir.mkdir(parents=True,exist_ok=True)
            for mode in ("full","common_only"):
                zpath=split_dir/f"{mode}_zszd.npy"; tpath=split_dir/f"{mode}_tokens.npy"; apath=split_dir/f"{mode}_available.npy"
                zmm=np.lib.format.open_memmap(zpath,mode="w+",dtype=np.float32,shape=(n,256))
                tmm=np.lib.format.open_memmap(tpath,mode="w+",dtype=np.float16,shape=(n,78,128))
                amm=np.lib.format.open_memmap(apath,mode="w+",dtype=np.uint8,shape=(n,78))
                rng=np.random.default_rng(SEED_BASE+di*10000+DRAW*101) if train else None
                for st in range(0,n,BATCH):
                    if free_ram_gb()<2.0: raise RuntimeError(f"HARD STOP RAM during cache {ds}/{split}/{mode}: {free_ram_gb():.2f}")
                    en=min(n,st+BATCH); ix=np.arange(st,en)
                    z,t,a=encode_batch(model,arr,ds,ix,mode,common_mask,rng,cfg,train)
                    zmm[st:en]=z; tmm[st:en]=t.astype(np.float16); amm[st:en]=a
                    if total_batches%40==0: gc.collect()
                    total_batches+=1
                    if time.time()-start>0 and (st==0 or (st//BATCH)%40==0):
                        log(f"CACHE_PROGRESS {ds}/{split}/{mode} rows={en}/{n} elapsed_s={time.time()-start:.1f} free_ram_gb={free_ram_gb():.2f}")
                    if args.smoke_batches and total_batches>=args.smoke_batches:
                        log(f"CACHE_SMOKE_COMPLETE batches={total_batches} elapsed_s={time.time()-start:.1f} free_ram_gb={free_ram_gb():.2f}")
                        return
                zmm.flush(); tmm.flush(); amm.flush()
                dm[f"{split}_{mode}"]={"rows":n,"zszd_sha256":digest(zpath),"tokens_sha256":digest(tpath),"available_sha256":digest(apath),
                    "zszd_shape":[n,256],"tokens_shape":[n,78,128],"tokens_dtype":"float16","available_shape":[n,78]}
                del zmm,tmm,amm; gc.collect()
            log(f"CACHE_COMPLETE {ds}/{split} rows={n}")
        manifest["datasets"][ds]={"input_artifact_sha256":input_hash,"splits":dm}
        del data; gc.collect(); torch.cuda.empty_cache()
    manifest["elapsed_seconds"]=time.time()-start
    manifest["status"]="STAGE7_11_INPUT_CACHE_COMPLETE"
    (OUT/"cache_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
    log(f"CACHE_BUILD_COMPLETE elapsed_s={manifest['elapsed_seconds']:.1f} manifest_sha256={digest(OUT/'cache_manifest.json')}")
if __name__=="__main__": main()



