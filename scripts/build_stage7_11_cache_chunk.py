from __future__ import annotations
import argparse,gc,hashlib,json,time
from pathlib import Path
import joblib,numpy as np,torch,yaml
from novel_ids.representation.dual_encoder import DualEncoder
from novel_ids.residual.stage6_fastpath import make_augmentation_masks

ROOT=Path(__file__).resolve().parents[1]
SNAP=Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
SCHEMA=ROOT/"data"/"processed"/"unified_representation"/"unified_feature_schema.yaml"
EXPECTED="9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b"
SEED_BASE=20261125; DRAW=1; BATCH=128

def ram():
    import ctypes
    class M(ctypes.Structure):
        _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
    m=M();m.dwLength=ctypes.sizeof(M);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return m.ullAvailPhys/1024**3

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
    return h.hexdigest()

@torch.no_grad()
def encode(model,arr,ds,ix,mode,common,rng,cfg,train,level3):
    av=np.asarray(arr["M_avail"][ix],dtype=np.float32).copy()
    mi=np.asarray(arr["M_miss"][ix],dtype=np.float32).copy()
    dropped=np.zeros_like(av,dtype=bool)
    if train:
        av,mi,dropped,_=make_augmentation_masks(av,mi,np.array(arr["numeric_token_mask"],copy=True),
          np.array(arr["categorical_token_mask"],copy=True),level3,rng,
          float(cfg["feature_set_dropout_rate"]),float(cfg["feature_set_dropout_all_probability"]),float(cfg["value_masking_rate"]))
    if mode=="common_only":
        av*=common[None,:].astype(np.float32); mi*=common[None,:].astype(np.float32)
        dropped|=(~common)[None,:]
    out=model(numeric_values=arr["numeric_values"][ix],categorical_values=arr["categorical_values"][ix],
      M_avail=av,M_miss=mi,numeric_token_mask=np.array(arr["numeric_token_mask"],copy=True),
      categorical_token_mask=np.array(arr["categorical_token_mask"],copy=True),dataset=ds,feature_descriptors=None)
    tokens=out["token_embeddings"].float()
    drop_t=torch.as_tensor(dropped,device=tokens.device,dtype=torch.bool)
    pool_valid=out["M_avail"].bool() | (~drop_t)
    w=pool_valid.unsqueeze(-1).to(tokens.dtype)
    pooled=(tokens*w).sum(1)/w.sum(1).clamp_min(1.0)
    zs=model.shared_encoder(pooled); zd=model.shared_context_adapter(pooled,out["feature_context"])
    return torch.cat((zs.float(),zd.float()),1).cpu().numpy(),tokens.cpu().numpy(),out["M_avail"].bool().cpu().numpy().astype(np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True);ap.add_argument("--input-cache-dir",required=True)
    ap.add_argument("--dataset",choices=["nsl_kdd","unsw_nb15"],required=True)
    ap.add_argument("--split",choices=["train","validation"],required=True)
    ap.add_argument("--mode",choices=["full","common_only"],required=True)
    ap.add_argument("--start",type=int,required=True);ap.add_argument("--stop",type=int,required=True)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if sha(SNAP)!=EXPECTED: raise RuntimeError("Stage5 snapshot differs from approved freeze manifest")
    if ram()<2.0: raise RuntimeError(f"HARD STOP RAM {ram():.2f} GiB")
    if torch.cuda.mem_get_info()[0]<5*1024**3: raise RuntimeError("GPU free memory below 5 GiB")
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    schema=yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    common=np.asarray([int(x.get("level",0))==1 for x in schema["features"]],dtype=bool)
    level3=[int(x["index"]) for x in schema["features"] if x.get("dataset")==args.dataset and int(x.get("level",0))==3]
    input_path=Path(args.input_cache_dir)/f"{args.dataset}_{args.split}_input_memmap.joblib"
    arr=joblib.load(input_path,mmap_mode="r")
    n=len(arr["numeric_values"])
    if args.start<0 or args.stop>n or args.start>=args.stop: raise ValueError("Invalid chunk range")
    ds_idx=0 if args.dataset=="nsl_kdd" else 1
    ck=torch.load(SNAP,map_location="cpu",weights_only=False);cfg=ck["config"]
    ids=np.load(ROOT/"data"/"processed"/"unified_representation"/"feature_identity_vectors.npy",allow_pickle=False)
    model=DualEncoder(ids,dropout=float(cfg["dropout"]),value_masking_rate=float(cfg["value_masking_rate"]),
      feature_set_dropout_rate=float(cfg["feature_set_dropout_rate"]),feature_set_dropout_all_probability=float(cfg["feature_set_dropout_all_probability"]))
    model.load_state_dict(ck["dual_encoder"],strict=True);del ck
    model.to("cuda").eval()
    for p in model.parameters():p.requires_grad_(False)
    split_dir=out/args.dataset/args.split;split_dir.mkdir(parents=True,exist_ok=True)
    paths={"z":split_dir/f"{args.mode}_zszd.npy","t":split_dir/f"{args.mode}_tokens.npy","a":split_dir/f"{args.mode}_available.npy"}
    if paths["z"].exists():
        zmm=np.load(paths["z"],mmap_mode="r+");tmm=np.load(paths["t"],mmap_mode="r+");amm=np.load(paths["a"],mmap_mode="r+")
    else:
        zmm=np.lib.format.open_memmap(paths["z"],mode="w+",dtype=np.float32,shape=(n,256))
        tmm=np.lib.format.open_memmap(paths["t"],mode="w+",dtype=np.float32,shape=(n,78,128))
        amm=np.lib.format.open_memmap(paths["a"],mode="w+",dtype=np.uint8,shape=(n,78))
    rng=np.random.default_rng(SEED_BASE+ds_idx*10000+DRAW*101+args.start*13) if args.split=="train" else None
    t0=time.time();last=t0
    for st in range(args.start,args.stop,BATCH):
        if ram()<2.0: raise RuntimeError(f"HARD STOP RAM below 2 GiB at row {st}: {ram():.2f}")
        en=min(args.stop,st+BATCH);ix=np.arange(st,en)
        z,t,a=encode(model,arr,args.dataset,ix,args.mode,common,rng,cfg,args.split=="train",level3)
        
        if not np.isfinite(z).all() or not np.isfinite(t).all(): raise FloatingPointError(f"Non-finite Stage5 cache values for {args.dataset}/{args.split}/{args.mode} rows {st}:{en}")
        zmm[st:en]=z;tmm[st:en]=t;amm[st:en]=a
        if (st//BATCH)%20==0:
            zmm.flush();tmm.flush();amm.flush();gc.collect()
        if time.time()-last>=60 or st==args.start:
            print(f"{args.dataset}/{args.split}/{args.mode} rows={en}/{n} chunk={args.start}:{args.stop} elapsed={time.time()-t0:.1f}s free_ram={ram():.2f}GiB",flush=True);last=time.time()
    zmm.flush();tmm.flush();amm.flush()
    marker=out/"chunks"/f"{args.dataset}_{args.split}_{args.mode}_{args.start}_{args.stop}.json";marker.parent.mkdir(exist_ok=True)
    marker.write_text(json.dumps({"dataset":args.dataset,"split":args.split,"mode":args.mode,"start":args.start,"stop":args.stop,
      "rows_written":args.stop-args.start,"seed":SEED_BASE+ds_idx*10000+DRAW*101+args.start*13 if args.split=="train" else None,
      "token_dtype":"float32","stage5_sha256":EXPECTED},indent=2),encoding="utf-8")
    print(f"CHUNK_COMPLETE {marker} elapsed={time.time()-t0:.1f}s free_ram={ram():.2f}GiB",flush=True)
if __name__=="__main__":main()
