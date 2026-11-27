from __future__ import annotations
import os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts"/"build_stage7_11_cache_chunk.py"
OUT=Path(r"C:\NOVEL_IDS_runs\phaseB_input_cache_20261126")
INPUT=Path(r"C:\NOVEL_IDS_runs\phaseB_shared_only_smoke_memmap_retry2_seed43")
JOBS=[("nsl_kdd","train",100778),("nsl_kdd","validation",25074),
      ("unsw_nb15","train",140273),("unsw_nb15","validation",32999)]
CHUNK=5000
def free_ram():
    import ctypes
    class M(ctypes.Structure):
        _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
    m=M();m.dwLength=ctypes.sizeof(M);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return m.ullAvailPhys/1024**3
def log(msg):
    line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line,flush=True)
    with (OUT/"orchestration.log").open("a",encoding="utf-8") as f:f.write(line+"\n")
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env["PYTHONPATH"]=str(ROOT/"src")
    total=0;start_time=time.time()
    for ds,split,n in JOBS:
        for mode in ("full","common_only"):
            for begin in range(0,n,CHUNK):
                end=min(begin+CHUNK,n)
                if free_ram()<2.0: raise RuntimeError(f"HARD STOP RAM before chunk {ds}/{split}/{mode}/{begin}:{end}: {free_ram():.2f}")
                if shutil.disk_usage("C:\\").free<10*1024**3: raise RuntimeError("HARD STOP disk free below 10 GiB")
                args=[sys.executable,str(SCRIPT),"--out",str(OUT),"--input-cache-dir",str(INPUT),
                      "--dataset",ds,"--split",split,"--mode",mode,"--start",str(begin),"--stop",str(end)]
                log(f"CHUNK_START {ds}/{split}/{mode} {begin}:{end} free_ram_gb={free_ram():.2f}")
                p=subprocess.run(args,env=env,text=True)
                if p.returncode: raise RuntimeError(f"CHUNK_FAILED {ds}/{split}/{mode} {begin}:{end} exit={p.returncode}")
                total+=1
                log(f"CHUNK_OK count={total} elapsed_s={time.time()-start_time:.1f} free_ram_gb={free_ram():.2f}")
    log(f"ALL_CACHE_CHUNKS_COMPLETE chunks={total} elapsed_s={time.time()-start_time:.1f}")
if __name__=="__main__":main()
