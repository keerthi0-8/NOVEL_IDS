from __future__ import annotations
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
spec=importlib.util.spec_from_file_location("stage7runner",ROOT/"scripts"/"run_stage7_11_experiment.py")
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
OUT=Path(r"C:\NOVEL_IDS_runs\stage7_11\gate_diag_seed42")
OUT.mkdir(parents=True,exist_ok=False)
runner.RUN_DIR=OUT
runner.DEVICE=torch.device("cuda")
torch.set_num_threads(2); torch.set_num_interop_threads(2)
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
runner.verify_protected()
helper,branches=runner.read_data()
ck5=torch.load(runner.STAGE5,map_location="cpu",weights_only=False)
aug_cfg=ck5["config"]; del ck5
level,confidence,group,level3=runner.make_metadata()
runs={
 "lambda0.0":Path(r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0_seed42\best.pt"),
 "lambda0.05":Path(r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed42\best.pt"),
 "lambda0.2":Path(r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p2_seed42\best.pt"),
 "lambda1.0":Path(r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda1_seed42\best.pt"),
}
result={"stage5_sha256":runner.STAGE5_SHA,"stage6_sha256":runner.STAGE6_SHA,"datasets":{},"runs":{}}
t0=time.perf_counter()
for name,path in runs.items():
 state=torch.load(path,map_location="cpu",weights_only=False)
 stage7=runner.AdaptiveResidualFusion().to(runner.DEVICE)
 stage7.load_state_dict(state["stage7"],strict=True); stage7.eval()
 run={}
 for di,ds in enumerate(runner.DATASETS):
  n=len(helper.datasets[ds]["validation"]["numeric_values"])
  sums=torch.zeros(128,device="cpu",dtype=torch.float64)
  lows=torch.full((128,),float("inf"),device="cpu")
  highs=torch.full((128,),float("-inf"),device="cpu")
  seen=0; rng=torch.Generator(device=runner.DEVICE); rng.manual_seed(42000+di)
  for st in range(0,n,runner.VAL_BATCH):
   idx=np.arange(st,min(n,st+runner.VAL_BATCH))
   b=runner.source_batch(helper,branches,ds,"validation",idx,False,rng,level3,aug_cfg)
   with torch.no_grad():
    g=stage7(b["Z_s"],b["R_proj"])["G"].float()
   gm=g.mean(0).double().cpu()
   sums+=g.double().sum(0).cpu()
   lows=torch.minimum(lows,g.min(0).values.cpu())
   highs=torch.maximum(highs,g.max(0).values.cpu())
   seen+=len(idx)
  means=(sums/seen).numpy()
  low_dims=np.where(means<0.01)[0].tolist()
  high_dims=np.where(means>0.99)[0].tolist()
  sat=sorted(set(low_dims+high_dims))
  run[ds]={"rows":seen,"global_mean":float(means.mean()),
   "per_dimension_mean":means.tolist(),"per_dimension_min":lows.numpy().tolist(),
   "per_dimension_max":highs.numpy().tolist(),"low_saturated_dimensions":low_dims,
   "high_saturated_dimensions":high_dims,"saturated_dimension_count":len(sat),
   "saturation_fraction":float(len(sat)/128),"all_dimensions_saturated":len(sat)==128}
  print(name,ds,"mean",run[ds]["global_mean"],"saturated",len(sat),"/128",flush=True)
 result["runs"][name]=run
 del stage7,state; torch.cuda.empty_cache()
result["elapsed_seconds"]=time.perf_counter()-t0
(OUT/"gate_saturation_diagnostics.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
print("GATE_DIAGNOSTICS_COMPLETE",result["elapsed_seconds"],flush=True)
