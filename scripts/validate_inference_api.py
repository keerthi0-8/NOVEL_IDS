from __future__ import annotations
import json, sys, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project")
RUN=Path(r"C:\NOVEL_IDS_runs\phaseE_validation_repro_20261126")
OUT=RUN/"inference_api_validation"
OUT.mkdir(parents=True,exist_ok=True)
LOG=OUT/"progress.log"
def emit(s):
    line=f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
    print(line,flush=True)
    with LOG.open("a",encoding="utf-8") as f:f.write(line+"\n")

sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
from novel_ids.inference import load_model, onboard_csv, predict_csv
from novel_ids.ood.stage11 import energy_score
from novel_ids.alignment.stage8 import NSL_KDD, UNSW_NB15

checkpoint=Path(r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43\best.pt")
ref=np.load(RUN/"validation_distributions.npz")
emit("API_LOAD_START")
model=load_model(checkpoint,device="cuda")
emit(f"API_LOAD_COMPLETE device={model.device} checkpoint_sha={model.model_info()['checkpoint_sha256']}")

# Streaming API smoke: unfamiliar schema, label exclusion, all-NaN, Inf and unknown category.
n=37
rng=np.random.default_rng(20261126)
frame=pd.DataFrame({
    "renamed_duration":rng.normal(size=n),
    "renamed_bytes":rng.lognormal(3,1,size=n),
    "protocol_unknown":["tcp","udp","new-protocol"]*(n//3)+["tcp"],
    "all_nan":np.full(n,np.nan),
    "with_inf":np.where(np.arange(n)%2==0,np.inf,np.arange(n,dtype=float)),
    "Label":["BENIGN","DoS"]*(n//2)+["BENIGN"],
})
p1=OUT/"synthetic_schema_a.csv"; p2=OUT/"synthetic_schema_b.csv"
frame.to_csv(p1,index=False)
frame2=frame.copy(); frame2["Label"]=["changed-label"]*n; frame2.to_csv(p2,index=False)
rep1=onboard_csv(p1,model=model,label_column="Label",chunksize=9,calibration_rows=19)
rep2=onboard_csv(p2,model=model,label_column="Label",chunksize=9,calibration_rows=19)
pred1,sum1=predict_csv(p1,model,onboarding=rep1,batch_size=7,chunk_size=9)
pred2,sum2=predict_csv(p2,model,onboarding=rep2,batch_size=7,chunk_size=9)
assert pred1.equals(pred2), "Changing labels changed predictions."
assert len(pred1)==n and sum1["rows"]==n
assert np.isfinite(pred1.select_dtypes(include=[np.number]).to_numpy()).all()
emit(f"STREAMING_API_SMOKE_PASS rows={n} decisions={sum1['decision_counts']} label_invariance=PASS chunksize=9 batch=7")

# Reproduce the frozen source-validation probabilities using the exact frozen model
# objects loaded by the public API and the approved source validation batch path.
runner=model.resource_module
runner.RUN_DIR=RUN/"parity_source_validation_loader_gpu"
runner.RUN_DIR.mkdir(parents=True,exist_ok=True)
helper, source_branches=runner.read_data()
emit("SOURCE_VALIDATION_KEYS="+str({d:list(helper.datasets[d]["validation"].keys()) for d in runner.DATASETS}))
runner.helper=helper
helper.stage5=model.stage5
labels=runner.load_labels(model.taxonomy)
import run_phaseC_one_shot as phasec
stage5_ckpt=torch.load(phasec.STAGE5,map_location="cpu",weights_only=False)
aug_cfg=stage5_ckpt["config"]; del stage5_ckpt
level,confidence,group_only,level3_mask=runner.make_metadata()
results={}
for di,ds in enumerate(runner.DATASETS):
    nrows=len(helper.datasets[ds]["validation"]["numeric_values"])
    probs=[]; scores=[]
    for st in range(0,nrows,256):
        ix=np.arange(st,min(nrows,st+256))
        batch=runner.source_batch(helper,source_branches,ds,"validation",ix,False,
            torch.Generator(device=model.device).manual_seed(4300+di),level3_mask,aug_cfg)
        sid=torch.full((len(ix),),NSL_KDD if ds=="nsl_kdd" else UNSW_NB15,device=model.device,dtype=torch.long)
        with torch.no_grad():
            logits,_,_,fused=runner.batch_forward(
                batch,model.variant["stage7"],model.variant["stage8"],model.variant["stage9"],
                model.variant["stage10"],model.variant["classifier"],sid,
                level,confidence,group_only,1.0,
            )
            temp=float(model.calibration["temperature"])
            probs.append(torch.softmax(logits.float()/temp,dim=1).cpu().numpy())
            scores.append(model.ood_combiner.score(
                model.mahalanobis.score(fused["Z_u"].detach().cpu()),
                energy_score(logits.float().detach().cpu(),temperature=temp),
            ).cpu().numpy())
    p=np.concatenate(probs); o=np.concatenate(scores)
    pref=ref[f"{ds}_probabilities"]; oref=ref[f"{ds}_ood_scores"]
    pmax=float(np.max(np.abs(p-pref)))
    omax=float(np.max(np.abs(o-oref)))
    results[ds]={"rows":int(nrows),"max_abs_probability_difference":pmax,
                  "max_abs_ood_difference":omax,"probability_parity_le_1e_5":bool(pmax<=1e-5)}
    emit(f"PARITY {ds} rows={nrows} max_abs_prob={pmax:.10g} max_abs_ood={omax:.10g}")
payload={"checkpoint_sha256":model.model_info()["checkpoint_sha256"],
         "reference":str(RUN/"validation_distributions.npz"),
         "api_load_model_used":True,"source_validation_batch_path":"run_stage7_11_experiment.batch_forward",
         "probability_tolerance":1e-5,"datasets":results,
         "overall_pass":all(v["probability_parity_le_1e_5"] for v in results.values()),
         "note":"This checks the frozen API-loaded model's source-validation forward path against the saved Phase-E validation distributions. It is not a raw-source CSV onboarding test; predict_csv is the unseen-data streaming path."}
(OUT/"validation_parity_gpu.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
emit("VALIDATION_PARITY_COMPLETE overall_pass="+str(payload["overall_pass"]))
print(json.dumps(payload,indent=2))
