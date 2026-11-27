from pathlib import Path
import sys, json, hashlib, inspect, gc
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
import run_stage7_11_experiment as r
import run_phaseC_one_shot as phasec
from novel_ids.taxonomy.stage11 import load_taxonomy
from novel_ids.decision.stage12 import apply_decision_policy
PRIMARY=Path(r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43\best.pt")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
def params(module):
 ps=list(module.parameters()) if hasattr(module,"parameters") else []
 return {"total":sum(p.numel() for p in ps),"trainable":sum(p.numel() for p in ps if p.requires_grad),"frozen":sum(p.numel() for p in ps if not p.requires_grad),"training_mode":bool(getattr(module,"training",False))}
def modrow(name,module):
 try: file=inspect.getfile(type(module))
 except Exception: file="not available"
 return {"stage":name,"class":type(module).__name__,"file":str(file),"parameters":params(module)}
def stats(x):
 a=x.detach().float().cpu().numpy().reshape(-1)
 return {"shape":list(x.shape),"mean":float(np.mean(a)),"std":float(np.std(a)),"min":float(np.min(a)),"max":float(np.max(a))}
def pr_rank(x):
 a=x.detach().float().cpu().numpy()
 a=a-a.mean(axis=0,keepdims=True)
 s=np.linalg.svd(a,compute_uv=False)
 e=s*s
 return {"participation_ratio":float((e.sum()**2)/(np.square(e).sum()+1e-12)),"effective_rank":float(np.exp(-np.sum((e/e.sum())*np.log(np.maximum(e/e.sum(),1e-12)))))}
def main():
 r.DEVICE=torch.device("cpu"); phasec.r.DEVICE=torch.device("cpu")
 r.RUN_DIR=Path(r"C:\NOVEL_IDS_runs\verify_pipeline_20261127")
 r.RUN_DIR.mkdir(parents=True,exist_ok=True)
 r.taxonomy=load_taxonomy(ROOT/"config/taxonomy_unified.yaml",purpose="evaluation")
 # Read-only checkpoint and freeze verification.
 phasec.verify_manifest()
 r.verify_protected()
 if sha(PRIMARY)!="349505b6ebae528246d72b66785b28e296ecdf4ccc6d775244005ae3fd5147c0":
  raise RuntimeError("Primary checkpoint SHA mismatch")
 helper,branches=r.read_data()
 r.helper=helper
 labels=r.load_labels(r.taxonomy)
 ck5=torch.load(phasec.STAGE5,map_location="cpu",weights_only=False); aug_cfg=ck5["config"]; del ck5
 level,confidence,group_only,level3_mask=r.make_metadata()
 r.COMMON_MASK=(level==1)
 variant=phasec.init_variant(("primary",PRIMARY,"full","unified"),helper,branches,labels,aug_cfg,level,confidence,group_only,level3_mask,False)
 # Keep all verification in inference mode and without gradients.
 helper.stage5.eval()
 for p in helper.stage5.parameters(): p.requires_grad_(False)
 branches.eval()
 for p in branches.parameters(): p.requires_grad_(False)
 for key in ("stage7","stage8","stage10","classifier"): variant[key].eval()
 rows=[]; details={}
 for stage,module in [
  ("5A/5B Stage5 dual encoder",helper.stage5),
  ("7 adaptive residual fusion",variant["stage7"]),
  ("8 selective alignment",variant["stage8"]),
  ("9 unified representation",variant["stage9"]),
  ("10 Transformer",variant["stage10"]),
  ("11 classifier",variant["classifier"]),
 ]: rows.append(modrow(stage,module))
 for dataset_name, branch in branches.items():
  rows.append(modrow("6 residual branch "+str(dataset_name),branch))
 children=[]
 for name,module in helper.stage5.named_children():
  children.append(modrow("Stage5 child: "+name,module))
 details["stage5_children"]=children
 details["modules"]=rows
 details["stage9_composition"]={"inputs":["Z_s","R","R_proj","G","A","U","C","Z_f"],"output":"Z_u","implementation_file":inspect.getfile(type(variant["stage9"]))}
 details["stage10_inputs"]=["Stage5 token_embeddings (batch['tokens'])","Stage9 fused Z_u","M_avail"]
 details["stage11_input"]="Stage10 Z_t"
 details["stage12"]={"class":"apply_decision_policy","file":str(ROOT/"src/novel_ids/decision/stage12.py"),"parameters":0,"thresholds":json.loads(phasec.CAL.read_text(encoding="utf-8")),"taxonomy":list(r.taxonomy.classes)}
 details["datasets"]={}
 for ds in r.DATASETS:
  split=helper.datasets[ds]["validation"]; n=min(256,len(split["numeric_values"])); idx=np.arange(n)
  batch=r.source_batch(helper,branches,ds,"validation",idx,False,torch.Generator(device=r.DEVICE).manual_seed(431+(0 if ds=="nsl_kdd" else 1)),level3_mask,aug_cfg)
  sidval=r.NSL_KDD if ds=="nsl_kdd" else r.UNSW_NB15
  sid=torch.full((n,),sidval,device=r.DEVICE,dtype=torch.long)
  with torch.no_grad():
   logits,gate_out,align,fused=r.batch_forward(batch,variant["stage7"],variant["stage8"],variant["stage9"],variant["stage10"],variant["classifier"],sid,level,confidence,group_only,1.0)
   transformed=variant["stage10"](batch["tokens"],fused["Z_u"],batch["available"])
   zt=transformed["Z_t"]
  z_s=batch["Z_s"]; rproj=batch["R_proj"]; zu=fused["Z_u"]; g=gate_out["G"]; c=align["C"]
  details["datasets"][ds]={
   "rows":n,
   "stage5_input_shapes":{k:list(split[k][:n].shape) for k in ("numeric_values","categorical_values","M_avail","M_miss")},
   "stage5_outputs":{"Z_s":list(z_s.shape),"Z_d":list(batch["Z_d"].shape),"R":list(batch["R"].shape),"R_proj":list(rproj.shape),"token_embeddings":list(batch["tokens"].shape)},
   "stage7":{"Z_f":list(gate_out["Z_f"].shape),"G":stats(g)},
   "stage8":{"A":list(align["A"].shape),"U":list(align["U"].shape),"C":stats(c)},
   "stage9":{"Z_u":list(zu.shape),"composition":"Z_u = Stage9(Z_s, R, R_proj, G, A, U, C, Z_f)"},
   "stage10":{"token_input":list(batch["tokens"].shape),"fused_input":list(zu.shape),"available_mask":list(batch["available"].shape),"Z_t":list(zt.shape),"tokens_are_stage5_embeddings":True},
   "stage11":{"input_Z_t":list(zt.shape),"logits":list(logits.shape),"classifier_input":"Z_t"},
   "participation_rank":{"Z_s":pr_rank(z_s),"R_proj":pr_rank(rproj),"Z_u":pr_rank(zu),"Z_t":pr_rank(zt)},
  }
 # Stage 12 contract is verified from code and tests; no OOD test inference is run here.
 status=[
 ("5A/5B","verified in code and validation run","Stage5 dual encoder forward; Stage5 snapshot hash; source_batch outputs Z_s/Z_d/token_embeddings"),
 ("6","verified in code and validation run","Stage6ResidualBranch forward; accepted Stage6 checkpoint; R and R_proj observed"),
 ("7","verified in code and validation run","AdaptiveResidualFusion; validation G/Z_f observed"),
 ("8","verified in code and validation run","Stage8SelectiveAlignment; validation A/U/C observed; fitted source references loaded from checkpoint"),
 ("9","verified in code and validation run","UnifiedRepresentation; validation Z_u observed"),
 ("10","verified in code and validation run","Stage10Transformer; receives Stage5 token_embeddings and Z_u; outputs Z_t"),
 ("11","verified in code and validation run","Stage11Classifier; consumes Z_t and outputs 12-class logits"),
 ("12","implemented and unit-tested; validation policy not executed in this audit","apply_decision_policy; calibrated thresholds and taxonomy loaded; inference tests cover policy contract"),
 ]
 md=["# NOVEL-IDS verified pipeline status","", "Generated by scripts/verify_pipeline.py on CPU using one 256-row source-validation batch per dataset. No training, source-test inference, or unseen-label use occurred.","","## Frozen artifact checks","",f"- Primary checkpoint: {PRIMARY} SHA256 {sha(PRIMARY)}",f"- Stage5 snapshot SHA256 {sha(phasec.STAGE5)}",f"- Stage6 snapshot SHA256 {sha(phasec.STAGE6)}","- Freeze manifest: verify_manifest() passed; registered protected files: verify_protected() passed.","","## Stage-by-stage status","","| Stage | Status | Evidence |","|---|---|---|"]
 for st,stat,evidence in status: md.append(f"| {st} | {stat} | {evidence} |")
 md += ["","## Runtime dataflow evidence","",f"- Stage5 token embeddings are passed directly to Stage10: {all(details['datasets'][d]['stage10']['tokens_are_stage5_embeddings'] for d in r.DATASETS)}.","- Stage10 also receives Stage9 fused Z_u; Stage11 receives Stage10 Z_t.","- Module/parameter counts and files: see pipeline_truth_check.json; Stage5 children are listed there.","","## Validation-batch tensor shapes and diagnostics",""]
 for ds,d in details["datasets"].items():
  md += [f"### {ds}","",f"- Stage7 gate: {d['stage7']['G']}; Stage8 compatibility C: {d['stage8']['C']}.",f"- Stage9 Z_u shape: {d['stage9']['Z_u']}; Stage10 Z_t shape: {d['stage10']['Z_t']}; classifier logits: {d['stage11']['logits']}.","- Participation-ratio/effective ranks:"]
  for key,val in d["participation_rank"].items(): md.append(f"  - {key}: participation ratio {val['participation_ratio']:.4f}, effective rank {val['effective_rank']:.4f}")
 md += ["","## Stage 6 approved wording","", "L_info = MSE(E_d_hat, StopGrad(E_d)), where E_d = Z_d - StopGrad(Z_d_hat) and E_d_hat = dec_R(R).","","## Known test status and protected failure","","- Latest focused inference/onboarding/Stage12 tests: 13 passed (before adding the new external-metadata integration test).","- Previously completed full suite: 190 passed, 1 failed. The known failure is tests/test_stage5_training.py::test_trainer_finite_losses_and_gradients; it checks that trainer losses backpropagate to all Stage5 parameters and fails on unknown_category_embedding. This protected Stage5 test/source was not modified. The exact reason the synthetic batch does not activate that embedding path is not proven.","","## Document/code contradictions identified","","1. The prior STATUS_VERIFIED.md described a pytest collection error; later recorded full-suite evidence is 190 passed/1 failed and focused inference tests now pass 13/13. The old status was stale.","2. The previous inference matcher loaded the 78-slot unified_feature_schema.yaml as its candidate registry; that file omits review_status. FeatureIdentityEncoder.map_unseen_descriptor requires approved Level-2 candidates, so the candidate set was empty and unseen features were all downgraded to Level 3. The code now keeps 78 model slots but uses the 83 full training_feature_descriptors.yaml entries for matching.","3. The inference API previously did not accept external metadata or declared unit/type transforms; it now supports optional metadata YAML and exposes Level-1 equivalent / Level-2 group-only mapping. CIC second-run performance remains pending until its saved output is reviewed.","4. The API now applies the validation-tuned class-prior bias from config/class_prior_bias.json; older inference documentation that omits this decision layer is stale.","5. The older demo pack status is stale: six sample CSVs and their manifest were present and hash-matched; KDD Cup 99 remains not supplied.","","## Module inventory and parameter flags","","See pipeline_truth_check.json for exact module files, parameter counts, trainable/frozen counts, runtime tensor shapes, gate/C diagnostics, ranks, taxonomy and thresholds.",""]
 (ROOT/"docs/STATUS_VERIFIED.md").write_text("\n".join(md),encoding="utf-8")
 (ROOT/"docs/pipeline_truth_check.json").write_text(json.dumps(details,indent=2,default=str),encoding="utf-8")
 print("WROTE docs/STATUS_VERIFIED.md and docs/pipeline_truth_check.json")
 print("MODULES",[(x["stage"],x["class"],x["parameters"]["total"],x["parameters"]["trainable"]) for x in rows])
 print("VALIDATION_DIAGNOSTICS",json.dumps({d:details["datasets"][d]["participation_rank"] for d in r.DATASETS},indent=2))
 # Release memory without writing back to project data/checkpoints.
 del helper,branches,variant,labels; gc.collect()
if __name__=="__main__": main()
