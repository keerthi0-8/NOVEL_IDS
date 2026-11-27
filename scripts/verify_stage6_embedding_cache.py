from __future__ import annotations
import copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch, yaml
ROOT=Path(r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project"); sys.path.insert(0,str(ROOT/"src"))
from novel_ids.representation.dual_encoder import DualEncoder
from novel_ids.residual.stage6 import Stage6ResidualBranch, freeze_stage5, prediction_loss, information_reconstruction_loss, magnitude_stability_loss, standardized_cross_covariance_loss, projection_geometry_loss, Stage6LossCoefficients, combine_stage6_losses_with_projection
from novel_ids.residual.stage6_decorr import residual_decorrelation_loss
from novel_ids.residual.stage6_fastpath import make_augmentation_masks, stage5_encode_augmented, sha256_file
from novel_ids.residual.stage6_data import load_stage6_input_artifact
SNAP=Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt")
CKPT=Path(r"C:\NOVEL_IDS_runs\stage6\phase2_decorr_20261125\candidate09_lambda0.1_seed42\best.pt")
OUT=Path(r"C:\NOVEL_IDS_runs\stage6\phase2_fastcache_20261125")
CACHE=OUT/"cache"; INPUT=ROOT/"data"/"interim"/"stage6_trainval_only"; SCHEMA=ROOT/"data"/"processed"/"unified_representation"/"unified_feature_schema.yaml"
SEED_BASE=20261125; DRAW=1; N=256; CACHE_BATCH=512; FLOOR=1e-4
torch.manual_seed(42); np.random.seed(42)
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
if (OUT/"cache_manifest.json").exists(): raise FileExistsError("Refusing to overwrite cache manifest")
def state_hash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):
  h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def calc(b,zs,zd):
 o=b(zs.float(),zd.float())
 p=prediction_loss(o["Z_d_hat"],zd); i=information_reconstruction_loss(o["E_d_hat"],o["E_d"]); m,_=magnitude_stability_loss(o["R"])
 r,rs=standardized_cross_covariance_loss(zs,o["R"],variance_floor=FLOOR); d,ds=residual_decorrelation_loss(o["R"],variance_floor=FLOOR)
 q=projection_geometry_loss(o["R"],o["R_proj_objective"])
 total=combine_stage6_losses_with_projection(l_pred=p,l_info=i,l_mag=m,l_red=r,l_proj=q,coefficients=Stage6LossCoefficients(1,1,1,1),projection_coefficient=1)+0.1*d
 return {"L_pred":p,"L_info":i,"L_mag":m,"L_red":r,"L_decorr":d,"L_proj":q,"total":total},o,{"red":rs,"decorr":ds}
def values(d): return {k:float(v.detach().cpu()) for k,v in d.items()}
def absrel(a,b):
 return {k:{"a":a[k],"b":b[k],"abs":abs(a[k]-b[k]),"rel":abs(a[k]-b[k])/max(abs(a[k]),abs(b[k]),1e-12)} for k in a}
def floor_report(zs,r):
 z=zs.detach().float(); x=r.detach().float()
 zv=(z-z.mean(0)).square().mean(0).sqrt(); rv=(x-x.mean(0)).square().mean(0).sqrt()
 cov=(x-x.mean(0)).T@(x-x.mean(0))/float(x.shape[0]-1); dv=torch.diagonal(cov).clamp_min(0).sqrt()
 def item(v):
  near=torch.where((v-FLOOR).abs()<=1e-6)[0].cpu().tolist()
  masked=torch.where(v<FLOOR)[0].cpu().tolist()
  return {"masked_indices":masked,"near_threshold_1e-6_indices":near,"min_std":float(v.min().cpu()),"min_abs_distance_to_floor":float((v-FLOOR).abs().min().cpu()),"dimensions":int(v.numel())}
 return {"shared_Lred":item(zv),"residual_Lred":item(rv),"residual_Ldecorr":item(dv)}
if sha256_file(str(SNAP))!="9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b": raise RuntimeError("Stage5 snapshot hash mismatch")
stage5ck=torch.load(SNAP,map_location="cpu",weights_only=False); cfg=stage5ck["config"]
ident=np.load(ROOT/"data"/"processed"/"unified_representation"/"feature_identity_vectors.npy",allow_pickle=False)
s5=DualEncoder(ident,dropout=float(cfg["dropout"]),value_masking_rate=float(cfg["value_masking_rate"]),feature_set_dropout_rate=float(cfg["feature_set_dropout_rate"]),feature_set_dropout_all_probability=float(cfg["feature_set_dropout_all_probability"]))
s5.load_state_dict(stage5ck["dual_encoder"],strict=True); del stage5ck
s5=freeze_stage5(s5).cuda().eval()
ck=torch.load(CKPT,map_location="cpu",weights_only=False)
datasets=("nsl_kdd","unsw_nb15")
branches=torch.nn.ModuleDict({d:Stage6ResidualBranch(128,128,128,128,128,128) for d in datasets})
branches.load_state_dict(ck["branches"],strict=True); branches.eval()
schema=yaml.safe_load(SCHEMA.read_text(encoding="utf-8")); report={}
print("SETTINGS",{"stage5_mode":"eval","stage6_mode":"eval","autocast_enabled":False,"tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"device":"cuda","batch_rows_per_dataset":N,"cache_generation_batch":CACHE_BATCH},flush=True)
for di,ds in enumerate(datasets):
 arr=load_stage6_input_artifact(INPUT/f"{ds}_inputs.joblib",ds)["train"]
 if len(arr["numeric_values"])<CACHE_BATCH: raise RuntimeError(f"{ds}: cache first batch shorter than 512")
 l3=[int(x["index"]) for x in schema["features"] if x.get("dataset")==ds and int(x.get("level",0))==3]
 seed=SEED_BASE+di*10000+DRAW*101
 def masks(batch,seed0):
  return make_augmentation_masks(arr["M_avail"][:batch],arr["M_miss"][:batch],arr["numeric_token_mask"],arr["categorical_token_mask"],l3,np.random.default_rng(seed0),float(cfg["feature_set_dropout_rate"]),float(cfg["feature_set_dropout_all_probability"]),float(cfg["value_masking_rate"]))
 av,mi,drop,vmark=masks(CACHE_BATCH,seed)
 av2,mi2,drop2,vmark2=masks(CACHE_BATCH,seed)
 deterministic=all(np.array_equal(x,y) for x,y in ((av,av2),(mi,mi2),(drop,drop2),(vmark,vmark2)))
 # Demonstrate that the previous 256-row harness consumed a different RNG sequence.
 av_old,mi_old,drop_old,vmark_old=masks(N,seed)
 old_vs_cache={k:bool(np.array_equal(x[:N],y)) for k,x,y in (("available",av,av_old),("missing",mi,mi_old),("feature_drop",drop,drop_old),("value_mask",vmark,vmark_old))}
 kw={"numeric_values":arr["numeric_values"][:CACHE_BATCH],"categorical_values":arr["categorical_values"][:CACHE_BATCH],"available":av,"missing":mi,"numeric_token_mask":arr["numeric_token_mask"],"categorical_token_mask":arr["categorical_token_mask"],"feature_drop_mask":drop,"dataset":ds}
 zsfull,zdfull=stage5_encode_augmented(s5,**kw)
 zsdirect=zsfull[:N].contiguous(); zddirect=zdfull[:N].contiguous()
 cache_np=np.load(CACHE/ds/"train_zszd.npy",mmap_mode="r")
 if cache_np.dtype!=np.float32: raise RuntimeError(f"{ds}: cache dtype is not float32")
 cache_rows=np.array(cache_np[:N,DRAW,:],dtype=np.float32,copy=True)
 zscache=torch.as_tensor(cache_rows[:,:128],device="cuda"); zdcache=torch.as_tensor(cache_rows[:,128:],device="cuda")
 direct_embed={"zs":float((zscache-zsdirect[:N]).abs().max().cpu()),"zd":float((zdcache-zddirect[:N]).abs().max().cpu())}
 branch_gpu=copy.deepcopy(branches[ds]).cuda().eval()
 branch_cpu=copy.deepcopy(branches[ds]).cpu().eval()
 hashes={part:{"type":type(getattr(branches[ds],part)).__name__,"sha256":state_hash(getattr(branches[ds],part))} for part in ("predictor","encoder","decoder","projection")}
 if any(state_hash(getattr(branch_gpu,part))!=hashes[part]["sha256"] for part in hashes): raise RuntimeError(f"{ds}: GPU branch weights differ from checkpoint state")
 if any(state_hash(getattr(branch_cpu,part))!=hashes[part]["sha256"] for part in hashes): raise RuntimeError(f"{ds}: CPU branch weights differ from checkpoint state")
 with torch.autocast(device_type="cuda",enabled=False):
  gpu_cache,ogc,stats_gc=calc(branch_gpu,zscache,zdcache)
  gpu_direct,ogd,stats_gd=calc(branch_gpu,zsdirect[:N],zddirect[:N])
 cpu_zs=zscache.detach().cpu().contiguous(); cpu_zd=zdcache.detach().cpu().contiguous()
 cpu_loss,oc,stats_cpu=calc(branch_cpu,cpu_zs,cpu_zd)
 same_device=absrel(values(gpu_cache),values(gpu_direct))
 cpu_gpu=absrel(values(cpu_loss),values(gpu_cache))
 grad_gpu_total,_,_=calc(branch_gpu,zscache,zdcache); branch_gpu.zero_grad(set_to_none=True); grad_gpu_total["total"].backward()
 grad_gpu=branch_gpu.encoder.network[0].weight.grad.detach().cpu().clone()
 grad_cpu_total,_,_=calc(branch_cpu,cpu_zs,cpu_zd); branch_cpu.zero_grad(set_to_none=True); grad_cpu_total["total"].backward()
 grad_cpu=branch_cpu.encoder.network[0].weight.grad.detach().cpu().clone()
 grad_rel=float(torch.linalg.vector_norm(grad_cpu-grad_gpu)/torch.linalg.vector_norm(grad_gpu).clamp_min(1e-12))
 floor=floor_report(zscache,ogc["R"])
 max_same=max(x["abs"] for k,x in same_device.items() if k!="total")
 max_cpu_rel=max(x["rel"] for x in cpu_gpu.values())
 checks={"mask_reproducible_same512_seed":deterministic,"old256_harness_mask_match":old_vs_cache,"cache_dtype_float32":str(cache_np.dtype)=="float32","cache_vs_direct_embedding_max_abs":direct_embed,"same_device_loss_max_abs_by_term":{k:v["abs"] for k,v in same_device.items()},"cpu_gpu_loss_relative_by_term":{k:v["rel"] for k,v in cpu_gpu.items()},"h_first_layer_gradient_relative_l2":grad_rel,"variance_floor_masks":floor}
 passed=deterministic and str(cache_np.dtype)=="float32" and max_same<=1e-5 and max_cpu_rel<=1e-4 and grad_rel<=1e-3 and all(not floor[g]["masked_indices"] for g in floor)
 if not passed: raise RuntimeError(f"{ds} equivalence failed: {json.dumps(checks)}")
 report[ds]={"rows":list(range(N)),"batch_size":N,"augmentation_draw":DRAW,"seed":seed,"stage5_forward_batch_size":CACHE_BATCH,"mask_counts":{"feature_drop_first256":int(drop[:N].sum()),"value_mask_first256":int(vmark[:N].sum())},"old_256_harness_mask_counts":{"feature_drop":int(drop_old.sum()),"value_mask":int(vmark_old.sum())},"mask_reproducible_same512_seed":deterministic,"old_256_harness_mask_match":old_vs_cache,"branch_weight_hashes":hashes,"cache_dtype":str(cache_np.dtype),"cache_sha256":sha256_file(str(CACHE/ds/"train_zszd.npy")),"cache_shape":list(cache_np.shape),"direct_embedding_max_abs":direct_embed,"same_device_cached_vs_direct":same_device,"cpu_gpu_same_input":cpu_gpu,"h_first_layer_gradient":{"relative_l2":grad_rel,"max_abs":float((grad_cpu-grad_gpu).abs().max()),"tolerance":1e-3,"passed":grad_rel<=1e-3},"variance_floor_masks":floor,"passed":True}
 print("DATASET_EQUIVALENCE_PASS",ds,json.dumps(report[ds],sort_keys=True),flush=True)
 del arr,cache_np,branch_gpu,branch_cpu
 torch.cuda.empty_cache()
manifest={"status":"CACHE_EQUIVALENCE_PASSED","created_at_local":time.strftime("%Y-%m-%d %H:%M:%S"),"stage5_snapshot_sha256":sha256_file(str(SNAP)),"stage6_checkpoint_sha256":sha256_file(str(CKPT)),"stage6_checkpoint":"candidate09_lambda0.1_seed42/best.pt","cache_root":str(CACHE),"cache_generation_batch_size":CACHE_BATCH,"stage6_equivalence_batch_size":N,"augmentation_draw":DRAW,"autocast_disabled":True,"tf32_disabled":True,"tolerances":{"same_device_abs_per_term":1e-5,"cpu_gpu_relative_per_term":1e-4,"h_first_layer_gradient_relative_l2":1e-3},"datasets":report}
manifest_path=OUT/"cache_manifest.json"; tmp=manifest_path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8"); tmp.replace(manifest_path)
print("ALL_EQUIVALENCE_CHECKS_PASSED",json.dumps({"manifest":str(manifest_path),"datasets":list(report),"same_device_tolerance":1e-5,"cpu_gpu_relative_tolerance":1e-4,"gradient_relative_tolerance":1e-3},indent=2),flush=True)
