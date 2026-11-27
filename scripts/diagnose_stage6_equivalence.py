from __future__ import annotations
import copy, hashlib, json, sys
from pathlib import Path
import numpy as np, torch, yaml
ROOT=Path(r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project"); sys.path.insert(0,str(ROOT/"src"))
from novel_ids.representation.dual_encoder import DualEncoder
from novel_ids.residual.stage6 import Stage6ResidualBranch,freeze_stage5,prediction_loss,information_reconstruction_loss,magnitude_stability_loss,standardized_cross_covariance_loss,projection_geometry_loss,Stage6LossCoefficients,combine_stage6_losses_with_projection
from novel_ids.residual.stage6_decorr import residual_decorrelation_loss
from novel_ids.residual.stage6_fastpath import make_augmentation_masks,stage5_encode_augmented
from novel_ids.residual.stage6_data import load_stage6_input_artifact
SNAP=Path(r"C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt"); CKPT=Path(r"C:\NOVEL_IDS_runs\stage6\phase2_decorr_20261125\candidate09_lambda0.1_seed42\best.pt")
CACHE=Path(r"C:\NOVEL_IDS_runs\stage6\phase2_fastcache_20261125\cache"); INPUT=ROOT/"data"/"interim"/"stage6_trainval_only"; SCHEMA=ROOT/"data"/"processed"/"unified_representation"/"unified_feature_schema.yaml"; SEED=20261125
torch.manual_seed(42); np.random.seed(42); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
def sh(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def calc(b,zs,zd):
 o=b(zs.float(),zd.float()); p=prediction_loss(o["Z_d_hat"],zd); i=information_reconstruction_loss(o["E_d_hat"],o["E_d"]); m,_=magnitude_stability_loss(o["R"]); r,rs=standardized_cross_covariance_loss(zs,o["R"],variance_floor=1e-4); d,ds=residual_decorrelation_loss(o["R"],variance_floor=1e-4); q=projection_geometry_loss(o["R"],o["R_proj_objective"])
 t=combine_stage6_losses_with_projection(l_pred=p,l_info=i,l_mag=m,l_red=r,l_proj=q,coefficients=Stage6LossCoefficients(1,1,1,1),projection_coefficient=1)+0.1*d
 return {"L_pred":p,"L_info":i,"L_mag":m,"L_red":r,"L_decorr":d,"L_proj":q,"total":t},o,{"red":rs,"decorr":ds}
def fv(x): return {k:float(v.detach().cpu()) for k,v in x.items()}
def compare(a,b):
 return {k:{"cached_or_cpu":a[k],"direct_or_gpu":b[k],"abs":abs(a[k]-b[k]),"rel":abs(a[k]-b[k])/max(abs(a[k]),abs(b[k]),1e-12)} for k in a}
st=torch.load(SNAP,map_location="cpu",weights_only=False); cfg=st["config"]; ids=np.load(ROOT/"data"/"processed"/"unified_representation"/"feature_identity_vectors.npy",allow_pickle=False)
s=DualEncoder(ids,dropout=float(cfg["dropout"]),value_masking_rate=float(cfg["value_masking_rate"]),feature_set_dropout_rate=float(cfg["feature_set_dropout_rate"]),feature_set_dropout_all_probability=float(cfg["feature_set_dropout_all_probability"])); s.load_state_dict(st["dual_encoder"],strict=True); del st
sg=freeze_stage5(copy.deepcopy(s).cuda().eval()); sc=freeze_stage5(copy.deepcopy(s).cpu().eval())
ck=torch.load(CKPT,map_location="cpu",weights_only=False); dslist=("nsl_kdd","unsw_nb15")
branches=torch.nn.ModuleDict({d:Stage6ResidualBranch(128,128,128,128,128,128) for d in dslist}); branches.load_state_dict(ck["branches"],strict=True); branches.eval()
print("WEIGHTS",json.dumps({d:{part:{"sha256":sh(getattr(branches[d],part)),"type":type(getattr(branches[d],part)).__name__} for part in ("predictor","encoder","decoder","projection")} for d in dslist},sort_keys=True),flush=True)
print("SETTINGS",{"stage5_mode":"eval","stage6_mode":"eval","autocast":False,"tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"cuda_available":torch.cuda.is_available()},flush=True)
schema=yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
for di,ds in enumerate(dslist):
 a=load_stage6_input_artifact(INPUT/f"{ds}_inputs.joblib",ds)["train"]; l3=[int(x["index"]) for x in schema["features"] if x.get("dataset")==ds and int(x.get("level",0))==3]
 draw=1; seed=SEED+di*10000+draw*101
 # Exact cache-generation RNG call: 512 rows in the first cache batch.
 r512=np.random.default_rng(seed); av512,mi512,dr512,vm512=make_augmentation_masks(a["M_avail"][:512],a["M_miss"][:512],a["numeric_token_mask"],a["categorical_token_mask"],l3,r512,float(cfg["feature_set_dropout_rate"]),float(cfg["feature_set_dropout_all_probability"]),float(cfg["value_masking_rate"]))
 # Recreate the former incorrect 256-row mask generation for a diagnostic contrast only.
 r256=np.random.default_rng(seed); av256,mi256,dr256,vm256=make_augmentation_masks(a["M_avail"][:256],a["M_miss"][:256],a["numeric_token_mask"],a["categorical_token_mask"],l3,r256,float(cfg["feature_set_dropout_rate"]),float(cfg["feature_set_dropout_all_probability"]),float(cfg["value_masking_rate"]))
 same={n:bool(np.array_equal(x[:256],y)) for n,x,y in (("available",av512,av256),("missing",mi512,mi256),("feature_drop",dr512,dr256),("value_mask",vm512,vm256))}
 print("MASK_SEQUENCE",ds,{"seed":seed,"draw":draw,"cache_generation_batch":512,"old_harness_batch":256,"same_first256":same,"cache_mask_counts":{"feature_drop":int(dr512[:256].sum()),"value_mask":int(vm512[:256].sum())},"old_harness_counts":{"feature_drop":int(dr256.sum()),"value_mask":int(vm256.sum())}},flush=True)
 # Use the identical cache-generated mask contents for direct forward and cached tensors.
 kw=dict(numeric_values=a["numeric_values"][:256],categorical_values=a["categorical_values"][:256],available=av512[:256].copy(),missing=mi512[:256].copy(),numeric_token_mask=a["numeric_token_mask"],categorical_token_mask=a["categorical_token_mask"],feature_drop_mask=dr512[:256].copy(),dataset=ds)
 zg,zdg=stage5_encode_augmented(sg,**kw); zc,zdc=stage5_encode_augmented(sc,**kw)
 mm=np.load(CACHE/ds/"train_zszd.npy",mmap_mode="r"); cache=np.asarray(mm[:256,draw,:],dtype=np.float32)
 zca=torch.as_tensor(cache[:,:128],device="cuda"); zda=torch.as_tensor(cache[:,128:],device="cuda")
 zcu=zca.detach().cpu(); zdu=zda.detach().cpu()
 zdiff={"cache_vs_direct_gpu":{"zs":float((zca-zg).abs().max()),"zd":float((zda-zdg).abs().max())},"cpu_vs_gpu_direct":{"zs":float((zc.cpu()-zg.cpu()).abs().max()),"zd":float((zdc.cpu()-zdg.cpu()).abs().max())}}
 print("EMBEDDINGS",ds,{"cache_dtype":str(mm.dtype),"loaded_dtype":str(cache.dtype),**zdiff},flush=True)
 bg=copy.deepcopy(branches[ds]).cuda().eval(); bc=copy.deepcopy(branches[ds]).cpu().eval()
 with torch.autocast(device_type="cuda",enabled=False): lc,oc,scs=calc(bg,zca,zda); ld,od,sds=calc(bg,zg,zdg)
 lcpu,ocpu,scpu=calc(bc,zcu,zdu)
 vc,vd,vu=fv(lc),fv(ld),fv(lcpu)
 print("LOSS_CACHE_VS_DIRECT_GPU",ds,json.dumps(compare(vc,vd),sort_keys=True),flush=True)
 print("LOSS_CPU_VS_GPU_DIRECT",ds,json.dumps(compare(vu,vc),sort_keys=True),flush=True)
 print("MASK_STATS",ds,{"cached":scs,"direct":sds,"cpu":scpu},flush=True)
 # Exact discrete masks for shared/residual variance floor.
 def mask_ids(zs,o):
  zz=zs.detach().float(); rr=o["R"].detach().float()
  zv=(zz-zz.mean(0)).square().mean(0)>=1e-8; rv=(rr-rr.mean(0)).square().mean(0)>=1e-8
  cov=(rr-rr.mean(0)).T@(rr-rr.mean(0))/255; dv=torch.diagonal(cov)>=1e-8
  return {"shared_masked":torch.where(~zv)[0].cpu().tolist(),"residual_red_masked":torch.where(~rv)[0].cpu().tolist(),"residual_decorr_masked":torch.where(~dv)[0].cpu().tolist()}
 print("VARIANCE_MASK_IDS",ds,{"cache":mask_ids(zca,oc),"direct_gpu":mask_ids(zg,od),"cpu":mask_ids(zcu,ocpu)},flush=True)
 bg.zero_grad(set_to_none=True); lg,_,_=calc(bg,zca,zda); lg["total"].backward(); gg=bg.encoder.network[0].weight.grad.detach().cpu()
 bc.zero_grad(set_to_none=True); l2,_,_=calc(bc,zcu,zdu); l2["total"].backward(); gc=bc.encoder.network[0].weight.grad.detach().cpu()
 rel=float(torch.linalg.vector_norm(gg-gc)/torch.linalg.vector_norm(gg).clamp_min(1e-12))
 print("H_FIRST_LAYER_GRAD",ds,{"relative_l2":rel,"max_abs":float((gg-gc).abs().max()),"tolerance":1e-3,"passed":rel<=1e-3},flush=True)
 del a,mm,bg,bc
 torch.cuda.empty_cache()
print("DIAGNOSTIC_COMPLETE",flush=True)



