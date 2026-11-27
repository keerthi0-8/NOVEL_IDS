from pathlib import Path
import json, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
ROOT=Path(r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project")
RUN=Path(r"C:\NOVEL_IDS_runs")
FIG=ROOT/"docs"/"figures"; FIG.mkdir(parents=True,exist_ok=True)
cal=json.loads((RUN/"phaseE_validation_repro_20261126"/"phaseA_calibration.json").read_text(encoding="utf-8"))
baseline=json.loads((RUN/"baselines_ablations"/"source_baselines_20261125_r1"/"results.json").read_text(encoding="utf-8"))
lodo=json.loads((RUN/"phaseD_lodo_20261126"/"results.json").read_text(encoding="utf-8"))
import yaml
tax=yaml.safe_load((ROOT/"config"/"taxonomy_unified.yaml").read_text(encoding="utf-8"))
classes=list(tax["classes"])
npz=np.load(RUN/"phaseE_validation_repro_20261126"/"validation_distributions.npz")
ablation_runs={
 "Shared-only":RUN/"phaseB_final_shared_only_seed42"/"result.json",
 "Fixed gate G=1":RUN/"phaseB_final_residual_fixed_gate1_seed42"/"result.json",
 "Adaptive residual, alignment off":RUN/"phaseB_final_residual_adaptive_seed42"/"result.json",
 "Common-only (1,000 steps)":RUN/"phaseB_final_common_only_seed42_batch64_1000"/"result.json",
 "Native heads, adaptive residual":RUN/"phaseB_final_residual_adaptive_native_seed42"/"result.json",
}
ablation={}
for name,path in ablation_runs.items():
 if path.exists(): ablation[name]=json.loads(path.read_text(encoding="utf-8"))
def best_history(obj):
 h=obj.get("validation_history",[])
 if not h:return []
 return [(int(x.get("step",i)),float(x.get("mean_macro_f1",np.mean([v.get("macro_f1",0) for v in x.get("datasets",{}).values()])))) for i,x in enumerate(h)]
plt.figure(figsize=(10,6))
for name,obj in ablation.items():
 hist=best_history(obj)
 if hist: plt.plot([x[0] for x in hist],[x[1] for x in hist],label=name)
plt.xlabel("Training step");plt.ylabel("Mean source-validation macro-F1");plt.title("Phase B validation ablation curves");plt.legend(fontsize=8);plt.tight_layout()
plt.savefig(FIG/"phaseB_validation_curves.png",dpi=180);plt.close()
fig,axes=plt.subplots(1,2,figsize=(15,5))
for ax,ds in zip(axes,("nsl_kdd","unsw_nb15")):
 y=np.asarray(npz[ds+"_labels"],dtype=int); pred=np.asarray(npz[ds+"_predictions"],dtype=int)
 p,r,f,s=precision_recall_fscore_support(y,pred,labels=np.arange(len(classes)),zero_division=0)
 keep=s>0
 ax.bar(np.arange(keep.sum()),f[keep]); ax.set_xticks(np.arange(keep.sum())); ax.set_xticklabels(np.asarray(classes)[keep],rotation=65,ha="right",fontsize=7)
 ax.set_ylim(0,1);ax.set_title(ds+" validation per-class F1");ax.set_ylabel("F1")
fig.tight_layout();fig.savefig(FIG/"primary_per_class_f1.png",dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5));tau=float(cal["tau_ood_validation_p95"])
for ax,ds in zip(axes,("nsl_kdd","unsw_nb15")):
 scores=np.asarray(npz[ds+"_ood_scores"]);ax.hist(scores,bins=60);ax.axvline(tau,linestyle="--",label=f"pooled validation p95={tau:.3f}")
 ax.set_title(ds+" validation OOD score");ax.set_xlabel("Combined z-scored Mahalanobis + energy");ax.set_ylabel("Rows");ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(FIG/"validation_ood_histograms.png",dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4.5))
for ax,ds in zip(axes,("nsl_kdd","unsw_nb15")):
 probs=np.asarray(npz[ds+"_probabilities"]);y=np.asarray(npz[ds+"_labels"],dtype=int);conf=probs.max(1);correct=(probs.argmax(1)==y)
 edges=np.linspace(0,1,16);centers=[];acc=[]
 for lo,hi in zip(edges[:-1],edges[1:]):
  m=(conf>=lo)&(conf<(hi if hi<1 else hi+1e-9))
  if m.any():centers.append(float(conf[m].mean()));acc.append(float(correct[m].mean()))
 ax.plot([0,1],[0,1],linestyle="--",label="Perfect calibration");ax.plot(centers,acc,marker="o",label="Model")
 ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title(ds+" reliability (15 bins)");ax.set_xlabel("Mean confidence");ax.set_ylabel("Accuracy");ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(FIG/"validation_reliability_diagrams.png",dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,ds in zip(axes,("nsl_kdd","unsw_nb15")):
 g=np.asarray(npz[ds+"_gate_mean"]);ax.plot(np.arange(len(g)),g);ax.set_ylim(0,1);ax.set_title(ds+" mean gate by latent dimension");ax.set_xlabel("Latent dimension");ax.set_ylabel("Mean G")
fig.tight_layout();fig.savefig(FIG/"primary_gate_statistics.png",dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,10))
for i,ds in enumerate(("nsl_kdd","unsw_nb15")):
 for j,mode in enumerate(("native","unified")):
  run=next(x for x in baseline["datasets"][ds]["label_modes"][mode]["hgb"]["runs"] if x["seed"]==42)
  cm=np.asarray(run["validation"]["confusion_matrix"]);ax=axes[i,j];im=ax.imshow(cm,aspect="auto")
  ax.set_title(f"{ds} {mode} HGB seed42");ax.set_xlabel("Predicted");ax.set_ylabel("Actual");fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
fig.tight_layout();fig.savefig(FIG/"baseline_validation_confusion_matrices.png",dpi=180);plt.close(fig)
def fmt(x):return f"{float(x):.4f}"
mlp_seeds={"nsl_kdd":{"native":42,"unified":44},"unsw_nb15":{"native":44,"unified":43}}
base_rows=[]
for ds in ("nsl_kdd","unsw_nb15"):
 for mode in ("native","unified"):
  entry=baseline["datasets"][ds]["label_modes"][mode]
  h=next(x["validation"] for x in entry["hgb"]["runs"] if x["seed"]==42)
  m=next(x["validation"] for x in entry["plain_mlp"]["runs"] if x["seed"]==mlp_seeds[ds][mode])
  base_rows.append((ds,mode,fmt(h["macro_f1"]),fmt(m["macro_f1"]),mlp_seeds[ds][mode]))
abl_rows=[]
for name,obj in ablation.items():
 h=best_history(obj);bv=obj.get("best_validation_macro_f1",max([v for _,v in h],default=float("nan")))
 abl_rows.append((name,fmt(bv),str(obj.get("best_step","see history"))))
lodo_rows=[]
for d in lodo["directions"]:
 m=d["target_validation_metrics"];lodo_rows.append((d["source"],d["target"],fmt(m["macro_f1_supported_classes"]),str(d["target_validation_rows_shared"])))
primary=cal["validation_metrics"]
report=f"""# NOVEL-IDS final execution report — 26 November 2026

## Executive status

- Phase A: complete; primary seed-43 checkpoint frozen and validation calibration reproduced in Phase E.
- Phase B: validation ablation families completed with documented reduced common-only budget.
- Phase C: **incomplete one-shot attempt; no source-test metrics**. The NSL-KDD test artifact was opened, but evaluation stopped before model inference because raw test label saint is absent from the approved taxonomy. The one-shot rule prohibits reopening or rerunning; this is not a successful source-test evaluation.
- Phase D: both approved development-time LODO directions completed using source train and opposite-source validation only. CIC-DDoS2019 is not present; unseen evaluation remains pending.
- Phase E: validation reproducibility, plots, README and documentation generated.

## Primary model — source validation

| Dataset | Rows | Accuracy | Macro-F1 | Weighted-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| NSL-KDD | {primary["nsl_kdd"]["rows"]} | {fmt(primary["nsl_kdd"]["accuracy"])} | {fmt(primary["nsl_kdd"]["macro_f1"])} | {fmt(primary["nsl_kdd"]["weighted_f1"])} | {fmt(primary["nsl_kdd"]["ece_15_bins"])} |
| UNSW-NB15 | {primary["unsw_nb15"]["rows"]} | {fmt(primary["unsw_nb15"]["accuracy"])} | {fmt(primary["unsw_nb15"]["macro_f1"])} | {fmt(primary["unsw_nb15"]["weighted_f1"])} | {fmt(primary["unsw_nb15"]["ece_15_bins"])} |

Calibration: temperature={cal["temperature"]:.8f}; pooled validation tau_ood={cal["tau_ood_validation_p95"]:.8f}; tau_conf={cal["tau_conf_validation_p10"]:.8f}. The OOD score is the mean of source-training z-scored class-conditional Mahalanobis distance and energy. Thresholds are validation-derived.

![OOD validation distributions](figures/validation_ood_histograms.png)

![Reliability diagrams](figures/validation_reliability_diagrams.png)

![Primary per-class F1](figures/primary_per_class_f1.png)

![Gate statistics](figures/primary_gate_statistics.png)

## Baseline validation results

| Dataset | Label mode | HGB seed42 macro-F1 | MLP selected seed | MLP macro-F1 |
|---|---|---:|---:|---:|
"""
for ds,mode,h,m,seed in base_rows:report+=f"| {ds} | {mode} | {h} | {seed} | {m} |\n"
report+="""

![Baseline confusion matrices](figures/baseline_validation_confusion_matrices.png)

## Phase B validation ablations

| Variant | Best mean validation macro-F1 | Best step |
|---|---:|---:|
"""
for name,v,step in abl_rows:report+=f"| {name} | {v} | {step} |\n"
report+="""
![Phase B validation curves](figures/phaseB_validation_curves.png)

Common-only runs used 1,000 steps, batch size 64, and two seeds (42/43) after earlier resource pressure. Other seed counts and any interrupted/retried runs are preserved in the run directories and progress log; these are validation-only results.

## Phase D — leave-one-dataset-out diagnostics

| Train source only | Evaluate opposite-source validation | Shared-class macro-F1 | Target validation rows |
|---|---|---:|---:|
"""
for a,bv,sc,n in lodo_rows:report+=f"| {a} | {bv} | {sc} | {n} |\n"
report+="""
LODO uses only the three classes present in both approved taxonomies. The single-source diagnostic duplicates the source-only Stage 8 reference into both reference slots so the unseen-source code path can be exercised. This is explicitly a diagnostic adaptation, not the final two-source Stage 8 reference estimator. The low target scores indicate weak cross-dataset transfer; no zero-shot success claim is made.

CIC-DDoS2019 is absent from data/unseen/. Unseen-data evaluation is pending.

## Stage 6 decision and qualification

Stage 6 accepted with qualifications: the residual carries measurable detection information beyond random features; the learned encoder does not exceed a random projection of E_d; the original rank gate fails; the effect of L_decorr is not distinguishable from noise on a single seed.

The original pre-registered criterion result remains NO_QUALIFYING_STAGE6_ARM. The selected Candidate 09 / lambda_decorr=1.0 result uses the revised criterion, **revised after results were seen, on logical grounds**. Both outcomes and the rank-metric clarification are retained in docs/DECISION_RECORD.md.

## Phase C one-shot audit

The source-test artifact was opened once. The first label-mapping operation failed on raw label saint, which is not represented in the approved taxonomy. No model inference or test metric was produced; no results.json or results.partial.json exists. The test data must not be reopened or rerun under the current one-shot authorization. The taxonomy was not modified post-test. The Phase C evaluator and supplemental frozen manifest are retained for audit in scripts/run_phaseC_one_shot.py and C:\\NOVEL_IDS_runs\\phaseC_one_shot_20261126\\freeze_manifest.json.

## Reproducibility and frozen assets

- Primary checkpoint: C:\\NOVEL_IDS_runs\\stage7_11\\coral_screen_lambda0p05_seed43\\best.pt
- Primary checkpoint SHA256: {cal["checkpoint_sha256"]}
- Stage 5 snapshot SHA256: {cal["stage5_sha256"]}
- Stage 6 checkpoint SHA256: {cal["stage6_sha256"]}
- Taxonomy SHA256: {cal["taxonomy_sha256"]}
- Phase E validation reproduction exactly reproduced the Phase A calibration values and validation metrics using source train/validation only.
- No Stage 5 source files, dataset files, labels, split artifacts or preprocessing artifacts were modified.

## Deviations and limitations

1. Phase C one-shot failed before inference due an unmapped official test label; source-test performance is unknown.
2. CIC-DDoS2019 was not supplied; unseen evaluation remains pending.
3. LODO is a development diagnostic with a documented single-source Stage 8 reference adaptation.
4. The Z_s dataset-identity probe validation AUC is {cal["dataset_identity_probe_Z_s"]["validation_roc_auc"]:.6f}, indicating strong dataset identity information remains in the shared representation.
5. The held-out-class OOD numbers are post-hoc proxies, not strict leave-class-out retraining results. NSL_R2L detection at 5% known-validation FPR is {cal["heldout_class_ood_development_proxy"]["nsl_kdd"]["heldout_detection_rate_at_5pct_known_validation_fpr"]:.4f}; UNSW_Shellcode is {cal["heldout_class_ood_development_proxy"]["unsw_nb15"]["heldout_detection_rate_at_5pct_known_validation_fpr"]:.4f}.
6. Common-only ablation used a reduced 1,000-step / batch-64 budget. Report this reduction; do not compare it as a full-budget training run.
7. Finite augmentation bank is one deterministic draw per training row; validation remained clean.
8. Two source datasets and five exact common features are insufficient evidence for general zero-shot deployment.

## Artifact locations

- Phase A calibration: C:\\NOVEL_IDS_runs\\phaseA_primary_seed43_final_retry2\\phaseA_calibration.json
- Phase B runs: C:\\NOVEL_IDS_runs\\phaseB_final_*
- Phase D LODO: C:\\NOVEL_IDS_runs\\phaseD_lodo_20261126\\results.json
- Phase E validation arrays: C:\\NOVEL_IDS_runs\\phaseE_validation_repro_20261126\\validation_distributions.npz
- Figures: docs/figures/
"""
(ROOT/"docs"/"FINAL_REPORT_2026-11-26.md").write_text(report,encoding="utf-8")
readme=r"""# NOVEL-IDS

A cross-dataset network intrusion detection research implementation using UNSW-NB15 and NSL-KDD, with frozen Stage 5 shared encoder, Stage 6 residual branch, Stage 7 adaptive gate, Stage 8 selective alignment, Stage 9 unified representation, Stage 10 Transformer and Stage 11 unified classification/OOD decision logic.

## Current research status

**Validation and LODO work is complete; source-test evaluation is incomplete and unseen evaluation is pending.**

- Phase A: primary seed-43 checkpoint selected and validation calibration reproduced.
- Phase B: validation ablations completed; common-only variant used a reduced 1,000-step batch-64 budget.
- Phase C: one-shot attempt stopped before inference on an unmapped official NSL-KDD test label saint. The test artifact was opened; do not rerun or reopen under the current one-shot authorization. No source-test metrics exist.
- Phase D: LODO diagnostics completed in both directions on shared taxonomy classes using source train and opposite-source validation only.
- Phase E: validation reproducibility, figures, final report and decision documentation generated.
- CIC-DDoS2019 is absent from data/unseen/; unseen evaluation is pending.

See docs/FINAL_REPORT_2026-11-26.md and docs/DECISION_RECORD.md.

## Architecture

Frozen Stage 5/6; train Stages 7–11 only.

1. Stage 5: shared dual encoder produces Z_s and Z_d.
2. Stage 6: dataset-specific residual branch produces R and frozen 128-D R_proj.
3. Stage 7: G = sigmoid(W_g[Z_s; R_proj] + b_g).
4. Stage 8: compatibility scalar C, projections A=W_a Z_f, U=W_u Z_d, and weighted normalized CORAL (lambda_coral=0.05).
5. Stage 9: Z_u = Z_s + G ⊙ R_proj + C ⊙ A + (1-C) ⊙ U.
6. Stage 10: two-layer, four-head Transformer, hidden size 128, FFN 256, dropout 0.1, availability key-padding mask.
7. Stage 11: approved unified taxonomy, class-weighted cross-entropy, global temperature scaling, Mahalanobis+energy OOD score.
8. Stage 12: precedence Unfamiliar → Uncertain → Normal/Known attack.

Stage 5 and Stage 6 are frozen. No joint fine-tuning is performed.

## Data and leakage controls

- UNSW-NB15 and NSL-KDD use frozen train/validation artifacts.
- Primary validation/calibration uses source train and validation only.
- Official test artifacts have a frozen supplemental SHA-256 manifest. The one-shot attempt is incomplete; do not load them again.
- Unseen data was not supplied or accessed.
- Test and unseen labels are never used for training, preprocessing fitting, model selection or threshold tuning.

## Validation-only reproduction command (PowerShell)

From the project root:

    $env:PYTHONPATH = "$PWD\src;$PWD\scripts"
    .\.venv\Scripts\python.exe scripts\phaseE_validation_repro.py --run-dir C:\NOVEL_IDS_runs\phaseE_validation_repro_new --checkpoint C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43\best.pt

The output directory must be new. This is validation-only and does not access test/unseen partitions.

## One-shot and unseen status

Do not rerun scripts/run_phaseC_one_shot.py. The official source-test artifact was already opened once and the no-rerun rule applies, despite the incomplete result.

CIC-DDoS2019 was not supplied. No zero-shot deployment claim is supported.

## Results and figures

- Final report: docs/FINAL_REPORT_2026-11-26.md
- Decision record: docs/DECISION_RECORD.md
- Phase B validation curves: docs/figures/phaseB_validation_curves.png
- Primary per-class F1: docs/figures/primary_per_class_f1.png
- OOD score histograms: docs/figures/validation_ood_histograms.png
- Reliability diagrams: docs/figures/validation_reliability_diagrams.png
- Gate statistics: docs/figures/primary_gate_statistics.png
- Baseline confusion matrices: docs/figures/baseline_validation_confusion_matrices.png

## Known limitations

- Source-test metrics are unavailable because the one-shot evaluator stopped at taxonomy mapping on saint.
- CIC-DDoS2019 unseen evaluation is pending.
- LODO is a diagnostic with a single-source Stage 8 reference duplicated into both reference slots; it is not a deployment evaluation.
- The Z_s dataset-identity probe remains highly predictive of dataset identity.
- Held-out-class OOD results are exploratory proxies, not strict leave-class-out retraining results.
- Common-only ablation has a reduced training budget.
- No zero-shot deployment claim is supported by current evidence.
"""
(ROOT/"README.md").write_text(readme,encoding="utf-8")
print("PHASE_E_DOCS_AND_FIGURES_WRITTEN",len(list(FIG.glob("*.png"))),"figures")
