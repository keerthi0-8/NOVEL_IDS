# NOVEL-IDS Stage 6 - Final Specification for User Review

**Status:** Prepared for review; NOT approved for implementation  
**Prepared:** 2026-11-24  
**Scope:** Stage 6 residual information extraction only  
**Implementation:** Not started; no Stage 6 source code or tests were created by this document task  
**Training:** Not started

## 1. Approval boundary

This document consolidates the already-approved Stage 6 decision families and the defaults recommended for review. The user's current authorization is to prepare this specification only. It is not authorization to implement Stage 6, run Stage 6 training, change Stage 5, or approve Stages 7-12.

Implementation remains blocked until the user explicitly approves this complete specification and the Stage 5 exit/checkpoint gates are satisfied. Any item marked **REVIEW DEFAULT** is a proposal for approval, not an existing approved decision.

## 2. Purpose and fixed architecture position

Stage 5 supplies a shared representation `Z_s` and a dataset-specific representation `Z_d`. Stage 6 extracts information in `Z_d` that is not predictable from `Z_s`, and returns residual representation `R` to the already-defined Stage 7 input. Stage 6 does not reorder, replace, or bypass any of the twelve stages.

For each source dataset `k` (UNSW-NB15 and NSL-KDD), with batch size `B_k`:

- `Z_s,k`: `[B_k, 128]`
- `Z_d,k`: `[B_k, 128]`
- `Zhat_d,k`: `[B_k, 128]`
- `E_d,k`: `[B_k, 128]`
- `R_k`: `[B_k, 128]`

**REVIEW DEFAULT:** `d_s = d_d = d_r = 128`, matching the current Stage 5 output contract. No projection or extra adapter is introduced.

## 3. Stage 6 modules

All three networks below are small MLPs. They are trained only during Stage 6; they do not modify Stage 5.

### 3.1 Residual predictor `g`

`g: R^128 -> R^128`

Architecture:

`Linear(128,128) -> GELU -> LayerNorm(128) -> Linear(128,128)`

Prediction:

`Zhat_d = g(StopGrad(Z_s))`

Prediction objective, computed separately for each dataset:

`L_pred,k = mean((Zhat_d - StopGrad(Z_d))^2)`

This objective updates `g` only. It is not a classification objective and consumes no labels.

### 3.2 Residual error and encoder `h`

Detach the predictor output before constructing the error:

`E_d = StopGrad(Z_d) - StopGrad(Zhat_d)`

`R = h(E_d)`

Architecture:

`Linear(128,128) -> GELU -> LayerNorm(128) -> Linear(128,128)`

No output normalization is applied, because residual magnitude must remain observable. The residual dimension is 128.

### 3.3 Residual decoder `dec_R`

`dec_R: R^128 -> R^128`

Architecture:

`Linear(128,128) -> GELU -> LayerNorm(128) -> Linear(128,128)`

`E_d_hat = dec_R(R)`

`dec_R` is a Stage 6 training/diagnostic branch. It is not required for the forward inference output `R` and is not the Stage 5 masked-feature decoder.

## 4. Frozen Stage 5 policy and exact gradient boundaries

The initial Stage 6 training policy is frozen Stage 5. The following remain in evaluation mode, `requires_grad=False`, and under `torch.no_grad()` when producing `Z_s` and `Z_d`:

- Stage 5 shared encoder and its learned feature/identity parameters;
- dataset-specific encoder/embedding path and Stage 5B shared context adapter;
- Stage 5 native classification heads;
- Stage 5 masked-feature reconstruction decoder.

The optimizer receives only parameters from `g`, `h`, and `dec_R`. Stage 5 parameters receive no gradients and no optimizer updates. Stage 5 input masking/feature-set augmentation is used only according to the approved Stage 5 training data path; evaluation/inference does not use stochastic training augmentation.

Gradient map:

| Objective | Parameters updated | Explicitly blocked |
|---|---|---|
| `L_pred` | `g` only | `Z_s`, `Z_d`, `h`, `dec_R`, all Stage 5 parameters |
| `L_mag` | `h` | `g`, `dec_R`, all Stage 5 parameters |
| `L_red` | `h` through `R` | `Z_s`, `g`, `dec_R`, all Stage 5 parameters |
| `L_info` | `h` and `dec_R` | target `E_d`, `g`, all Stage 5 parameters |

For `L_info`, the prediction branch passes through `dec_R` and `h`; the target branch is `StopGrad(E_d)`. `Zhat_d` is detached before `E_d` is formed. Because Stage 5 outputs are detached/frozen, no Stage 6 objective can update Stage 5.

## 5. Approved reconstruction-based information objective

The approved primary objective is:

`L_info,k = MSE(E_d_hat, StopGrad(E_d))`

where:

- `E_d = StopGrad(Z_d) - StopGrad(Zhat_d)`
- `R = h(E_d)`
- `E_d_hat = dec_R(R)`

This is residual-error reconstruction, not reconstruction of the full `Z_d`, and not Stage 5 masked-feature reconstruction. No Stage 5 decoder is used in this loss.

**No optional masked-feature reconstruction ablation is included.** It is excluded from this specification and must not be implemented by default.

## 6. Magnitude stability objective

Use the review defaults from the master specification. For each dataset independently:

`rms(X) = sqrt(mean(X^2) + epsilon)`

`rho = rms(R) / (rms(E_d) + epsilon)`

`s_j = std_batch(R[:,j]) / (rms(E_d) + epsilon)`

`L_mag = relu(0.1 - rho)^2 + relu(rho - 1.0)^2 + mean_j(relu(0.01 - s_j)^2)`

**REVIEW DEFAULTS:** `epsilon = 1e-6`; target residual-to-error RMS ratio `rho` in `[0.1, 1.0]`; normalized per-dimension standard-deviation floor `0.01`.

This penalizes global collapse, excessive residual energy, and dead dimensions. It does not force residual dimensions to have a particular direction. These thresholds are validation-reviewable and are not empirical optima.

## 7. Approved redundancy family: standardized cross-covariance

For each dataset independently, compute statistics in float32. Never pool UNSW-NB15 and NSL-KDD before centering, standardization, or covariance estimation.

1. Detach `Z_s` for this loss.
2. Center `Z_s` and `R` by their own batch means.
3. Standardize each dimension by its own batch standard deviation.
4. Retain only finite dimensions with standard deviation at or above the configured floor.
5. Let `I_s` and `I_r` be the active dimensions.

`C_k = (Zs_tilde[:, I_s].T @ R_tilde[:, I_r]) / (B_k - 1)`

`L_red,k = sum(C_k^2) / (|I_s| * |I_r|)`

`L_red` is the mean over datasets with valid active dimensions. If a dataset has too few rows or no active dimensions, skip that dataset's term and log the reason; never divide by zero. If neither dataset has a valid term, fail the training step rather than silently treating the batch as a successful update.

**REVIEW DEFAULTS / mandatory safeguards:**

- Numerical epsilon: `1e-6`.
- Standard-deviation floor: `1e-4` in unstandardized representation units.
- Per-dataset covariance sub-batch: `B_k = 256`.
- No EMA in the initial implementation; require the full configured per-dataset batch. Reject an undersized training batch with a clear error. This avoids introducing an unvalidated moving-statistics implementation.
- Compute centering, standardization, and covariance in float32 even if other operations later use mixed precision.
- Log active and masked dimensions per dataset and step.
- `lambda_red` warms linearly from zero to its selected target over the first 10% of planned optimizer steps.
- `L_info` remains active during warm-up; decorrelation is a soft penalty, never a hard constraint.

A nonlinear diagnostic is also required after fitting: freeze `Z_s` and `R`, fit a small MLP to predict `R` from `Z_s` using a deterministic 80/20 split of each source validation partition, and report held-out `R^2` separately per dataset. This is a diagnostic only, not a training loss. It must not use test or unseen data.

## 8. Combined Stage 6 objective and coefficient selection

For each dataset `k`, compute each term separately:

`L_6,k = lambda_pred * L_pred,k + lambda_mag * L_mag,k + lambda_red(t) * L_red,k + lambda_info * L_info,k`

Average valid dataset-level terms only after each dataset's term has been calculated. Give UNSW-NB15 and NSL-KDD equal dataset-level weight; do not let the larger dataset dominate.

**REVIEW DEFAULTS:**

- `lambda_pred = 1.0`
- `lambda_mag = 1.0`
- `lambda_info = 1.0`
- `lambda_red,target` candidates: `{0.01, 0.1, 1.0}`
- `lambda_red(t)` is the linear 10% warm-up described above.

Select `lambda_red,target` using source validation only. Use identical Stage 5 checkpoint, data partitions, seeds, training budget, and evaluation protocol for each candidate. Primary selection evidence is the validation probe comparison of `Z_s` versus `[Z_s, R]`, reported separately for each dataset, subject to finite/stable losses and residual-quality safeguards. Do not select coefficients using official test data or unseen data. Record every candidate and all per-dataset metrics; do not suppress a candidate that performs worse.

**Training-schedule defaults still proposed for approval:** AdamW, learning rate `1e-3`, weight decay `1e-4`, gradient clipping `1.0`, maximum `10,000` optimizer steps, validation every `500` steps, and early stopping after `5` validation checks without improvement in the mean of the two per-dataset validation macro-F1 probe scores. Use seed `42` for the initial reproducibility run. If these budget/schedule defaults are not approved, Stage 6 implementation must wait for replacement values.

## 9. Validation probe and reporting protocol

Stage 6's unsupervised training losses do not consume labels. To measure whether `R` helps intrusion detection, train two controlled probes using training-split labels only:

- Baseline probe input: frozen `Z_s`.
- Augmented probe input: frozen `[Z_s, R]`.

Use the same probe family, optimizer, seed, class-weight policy (weights from training labels only), training budget, and validation partition for both. Evaluate only on source validation labels. Report macro-F1, per-class support, and per-class results separately for UNSW-NB15 and NSL-KDD. A gain is not assumed; explicitly report no improvement or degradation.

Also report, per dataset:

- `L_pred`, `L_mag`, `L_red`, `L_info`, and total loss;
- `R` and `E_d` RMS/norm distributions;
- per-dimension residual standard deviations and effective rank;
- active/masked covariance dimension counts;
- mean absolute active cross-correlation;
- residual norms by class (evaluation/reporting only);
- nonlinear validation MLP `R^2` diagnostic;
- probe macro-F1, class support, and delta versus baseline.

Synthetic tests are correctness checks only. They must never be reported as research performance results.

## 10. Inference and unseen-data behavior

For an already prepared source or unseen example, Stage 5 produces `Z_s` and `Z_d` using its approved path. Apply frozen `g` and `h`:

`Zhat_d = g(Z_s)`

`E_d = Z_d - Zhat_d`

`R = h(E_d)`

`dec_R` is not needed for inference. No labels, gradient steps, optimizer updates, test outcomes, or unseen-label tuning are permitted. For unseen datasets, preserve the approved zero-shot protocol with unsupervised per-dataset normalization only; Stage 6 adds no additional fitting on unseen data.

## 11. Checkpoint and provenance requirements

Do not use a Stage 5 checkpoint in place. Before Stage 6 implementation/training, copy the accepted Stage 5 checkpoint to a separate Stage 6 input snapshot directory, verify that the copy's SHA-256 equals the source checkpoint's SHA-256, and load only the copy. Do not modify or write into the Stage 5 checkpoint directory.

A Stage 6 checkpoint must include:

- `g`, `h`, and `dec_R` state dictionaries;
- Stage 6 config and selected coefficients;
- optimizer/scheduler state and step/epoch;
- seed and training/validation provenance;
- per-dataset loss and metric history;
- Stage 5 checkpoint SHA-256 and source/copy paths;
- Stage 5 code/config hash manifest;
- software/runtime versions and device metadata.

Store Stage 6 artifacts under a separate Stage 6 run directory. Do not overwrite Stage 5 artifacts.

## 12. Synthetic-only test plan (implementation only after approval)

All tests below must use synthetic tensors, CPU only, and no research data. Cap numerical-library/PyTorch threads at 2. Tests are not to be created or run as part of this specification-preparation task.

1. Shape contract: `Z_s`, `Z_d`, `Zhat_d`, `E_d`, and `R` have the documented 128-dimensional shapes for both datasets.
2. Finite forward/loss: all outputs and four losses remain finite for normal, constant, and near-constant synthetic inputs.
3. Frozen Stage 5: no Stage 5 parameter gradients; parameter hashes are unchanged after a synthetic Stage 6 optimization step.
4. Optimizer isolation: optimizer parameter IDs equal exactly the Stage 6 parameters from `g`, `h`, and `dec_R`.
5. Predictor gradient: `L_pred` produces nonzero gradients for `g` and none for `h`, `dec_R`, or Stage 5.
6. Residual gradient: `L_mag` and `L_red` update `h`; neither updates `g`, `dec_R`, or Stage 5.
7. Reconstruction gradient: `L_info` updates `h` and `dec_R`; no gradient reaches the detached target, `g`, or Stage 5.
8. Residual detach: changing predictor parameters cannot receive gradients through `E_d`/`R`; predictor is optimized only by `L_pred`.
9. Cross-covariance signal: independent random `Z_s`/`R` has lower expected penalty than a deliberately linearly correlated fixture, within documented finite-sample tolerance.
10. Scale invariance: standardized cross-covariance is invariant to positive/negative nonzero rescaling away from the variance floor.
11. Variance-floor safety: constant/low-variance dimensions are masked and counted; loss remains finite; no divide-by-zero or NaN.
12. Per-dataset isolation: a synthetic dataset-shift fixture proves that statistics are computed separately and not pooled.
13. Batch policy: batch size 256 is accepted; undersized covariance batches fail clearly; no EMA path exists in the initial design.
14. Warm-up: `lambda_red` starts at zero, increases linearly, and reaches its target at 10% of the configured training budget; `L_info` remains active at step zero.
15. Magnitude collapse: a collapsed `R` triggers the lower magnitude penalty and nonzero gradient to `h` when the fixture permits it.
16. Float32 covariance: covariance and standardization remain float32 and finite under an autocast test context.
17. Checkpoint provenance: a synthetic checkpoint records the exact Stage 5 checkpoint hash and config.
18. Inference contract: inference returns finite `R` with no labels, gradient tracking, optimizer step, or parameter mutation; `dec_R` is not required.

## 13. Explicit exclusions

- No Stage 5 code/config/data/checkpoint/log changes.
- No Stage 6 implementation or training before final approval and Stage 5 exit.
- No optional Stage 6 masked-feature reconstruction ablation.
- No joint fine-tuning of Stage 5 and Stage 6.
- No test or unseen data for coefficient selection.
- No implementation of Stages 7-12 under this approval.
- No GPU use during this preparation task; later training hardware must be separately planned and memory-profiled.

## 14. Stage 5 integrity snapshot recorded before this document

SHA-256 values captured on 2026-11-24 before creating this Stage 6 review document. These hashes cover the Stage 5 representation/training source files and active top-level dataset/preprocessing YAML configs. They are a baseline, not a git commit or tag.

| File | SHA-256 |
|---|---|
| `src/novel_ids/representation/stage5_training.py` | `88BFF4FEB105F589FC1865D1AA17053043B7D2EC4DD2775F84EB02F87F036F8E` |
| `src/novel_ids/representation/stage5_training_candidate.py` | `0A1C75A046C9CE86BF122D4E4CECB65DF14365E2114760040B30D52787015130` |
| `src/novel_ids/representation/dual_encoder.py` | `1FF74D97E651C541B9AB6F84EC762F054D6CF03E24F054A8BDF9C8D723AC0FFC` |
| `src/novel_ids/representation/classification_head.py` | `D4D0C337AC8C9A234239461282DFD5A4753B7A660C9ADA91D9C404B77900EED8` |
| `src/novel_ids/representation/reconstruction_decoder.py` | `E043AEC1F54367A18A89DA762FE2BFD24A1898AA5FEA07AA2B81AB85F865243A` |
| `src/novel_ids/representation/feature_identity.py` | `D82C46AACCA3C4C53777A8AB7C2F1D0930FE8E44E2D554A912578482091942AC` |
| `config/datasets.yaml` | `ADE3CB9E5E12015542ECEB6BB2ACFA5F440551F72278ADE7C5BF4A955FE836BE` |
| `config/preprocessing_policy.yaml` | `123BDD2104D01C0839E93AAF7B2322072786AF7987AA7213A2247878CBCCD6EF` |

The active Stage 5 training process was still running when this snapshot was recorded. No Stage 5 file, config, run directory, checkpoint, log, or data artifact was modified by preparing this specification.

## 15. User decisions required before implementation

Please approve or reject the following as one complete Stage 6 specification. If rejecting any item, state the replacement value/approach; implementation remains blocked until the specification is internally consistent.

1. **Representation dimensions:** `d_s=d_d=d_r=128`.
2. **Network architectures:** the three specified 128-to-128 MLPs for `g`, `h`, and `dec_R`.
3. **Stage 5 freeze:** all Stage 5 components frozen/eval/no-grad; optimizer contains only Stage 6 parameters.
4. **Gradient policy:** `L_pred` trains only `g`; `L_mag`/`L_red` train `h`; `L_info` trains `h` and `dec_R`; no Stage 5 gradients.
5. **Magnitude loss:** formula and thresholds `rho=[0.1,1.0]`, per-dimension floor `0.01`, epsilon `1e-6`.
6. **Redundancy loss:** standardized per-dataset cross-covariance, squared Frobenius mean, std floor `1e-4`, epsilon `1e-6`, float32, batch 256, no EMA, undersized batch rejected, 10% linear warm-up.
7. **Information loss:** approved `MSE(E_d_hat, StopGrad(E_d))`; no Stage 5 decoder in this objective.
8. **Objective coefficients:** `lambda_pred=lambda_mag=lambda_info=1.0`; validation candidates for `lambda_red,target` are `{0.01,0.1,1.0}`.
9. **Training schedule:** AdamW `lr=1e-3`, `weight_decay=1e-4`, gradient clip `1.0`, max 10,000 steps, validation every 500 steps, early-stop patience 5, seed 42.
10. **Validation/model selection:** paired baseline `[Z_s]` versus augmented `[Z_s,R]` probes, training labels only for probe fitting, source validation only for evaluation/selection, equal dataset weighting, and no test/unseen tuning.
11. **Nonlinear diagnostic:** validation-only MLP prediction of `R` from `Z_s`, deterministic 80/20 split, report held-out `R^2`.
12. **Checkpoint provenance:** copy accepted Stage 5 checkpoint before use, verify SHA-256, load the copy, write Stage 6 artifacts to a separate run directory.
13. **Exclusions:** no optional masked-feature ablation, no Stage 5 joint fine-tuning, no Stages 7-12 implementation under this approval.

**Implementation gate:** No Stage 6 code, tests, or training until the user explicitly approves all required items above and Stage 5 exit/checkpoint acceptance is complete. Stages 7-12 remain specification/test-plan work only and each `[GO]` implementation requires separate approval.
