# NOVEL-IDS — Consolidated Stages 6–12 Specification

**Status:** Consolidated from user-approved decisions through 2026-11-24  
**Scope:** Stages 6–12. This document does not modify the master specification.  
**Implementation boundary:** CPU-only synthetic modules/tests are authorized in the approved milestone order. No training is authorized here. Stage 5 remains protected.

## 1. Authority, scope, and gates

This document consolidates approved decisions and explicitly marks unresolved choices. It does not silently approve open items or supersede NOVEL_IDS_MASTER_SPEC.md. The separately proposed Decision 9 master-spec edit remains unapplied pending explicit confirmation.

Order of work:
1. Complete the Stage 5 checkpoint acceptance/exit gate. A partial Phase B run is not an accepted checkpoint.
2. Implement Stage 6–10 modules and CPU-only synthetic tests, only within approved choices.
3. Prepare the Stage 11/12 specification, then generic code and a taxonomy proposal.
4. Stop at unresolved design decisions. Do not invent values, broadcasting rules, class mappings, thresholds, or architecture components.

Global constraints:
- Never modify Stage 5 source, configuration, data, checkpoints, run directories, or logs.
- No GPU for the currently authorized implementation/tests; use CPU and call torch.set_num_threads(2) before model/test execution.
- No package installation and no research-data training in this implementation sequence.
- Stage 5 is frozen for initial Stage 6 training: eval mode, requires_grad=False, and representations computed under torch.no_grad(). Stage 5 receives no Stage 6 gradients or optimizer updates.
- Any joint fine-tuning of Stage 5 with later stages requires separate explicit approval naming unfrozen components, losses, and checkpoint comparison.
- Preserve native source labels. A harmonized cross-dataset attack taxonomy is not approved; prepare an evidence-based proposal instead.
- Keep per-dataset results separate and report negative results and failures.

## 2. Stage 5 interface and provenance

Stage 5 supplies:
- Z_s: shared representation, shape [B,128].
- Z_d: dataset-specific/context-conditioned representation, shape [B,128].

The current Stage 5B adapter consumes the existing Stage 4-derived pooled token representation and context, not Z_s. Do not change this interface as part of Stages 6–12.

Before any future Stage 6 training:
- Wait for Stage 5 exit/checkpoint acceptance.
- Copy the accepted Stage 5 checkpoint into a separate Stage 6 input snapshot.
- Verify source/copy SHA-256 equality and record the source hash.
- Load the copy only; write Stage 6 artifacts to a separate run directory.
- Record the Stage 5 checkpoint hash and Stage 5 code/config hash manifest in every Stage 6 checkpoint.

## 3. Stage 6 — Residual information extraction

### 3.1 Predictor and residual

For each dataset independently:
- Zhat_d = g(StopGrad(Z_s))
- L_pred = MSE(StopGrad(Z_d), Zhat_d)
- E_d = Z_d - StopGrad(Zhat_d)
- R = h(E_d)

Decisions:
- g is a small MLP predictor.
- h is a bottleneck residual encoder.
- For L_pred, both Z_s and Z_d are detached; this objective updates g only.
- E_d is formed with the predictor output detached. The residual encoder has no gradient path into g.
- Initial residual width is 64. Evaluate widths 64 and 128 using source validation only. Project the selected residual to 128 dimensions before Stage 7/9.
- Do not choose the final residual width without recording the validation comparison.

### 3.2 Residual reconstruction objective

A small training-only decoder dec_R reconstructs the residual error:
- E_d_hat = dec_R(R)
- L_info = MSE(E_d_hat, StopGrad(E_d))

This is residual-error reconstruction, not full Z_d reconstruction and not Stage 5 masked-feature reconstruction. The decoder is removable at inference.

### 3.3 Magnitude and anti-collapse

Include a normalized residual-magnitude penalty with:
- a small numerical floor for stable division;
- a lower bound on per-dimension standard deviation of R to discourage collapsed/dead dimensions;
- per-dataset calculation and logging.

Exact numerical floors, bounds, coefficients, and schedules are not approved as fixed constants here. Keep them configurable; do not silently select empirical values.

### 3.4 Redundancy penalty

Use per-dataset standardized cross-covariance, computed independently (never pool NSL-KDD and UNSW-NB15 statistics):

L_red = (1 / (d_s * d_r)) * || (1 / (B - 1)) * Zs_tilde^T R_tilde ||_F^2

Approved safeguards:
- Stop-gradient through Z_s for this penalty.
- Compute centering, standardization, and covariance in float32.
- Use per-dataset sub-batches of 256. Use EMA only if memory forces it, and report that deviation.
- Use a warm-up coefficient; keep L_info active during warm-up.
- Treat decorrelation as a soft objective, not a hard constraint.
- Do not let one dataset dominate because it has more rows.

### 3.5 Stage 6 gradient and optimizer boundaries

Initial Stage 6 training:
- Stage 5 is frozen/eval/no-grad.
- The optimizer contains only Stage 6 parameters.
- L_pred updates g only.
- L_info updates h and dec_R; its target branch is detached.
- L_mag and L_red update h through R, with the approved detach boundaries.
- No Stage 6 loss updates Stage 5.

Loss coefficients and remaining exact detach details not specified above must remain configurable/open; do not invent approved constants.

### 3.6 Stage 6 validation and reporting

Use source training labels only to fit controlled probes, and source validation only for evaluation:
- Probe A: frozen Z_s.
- Probe B: frozen [Z_s,R].
- Same probe family, seed, optimizer, class-weight policy, training budget, and validation split.
- Report macro-F1 and per-class support/results separately for NSL-KDD and UNSW-NB15.

Also report per dataset:
- L_pred, L_info, L_mag, L_red and total objective;
- residual/error norms and RMS distributions;
- per-dimension standard deviations and effective rank;
- mean absolute cross-correlation with Z_s;
- per-class residual norm (diagnostic only);
- probe scores and delta versus Z_s baseline;
- optional validation-only MLP R-squared for predicting R from frozen Z_s.

Known diagnostic risk (not a failure by itself): both Z_s and Z_d are functions of the same pooled Stage 4 token representation, with Z_d additionally conditioned on context. Therefore g(Z_s) may predict Z_d very well and leave a small residual. Report the measured result; do not force a nonzero residual or treat a small residual as a bug without evidence.

## 4. Stage 7 — Adaptive residual fusion

Use the projected residual R_proj:
- G = sigmoid(W_g [Z_s; R_proj] + b_g)
- Z_f = Z_s + G * R_proj

Approved defaults:
- Gate is per-dimension.
- Initialize gate bias negatively (for example, b_g=-2) so initial residual contribution is conservative.
- Gate regularizer is configurable and off by default.
- Log per-dataset gate mean, standard deviation, and histogram.
- Do not discard R solely because early gate values are small.

## 5. Stage 8 — Compatibility-aware selective alignment

Compute scalar-per-sample compatibility C with shape [B,1], then broadcast it across the 128 latent dimensions in Stage 9. A config switch may expose a [B,128] per-dimension ablation, but it is OFF by default and its use remains blocked pending a separately approved per-dimension formula.

For sample i:
- s_conf_i is the mean onboarding match confidence over AVAILABLE features. Level 1 contributes confidence; Level 2 with relationship type group_only contributes 0.5 * confidence; Level 3 contributes 0. Unavailable features are excluded from the mean. If no feature is available, s_conf_i=0.
- Fit frozen reference means and diagonal variances on source TRAINING representations Z_f only, separately for NSL-KDD and UNSW-NB15. No labels are used.
- d_i is the mean over 128 dimensions of the diagonal Mahalanobis distance from sample Z_f to the OTHER source dataset's training reference for known-source samples. For an unseen sample, use the minimum of its distances to both source references.
- s_dist_i = exp(-d_i / tau), where tau is the median of cross-domain distances over the two source training sets, computed once and stored with the fitted references.
- C_i = clip(s_conf_i^alpha * s_dist_i^(1-alpha), 0.05, 0.95), alpha=0.5.
- Compute C without gradients and detach it before any alignment or fusion loss. Do not use unseen labels, gradients, or batch-adaptive statistics at inference.

Construct:
- A = W_a Z_f
- U = W_u Z_d

CORAL:
- L_coral = ||S_1-S_2||_F^2 / (||S_1||_F^2 + ||S_2||_F^2 + eps).
- S_d is the per-dataset centered weighted covariance of A, using sample weights C_i and float32 arithmetic. Use weighted mean and covariance normalized by sum of weights; constant weights must reproduce the ordinary population covariance.
- Each source sub-batch must contain at least 256 samples. Use EMA only if memory forces it and report that deviation.
- Candidate lambda values are {0, 0.05, 0.2, 1.0}, configurable, with a warm-up ramp. Lambda=0 is the no-alignment baseline. Do not select a coefficient or run coefficient-selection experiments in this implementation task.
- Later, select only on source validation: reject a candidate if either dataset's macro-F1 drops by more than max(5 percentage points, 2 * seed standard deviations) versus lambda=0; among remaining candidates choose the strongest alignment.

Retain dataset-specific information through U, weighted by 1-C in Stage 9.

Switches:
- no_alignment: disable the CORAL objective and aligned contribution.
- common_feature_only: calculate compatibility confidence using only available Level 1 correspondences; this is a compatibility-restricted ablation, not evidence that the latent A contains exclusively common-feature information.
- full_alignment: use unit alignment weights for the CORAL objective and aligned contribution.
- The default mode uses the approved compatibility C.

Report compatibility and alignment statistics separately for each dataset.

## 6. Stage 9 — Unified representation

The approved combination is:

Z_u = Z_s + G * R_proj + C * A + (1-C) * U

- C is [B,1] and broadcasts across the 128 dimensions.
- A = W_a Z_f and U = W_u Z_d, each [B,128].
- Preserve intermediate Z_s, R, R_proj, G, A, U, C, and Z_f for debugging and ablation.
- Preserve sample correspondence and feature-identity correspondence.
- C is stop-gradient. Do not allow downstream losses to update its confidence or distance inputs.
- Keep the optional [B,128] C mode disabled; no per-dimension compatibility formula has been approved.

## 7. Stage 10 — Feature-token conditioning and Transformer IDS

Approved initial architecture:
- Condition Stage 5 feature tokens using the unified representation: T_f = token_f + W_t Z_u.
- Initialize the CLS token from Z_u.
- Two Transformer layers.
- Four attention heads.
- Hidden dimension 128.
- Dropout 0.1.
- GELU and LayerNorm.
- Use M_avail as the attention key-padding mask so unavailable feature tokens do not influence attention/logits.

Required synthetic test:
- Change the value/embedding of an unavailable token while keeping its availability mask at unavailable; logits must remain unchanged within a documented numerical tolerance.

Report parameter count and estimated memory for the 6 GB target environment. Do not change layer count, heads, hidden size, or add unlisted components without approval.

## 8. Stage 11 — Classification, calibration, and unfamiliar-traffic detection

- Multi-class classification uses the configured/native taxonomy.
- Class-weighted cross-entropy; class weights are computed from training labels only.
- Temperature scaling is fitted on source validation only.
- OOD candidates: class-conditional Mahalanobis distance on Z_t and energy score.
- OOD thresholds are validation-only.
- Softmax confidence alone is not an unfamiliar-traffic detector.
- Held-out attack experiments, if used, are separate controlled experiments and do not alter the native-label training taxonomy.

**Not approved:** a harmonized cross-dataset attack taxonomy. Keep original labels and prepare a proposal with per-class evidence, ambiguous mappings, and dataset-specific classes. Do not merge or relabel attack classes without explicit approval.

## 9. Stage 12 — Final decision outputs

Per sample, expose one of:
- Normal.
- Known attack (category and confidence).
- Uncertain.
- Unfamiliar.

Never emit an attack category outside the trained/configured taxonomy. Confidence and OOD thresholds must be calibrated on source validation only. Exact thresholds remain open until validation evidence is available.

## 10. Required synthetic tests before research training

Synthetic CPU-only tests must cover:
1. Shape and finite-value contracts for every stage output.
2. Stage 5 frozen/eval/no-grad and optimizer isolation in Stage 6.
3. Correct gradient boundaries for L_pred, L_info, L_mag, and L_red.
4. Residual width options 64 and 128 and projection to 128.
5. Stable magnitude penalty and non-collapsed per-dimension residual variance.
6. Per-dataset covariance isolation, float32 calculation, batch-size policy, and warm-up behavior.
7. Stage 7 gate bounds, dimensions, initialization, and configurable regularizer default-off.
8. Stage 8 compatibility scalar shape/range, stop-gradient, confidence aggregation and availability handling, train-only frozen references, distance/tau computation, weighted CORAL behavior, switch behavior, and finite losses.
9. Stage 9 128-dimensional output, scalar-C broadcasting, sample correspondence, exact fusion equation, debug tensor preservation, and gradients to W_a/W_u without gradients through C.
10. Stage 10 unavailable-token invariance.
11. Stage 11 class-weight computation uses training labels only; temperature/OOD thresholds use validation only.
12. Stage 12 output categories are restricted to the configured taxonomy.

Synthetic tests establish implementation correctness only. They are not research performance results. No research-data training is included in this implementation authorization.

## 11. Open decisions — do not infer approval

- Exact Stage 6 loss coefficients and any unlisted optimizer/schedule values.
- Final residual width after validation comparison of 64 and 128.
- Exact numerical floors/bounds for the normalized magnitude penalty where not already specified.
- Whether covariance EMA is required by memory (default is full per-dataset sub-batch of 256; report any EMA deviation).
- Optional per-dimension [B,128] compatibility ablation formula; the config switch remains OFF and use is blocked pending separate approval.
- Selection of the Stage 8 CORAL coefficient from {0, 0.05, 0.2, 1.0}; this requires the later source-validation protocol and is not part of synthetic implementation.
- Exact Stage 11/12 confidence and OOD thresholds.
- Harmonized attack taxonomy.
- Any architecture change, new component, or joint fine-tuning.

## 12. Current implementation boundary

Authorized next work: implement the Stage 6–10 modules and CPU synthetic tests in the approved sequence, while leaving unresolved choices configurable or blocked. Then prepare the Stage 11/12 specification, generic code, and taxonomy proposal.

Not authorized by this document:
- Stage 5 modifications of any kind.
- GPU use, package installation, or research-data training.
- Stage 6 training before Stage 5 checkpoint acceptance and a separate training [GO].
- Enabling the optional per-dimension [B,128] compatibility mode without a separately approved formula.
- Selecting a CORAL coefficient or claiming alignment/performance improvement from synthetic tests.
- Harmonizing native attack labels.
- Joint fine-tuning of Stage 5 and later stages.
