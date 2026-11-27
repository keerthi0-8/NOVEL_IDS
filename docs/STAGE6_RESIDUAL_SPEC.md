
# NOVEL-IDS — Stage 6: Residual Information Extraction Specification

**Status:** Draft — submitted for user review  
**Last updated:** 2026-11-23  
**Implementation status:** Not started  
**Training status:** Not started

---

## 1. Purpose

Stage 6 extracts useful dataset-specific information that is not adequately represented by the shared representation produced by Stage 5.

The research objective is to preserve attack-relevant information in dataset-specific features while learning representations that can transfer across datasets.

Stage 6 must avoid simply duplicating information already present in the shared representation. It should retain useful residual information while controlling residual magnitude and reducing redundancy with the shared representation.

Stage 6 initially trains with Stage 5 frozen.

## 2. Position in the Architecture

Stage 5 produces:

- `Z_s`: shared representation.
- `Z_d`: dataset-specific representation.

Stage 6 receives these representations and produces:

- `R`: residual representation.

Conceptual flow:

```text
Stage 5
   |
   +---- Shared representation (Z_s)
   |
   +---- Dataset-specific representation (Z_d)
                 |
                 v
        Residual Predictor (g)
                 |
                 v
        Predicted dataset-specific
        representation (Z_d_hat)
                 |
                 v
        Prediction error
        (Z_d - Z_d_hat)
                 |
                 v
        Residual Encoder (h)
                 |
                 v
        Residual representation (R)
                 |
                 v
        Residual reconstruction branch
```

Stage 6 does not replace Stage 5 or change the fixed overall architecture.

## 3. Approved Decision 7 — Residual Extraction

The approved residual extraction approach is prediction-error-based residual learning.

The predictor estimates the dataset-specific representation from the shared representation:

\[
\hat Z_d = g(Z_s)
\]

The prediction error is:

\[
E_d = Z_d - \hat Z_d
\]

The residual encoder transforms the prediction error:

\[
R_d = h(E_d)
\]

Where:

- `g`: residual predictor.
- `h`: residual encoder.
- `Z_s`: shared representation.
- `Z_d`: dataset-specific representation.
- `Z_d_hat`: predicted dataset-specific representation.
- `E_d`: prediction error.
- `R_d`: extracted residual representation.

The purpose is to identify dataset-specific information that cannot be adequately predicted from the shared representation.

The precise gradient handling between the predictor and residual branch remains subject to the gradient-policy section and unresolved decisions below.

## 4. Approved Decision 7b — Three-Part Residual Objective

Stage 6 must consider three objectives:

1. Magnitude stability.
2. Shared/residual redundancy reduction.
3. Reconstruction-based information preservation.

All three objectives must be considered together.

Their coefficients must be configurable and selected using validation data only.

No numerical coefficient or threshold is approved by this decision.

### 4.1 Magnitude stability

Purpose:

- Prevent residual representations from becoming excessively large.
- Avoid trivial residual collapse.
- Maintain useful residual variation.

A possible magnitude objective is:

\[
L_{upper} =
\frac{1}{B}
\sum_{i=1}^{B}
\max(0,\|R_i\|_2-\tau_{upper})^2
\]

A possible lower-variation objective is:

\[
L_{lower} =
\frac{1}{D_r}
\sum_{j=1}^{D_r}
\max(0,\tau_{lower}-Std(R_{:,j}))^2
\]

Combined proposal:

\[
L_{mag}=L_{upper}+\alpha L_{lower}
\]

Where:

- `B`: batch size.
- `D_r`: residual dimension.
- `tau_upper`: upper magnitude threshold.
- `tau_lower`: minimum standard-deviation threshold.
- `alpha`: configurable coefficient.

**Status:** Proposed formulation; thresholds, coefficient and final formulation require approval.

Thresholds must be calibrated using training/validation information without using test or unseen labels.

## 5. Approved Decision 8 — Redundancy Reduction

The approved method is Option A: standardized cross-covariance penalty.

The objective is to reduce linear dependence between the shared representation and residual representation.

### 5.1 Standardization

For each dataset separately:

\[
\widetilde Z_s =
Standardize(Z_s)
\]

\[
\widetilde R =
Standardize(R)
\]

Standardization and covariance statistics must not pool the UNSW-NB15 and NSL-KDD sub-batches.

### 5.2 Cross-covariance penalty

\[
L_{red} =
\frac{1}{d_s d_r}
\left\|
\frac{1}{B-1}
\widetilde Z_s^T\widetilde R
\right\|_F^2
\]

Where:

- `d_s`: shared representation dimension.
- `d_r`: residual representation dimension.
- `B`: dataset-specific sub-batch size.
- `||.||_F`: Frobenius norm.

### 5.3 Mandatory safeguards

The following safeguards are explicitly approved.

**1. Stop-gradient through the shared representation**

Stop gradients through `Z_s` inside the redundancy loss.

The redundancy penalty must shape the residual representation rather than update the shared encoder.

**2. Per-dataset statistics**

Calculate centering, standardization and cross-covariance separately for each dataset.

Do not pool UNSW-NB15 and NSL-KDD sub-batches.

**3. Variance floor**

Mask dimensions whose batch standard deviation is below a configurable threshold.

Log the number of masked dimensions for every training step.

**4. Adequate batch size**

Use a sufficiently large dataset-specific sub-batch for a stable covariance estimate.

Report the actual sub-batch size.

If memory limitations require an exponential moving average (EMA) of statistics, document and validate that implementation.

**5. Warm-up**

Ramp the redundancy coefficient from zero to its configured target over a configurable number of training steps.

**6. Preserve information**

Redundancy reduction is a soft penalty, not a hard decorrelation constraint.

The reconstruction-based information-preservation objective must remain active.

Do not force residual representations to become zero or constant.

**7. Mixed precision**

Compute the covariance penalty in float32, even when the remaining model uses mixed precision.

### 5.4 Limitation and nonlinear diagnostic

Cross-covariance measures linear dependence. It does not establish statistical independence or eliminate nonlinear dependence.

A separate diagnostic is therefore required:

- Train a small MLP to predict `R` from frozen `Z_s`.
- Evaluate its held-out validation performance.
- Report held-out `R²`.
- Treat this as a diagnostic, not an additional training loss.

The exact MLP architecture and diagnostic protocol remain to be specified.


## 6. Approved Residual Reconstruction Target

The residual decoder reconstructs the prediction error rather than the complete dataset-specific representation.

The prediction error is:

\[
E_d=Z_d-\operatorname{StopGrad}(\hat Z_d)
\]

The residual representation is:

\[
R=h(E_d)
\]

The residual decoder produces:

\[
\hat E_d=dec_R(R)
\]

The reconstruction objective must measure how accurately the residual representation preserves the prediction error:

\[
L_{info}=L_{recon}(\hat E_d,\operatorname{StopGrad}(E_d))
\]

This is the approved reconstruction target and conceptual objective.

### 6.1 Intended behavior

- The residual decoder reconstructs the prediction error.
- It does not directly reconstruct the complete `Z_d`.
- The prediction error target is detached.
- Residual reconstruction losses must not update predictor `g`.
- The residual encoder and residual decoder learn to preserve information contained in the prediction error.

### 6.2 Details still requiring approval

The following remain unresolved:

- Exact reconstruction loss (for example, MSE).
- Whether a zero-residual or mean-residual baseline is required.
- Whether a reconstruction-improvement margin should be used.
- Any margin value or coefficient.
- Exact gradient behavior through the residual encoder and decoder.
- Final coefficient selection procedure.

These details must be reviewed before implementation.

No numerical values are approved by this decision.


## 7. Proposed Combined Stage 6 Objective

A possible combined objective is:

\[
L_6 =
\lambda_{pred}L_{pred}
+
\lambda_{mag}L_{mag}
+
\lambda_{red}L_{red}
+
\lambda_{info}L_{info}
\]

Prediction loss:

\[
L_{pred} =
L_{prediction}(
g(StopGrad(Z_s)),
StopGrad(Z_d)
)
\]

The prediction loss uses detached shared and dataset-specific representations.

The coefficients must be configurable.

**Important:** The combined equation is a proposed specification, not a fully approved final objective. The exact prediction loss, magnitude formulation, reconstruction objective, coefficient values and coefficient-selection procedure remain subject to review.

Coefficient selection must use validation data only.

No test or unseen data may be used to select objective coefficients.

## 8. Approved Decision 9 — Freeze Stage 5 During Initial Stage 6 Training

Stage 5 must remain frozen during the initial Stage 6 training phase.

### 8.1 Frozen components

The following Stage 5 components must be frozen:

- Stage 5 shared encoder.
- Dataset-specific encoders and embeddings.
- Stage 5B shared context adapter.
- Stage 5 native classification heads.
- Stage 5 reconstruction decoder.

### 8.2 Trainable components

Only the following Stage 6 components are trainable:

- Residual predictor `g`.
- Residual encoder `h`.
- Stage 6 residual reconstruction branch.

### 8.3 Frozen-module execution policy

Frozen Stage 5 components must:

- Run in evaluation mode.
- Run under `no_grad`.
- Have `requires_grad=False`.
- Receive no optimizer updates.

The optimizer must receive only Stage 6 parameters.

Feature-set dropout and value masking must remain active on the input.

`Z_s` and `Z_d` must be computed on the fly using the frozen Stage 5 modules.

### 8.4 Prediction and redundancy gradient policy

The prediction loss must detach both `Z_s` and `Z_d`.

The redundancy loss must stop gradients through `Z_s`.

All other gradient paths must be explicitly documented and tested before implementation.

The predicted representation Z_d_hat must be detached when constructing the residual error. Residual-related losses must not propagate gradients through Z_d_hat into predictor g. Predictor g is trained through its separate prediction loss.

## 9. Required Synthetic-Only Tests

The following tests are mandatory before Stage 6 training is accepted.

### 9.1 Stage 5 immutability

- Record hashes of Stage 5 parameters before a Stage 6 optimization step.
- Perform a synthetic Stage 6 optimization step.
- Recalculate Stage 5 parameter hashes.
- Verify that all Stage 5 hashes remain identical.

### 9.2 Gradient validation

Verify that:

- Predictor `g` receives non-zero gradients.
- Residual encoder `h` receives non-zero gradients.
- Residual decoder receives non-zero gradients.
- All frozen Stage 5 parameters receive zero gradients.

### 9.3 Optimizer validation

Verify that the optimizer contains only Stage 6 parameters.

### 9.4 Checkpoint provenance

Verify that the Stage 6 checkpoint records the checksum of the Stage 5 checkpoint used to generate its representations.

### 9.5 Redundancy safeguards

Synthetic tests must cover all seven safeguards specified in Section 5.3:

- Stop-gradient through `Z_s`.
- Per-dataset statistics.
- Variance-floor masking and logging.
- Batch-size adequacy or documented EMA behavior.
- Redundancy coefficient warm-up.
- Active information-preservation behavior.
- Float32 covariance computation.

These tests establish implementation correctness only. They do not constitute research performance evidence.

## 10. Residual Information Evaluation

Residual usefulness must be evaluated rather than assumed.

Use validation-only probes to compare:

**Baseline:**

\[
Z_s
\]

**Augmented representation:**

\[
[Z_s,R]
\]

Report macro-F1 separately for:

- UNSW-NB15.
- NSL-KDD.

Also report the approved residual-quality metrics, including:

- Residual magnitude statistics.
- Residual variance.
- Shared/residual cross-covariance.
- Number of masked low-variance dimensions.
- Held-out nonlinear MLP prediction `R²`.

The comparison must use the same validation partitions and evaluation protocol.

Report if the residual representation provides no measurable improvement.

Do not claim that residual learning improves detection unless the evaluation supports that conclusion.

No test data may be used for this validation-based design selection.

## 11. Unseen Dataset Inference

The unseen dataset path must reuse the same frozen Stage 5B and Stage 6 inference path.

The approved unseen scaling policy is Section 5.4.6 Option A:

**Zero-shot inference with unsupervised per-dataset normalization.**

For unseen input:

- Use unlabeled feature values only for the approved per-dataset normalization.
- Do not use unseen labels.
- Do not perform gradient updates.
- Do not fit model parameters on unseen data.
- Keep Stage 5 and Stage 6 frozen.
- Apply the same feature identity, masking and representation-generation policies.

No unseen dataset evaluation is authorized by this specification.

## 12. Joint Fine-Tuning Policy

Joint fine-tuning of Stage 5 and Stage 6 is NOT approved for the initial training phase.

It may be considered only as a separate controlled experiment after measurable residual improvement has been demonstrated during frozen Stage 5 training.

Any future experiment must include:

- A lower Stage 5 learning rate.
- A fixed training budget.
- A configured maximum permitted Stage 5 validation macro-F1 degradation.
- A same-seed comparison against the frozen Stage 5 result.

The experiment requires separate user approval before implementation.

## 13. Dataset Handling

The initial Stage 6 source datasets are:

- UNSW-NB15.
- NSL-KDD.

Use the active, approved training and validation artifacts.

Dataset-specific statistics must remain separate where required by the redundancy objective.

Do not:

- Modify original datasets.
- Regenerate approved splits without authorization.
- Use test labels for training or model selection.
- Use unseen labels.
- Fit model parameters on test or unseen data.

The existing overlap-corrected validation artifacts and evaluation-index policies remain authoritative.

## 14. Risks and Limitations

Potential risks include:

1. Residual collapse or excessively large residual magnitudes.
2. Residual duplication of shared information.
3. Unstable covariance estimates from small sub-batches.
4. Dataset-specific residual representations that do not transfer.
5. Reconstruction objectives that reward irrelevant information.
6. Residual performance gains that do not generalize.
7. Gradient leakage into frozen Stage 5 components.
8. Excessive complexity without measurable improvement.

These risks must be addressed through the specified safeguards, synthetic tests, validation metrics and ablation.

## 15. Decisions Still Requiring Review

The following items are not approved merely because equations or options appear in this draft:

- Final magnitude-stability formulation and thresholds.
- Exact residual reconstruction loss, baseline, margin formulation and coefficients.
- Whether the Stage 5 decoder participates in residual reconstruction.
- Gradient flow from residual reconstruction into the predictor.

- Final Stage 6 loss coefficients and their selection procedure.
- Exact predictor and residual encoder architectures.
- Residual dimension.
- Nonlinear diagnostic MLP configuration.
- Stage 6 checkpoint and scheduler details.
Approved gradient policy: Z_d_hat is detached when constructing the residual error.
Other unresolved `[GO]` decisions for Stages 7–11 and the overall architecture remain governed by the master specification.

## 16. Implementation Gate

Stage 6 implementation must not begin until all of the following conditions are met:

1. The user reviews and approves this Stage 6 specification.
2. The Section 6.0 gate is accepted.
3. Stage 5 implementation, training and exit criteria are completed and accepted.
4. The Stage 5 checkpoint required by Stage 6 is available and verified.
5. Remaining Stage 6 design choices are explicitly resolved.

Until then:

- Do not implement Stage 6 modules.
- Do not start Stage 6 training.
- Do not use test or unseen data.
- Do not resolve unrelated `[GO]` decisions.
- Do not modify the authoritative master specification without approval.

**Current disposition: Stage 6 design review only.**
