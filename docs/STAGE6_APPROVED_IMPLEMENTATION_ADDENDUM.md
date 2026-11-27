# NOVEL-IDS Stage 6 — Approved Implementation Addendum

**Date:** 2026-11-24  
**Purpose:** Record later explicit Stage 6 approvals and implementation status without rewriting the authoritative master specification. This addendum supersedes older Stage 6 draft defaults only where it records a later explicit approval.

## 1. Stage 5 input and protection

- Accepted Stage 5 checkpoint: C:\NOVEL_IDS_runs\stage6\stage5_input_snapshot\best.pt
- SHA256: 9c4655286d6866e245320adc770dac7662d83b04afa2aedb3c9712e406dcf02b
- Checkpoint metadata: seed 42, Phase A, lambda 0.1, global step 8,000.
- The Stage 5 snapshot is loaded read-only. Stage 5 is eval-mode, requires_grad=False, and executed under torch.no_grad().
- The eight protected Stage 5 source/config files were checked against their accepted hashes: **8/8 match**. The repaired stage5_training.py hash is EE3D8FE2DA5C76F9FBAC1D75E48EAA029E6A13AF68F96CF3BC81F13D22F98267.
- Stage 6 checkpoints record the Stage 5 snapshot hash, protected-file hash manifest, source-artifact paths and hashes, software versions, and device metadata.
- Stage 6 training input contract permits only source train and validation numeric/categorical values and masks. It rejects label fields. Test and unseen splits are not used by the training path.

## 2. Approved Stage 6 representation and losses

For each source dataset independently:

- Z_s, Z_d: frozen Stage 5 representations, each 128-D.
- Zhat_d = g(StopGrad(Z_s)).
- E_d = StopGrad(Z_d) - StopGrad(Zhat_d).
- R = h(E_d), with residual width d_r selected from {64, 128}.
- Ehat_d = dec_R(R).
- L_pred = MSE(Zhat_d, StopGrad(Z_d)).
- L_info = MSE(Ehat_d, StopGrad(E_d)).

Approved magnitude objective:

\[
L_{mag}=(RMS(R)-1)^2+
\operatorname{mean}_j\left[\operatorname{ReLU}(0.1-\operatorname{std}_j(R))^2\right]
\]

The implementation computes this in float32 with epsilon 1e-6 and population standard deviation. It is evaluated separately for each dataset.

Approved redundancy objective:

\[
L_{red}=\frac{1}{128d_r}
\left\|\frac{1}{B-1}\widetilde Z_s^\top\widetilde R\right\|_F^2
\]

- Stop gradients through Z_s.
- Compute centering, standardization, and covariance independently for each dataset.
- Use full 256-row sub-batches; no EMA in the initial implementation.
- Compute covariance in float32.
- Warm lambda_red linearly from zero to its candidate target over the first 10% of planned optimizer steps.
- Candidate targets: {0.01, 0.1, 1.0}.
- Approved numerical variance_floor = 1e-4; the standard-deviation threshold comparison uses std >= variance_floor.
- The validation collapse counter counts residual dimensions with population std < 1e-4; reject when more than 25% of dimensions are below this same floor (or effective rank is below 25% of d_r).

Loss weighting and training schedule:

- lambda_pred = lambda_info = lambda_mag = 1.0.
- Equal dataset-level weighting: 0.5 NSL-KDD and 0.5 UNSW-NB15.
- AdamW: learning rate 1e-3, weight decay 1e-4.
- Gradient clipping: 1.0.
- Maximum 10,000 optimizer steps; validation every 500 steps.
- Early stop after five consecutive validation checks without at least 0.5% relative improvement in mean validation L_pred + L_info.
- Screening seed 42; finalists use seeds 42, 43, and 44.
- Training uses no labels. Labels are reserved for the separate validation probes.

## 3. Projection objective

Approved output contract: R_proj is 128-D.

Approved label-free Gram-preservation objective:

\[
R_{proj}=WR
\]
\[
K_R=\frac{R_cR_c^\top}{B-1},\qquad
K_P=\frac{P_cP_c^\top}{B-1}
\]
\[
L_{proj}=\frac{1}{B^2}\|K_P-\operatorname{StopGrad}(K_R)\|_F^2
\]

- lambda_proj = 1.0.
- The objective receives detached R; its gradient path updates projection parameters only.
- For 64-D residuals, the trainable projection is Linear(64,128) with standard PyTorch initialization.
- The fixed comparison baseline is semi-orthogonal Linear(64,128), with zero bias and frozen parameters.
- For 128-D residuals, the projection is identity.
- The four original Stage 6 losses remain unchanged.

Approved screening matrix: nine runs, all with seed 42:
- 64-D trainable projection ? lambda_red {0.01, 0.1, 1.0} (3 runs).
- 64-D fixed semi-orthogonal projection ? lambda_red {0.01, 0.1, 1.0} (3 runs).
- 128-D identity projection ? lambda_red {0.01, 0.1, 1.0} (3 runs).

The two lowest mean validation objectives among candidates passing both-dataset collapse and positive-reconstruction-skill gates advance to finalist seeds 42, 43, and 44. Seed 42 reuses its screening checkpoint.

## 4. Predictor diagnostic gate exception

The NSL-KDD predictor diagnostic recorded validation R-squared = 0.9836488962173462, above the original 0.98 stop threshold. The user explicitly approved the revised gate:

- Keep the predictor and residual definition unchanged.
- Permit Stage 6 residual-encoder training with the approved objectives.
- Retain the original R-squared result and document the exception.
- No test/unseen evaluation or Stage 7+ work is authorized by this exception.

The separate residual stability audit found nonzero residual variance and stable residual magnitude/effective-rank estimates across validation partitions. It did not establish attack-detection utility.

## 5. Validation probes

- Baseline probe input: frozen Z_s.
- Augmented probe input: frozen [Z_s, R]; do not substitute R_proj.
- Fit paired multinomial logistic-regression probes on source training features and native training labels only.
- Scale features using training-fit statistics only; balanced class weights are computed from training labels only.
- Score on source validation labels only.
- Report macro-F1, per-class precision/recall/F1/support, and per-dataset gain. The mean gain is the mean of NSL-KDD and UNSW-NB15 gains.
- Probe results are selection/reporting evidence, not Stage 6 training losses.

## 6. Implementation status at this addendum

Implemented:

- src/novel_ids/residual/stage6.py: predictor, residual encoder/decoder, 64/128 projection, four approved loss primitives, separate projection objective and gradient-isolated projection path.
- src/novel_ids/residual/stage6_training.py: frozen Stage 5 loading, train/validation-only input contract, equal-weight two-dataset training step, validation losses and residual-quality metrics, collapse/skill rejection helper, checkpoint provenance.
- src/novel_ids/residual/stage6_probes.py: paired Z_s and [Z_s,R] validation probes.
- config/stage6_projection.yaml: projection objective and initialization comparison settings.
- scripts/run_stage6_screening.py: sequential nine-run screening, validation-only gates, top-two finalist replication, paired source-validation probes, provenance, and aggregate report.
- config/stage6_projection.yaml: approved variance_floor=1e-4 and explicit nine-run matrix.
- Synthetic CPU tests: **32 passed** across Stage 6 training, probes, residual losses, and variance-clamp regression tests.

Actual-data loader preflight passed: the accepted Stage 5 snapshot loaded, all eight protected hashes matched, both source train/validation partitions were loaded, Stage 5 was frozen, and the optimizer contained Stage 6 parameters only.

Training audit:
- The first screening attempt was interrupted at approximately step 4,000 after detecting that the residual-collapse counter still used 1e-6 instead of the approved 1e-4 threshold. Its checkpoint is preserved under C:\NOVEL_IDS_runs\stage6\approved_screening_20261124_2045 and is excluded from selection.
- The counter was corrected to use std < variance_floor, with boundary tests for 16/64 versus 17/64 dimensions. Focused Stage 6 tests then passed 32/32.
- The corrected nine-run screening was restarted under C:\NOVEL_IDS_runs\stage6\approved_screening_20261124_2045_gatefix01. The first candidate is running; its best checkpoint is saved during training. This is not yet a completed screening result.

No probes, test/unseen evaluation, or Stage 7+ training has started. No Stage 5 file, checkpoint, dataset, or protected Stage 5 config was changed by Stage 6 implementation.

## 7. Final pre-training approvals

On 2026-11-24, the user explicitly approved:
- variance_floor = 1e-4.
- The nine-run screening matrix defined in Section 3.

The training runner enforces these exact settings, refuses to overwrite an existing run directory, checks Stage 5 snapshot/protected-file hashes before each run, and stops on exceptions or failed integrity checks. The first attempt was excluded and the corrected run is in progress. No test/unseen split or Stage 7+ stage is in scope.


## 8. User-approved anti-leakage correction (2026-11-24)

The initial Stage 6 loader used joblib.load() on the combined unified artifacts. Although its code selected only train and validation keys afterward, loading the combined objects also deserialized test_full and test_independent partitions. This was not acceptable under the project test-isolation rule. Screening process PID 42296 was stopped at the latest saved candidate-3 checkpoint (step 4,000; best validation objective 0.002485986928165799). That interrupted screening run is preserved for audit only and is excluded from all selection.

The user approved a narrowly scoped correction. The replacement data path is:

- scripts/build_stage6_trainval_artifacts.py reads only the dedicated audited source train/validation split artifacts, the fitted training preprocessors, and the training-only feature descriptors. It does not open the combined unified artifacts or original test files.
- It creates data/interim/stage6_trainval_only/{nsl_kdd,unsw_nb15}_inputs.joblib with only train/validation feature arrays and masks.
- It creates separate {dataset}_labels.joblib files containing only train/validation labels. The Stage 6 trainer does not load these label files; the probe workflow loads them only when probes are run.
- src/novel_ids/residual/stage6_data.py enforces the exact allowed top-level and split fields for each artifact type and rejects test partitions, labels in input artifacts, or features in label artifacts.
- Stage6Trainer.from_project now loads only the split-isolated input artifacts. run_source_validation_probes loads only the separate label artifacts.
- scripts/run_stage6_screening.py no longer breaks out of the finalist seed loop after a failed seed-43 gate; it still runs seed 44 to preserve the approved paired 42/43/44 schedule. Any candidate with a failed finalist seed remains ineligible.

Artifact creation completed successfully. Train/validation row counts: NSL-KDD 100,778 / 25,074; UNSW-NB15 140,273 / 32,999. Input artifacts contain six feature/mask fields per split and no labels; label artifacts contain only a labels field per split. The manifest is data/interim/stage6_trainval_only/manifest.json.

CPU-only actual-data preflight passed: the accepted Stage 5 snapshot hash matched, all eight protected Stage 5 file hashes matched, all four dataset/split input partitions loaded with exactly the six approved feature fields, no labels were present in trainer memory, and global_step remained 0. No GPU training was performed during preflight.

Focused Stage 6 suite after the correction: 39 passed. Python compilation passed for the changed Stage 6 modules and runner. The corrected screening has not been restarted; the earlier run's candidates 1 and 2 and partial candidate 3 are not selection evidence for the corrected data-loading workflow. A fresh screening run must use a new run directory and repeat all nine approved candidates.
