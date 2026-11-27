

# NOVEL-IDS: Consolidated Master Prompt (Single Source of Truth)

**Project:** NOVEL-IDS: Uncertainty-Aware Heterogeneous Feature Integration with Residual-Preserving Cross-Domain Intrusion Detection.

This document merges and replaces the earlier master prompt, preprocessing/missing-feature document, correction prompt, Stage 5 training prompt, and both Stage 5 addenda. If this document conflicts with an earlier one, follow this document and tell the user about the conflict.

**Role:** Act as a senior Machine Learning Engineer, Deep Learning Researcher, Cybersecurity Researcher, and Software Architect.

---

## 0. Working rules for this session

1. Continue from the current state in Section 2. Do not restart, redo completed work, or rewrite completed milestones.
2. Inspect existing code and artifacts before editing. Make the smallest change that meets the requirement. Do not do broad repository scans.
3. Work milestone by milestone (Section 15). Do not start a milestone until the previous one passes its checks. At the end of each milestone: explain what was done, give inputs and outputs, run the tests, report real results, update `PROGRESS.md`, and stop for user review.
4. The 12-stage architecture is FINAL and its order is fixed. Additional components INSIDE the pipeline are permitted when needed to make it work properly, provided each addition is minimal, justified with evidence, switchable by config, covered by an ablation entry, and reported to the user. No stage may be removed, reordered, replaced, or bypassed.
5. Never fabricate results. Report only what actually ran. Never present synthetic-data results as research results.
6. Items marked **[GO]** need a short written plan (equations, alternatives, recommendation, risks) shown to the user and an explicit "go" before implementation. Recommended defaults are given so that the plan can be short.
7. Do not treat a recommendation, proposed default, or implementation suggestion as user approval.
8. Do not use Cline. Conserve Codex usage by consolidating necessary inspections and checks into focused tasks.
9. Do not modify active training processes, checkpoints, logs, or configurations unless the user explicitly authorizes it.
10. When a decision is unresolved, preserve it as unresolved and ask the user before proceeding.

---

## 1. Objective and what "works well" means

**Primary research objective:** Preserve attack-relevant information in dataset-specific features while learning a representation that transfers across heterogeneous intrusion detection datasets, evaluated by genuine zero-shot testing on a new dataset.

The system must:

- Accept heterogeneous IDS datasets and onboard a new one automatically.
- Keep every original feature, including dataset-specific and unmatched features.
- Perform primarily MULTI-CLASS detection with a Transformer.
- Separate Normal, known attacks, Uncertain, and Unfamiliar traffic.
- Be computationally practical on an NVIDIA RTX 3050 laptop GPU with 6 GB VRAM, with CPU fallback.

"Works efficiently" is defined by measured results (Section 10), not by claims: strong and honestly reported metrics on both source datasets, a clear gain over simpler variants in the ablation, a documented zero-shot result on the unseen dataset, and reasonable training time, memory, and inference speed.

No result may be obtained by tuning on test or unseen data (Section 11.1).

---

## 2. Current state (verified by the user's progress notes; re-verify in code)

**Project root:**

`C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project\`

Completed (Milestone 2):

- Dataset inventory and profiling; NSL-KDD train 125,973 and test 22,544 rows; UNSW-NB15 files with 82,332 and 175,341 rows; zero missing values in the original files.
- NSL-KDD train/test overlap: 610 shared row hashes. UNSW-NB15: 0.
- Semantic feature registry with 5 validated common features (flow duration, transport protocol, network service, source-to-destination bytes, destination-to-source bytes). 7 candidate relationships reviewed; partial or unverified ones are not yet used.
- Preprocessing policy validated; unified schema with 78 identities (5 common at positions 0 to 4, NSL-KDD-specific 36 at positions 5 to 40, UNSW-NB15-specific 37 at positions 41 to 77); X=(V,I,M) built and saved under `data/processed/unified_representation/`.
- Artifacts per partition: `numeric_values`, `categorical_values`, `M_avail`, `M_miss`, `numeric_token_mask`, `categorical_token_mask`, `labels`; categorical positions NSL-KDD `[1,2,5]` (vocab `[3,70,11]`), UNSW-NB15 `[1,2,41]` (vocab `[133,13,8]`).
- Stages 5A and 5B implemented in `src/novel_ids/representation/dual_encoder.py` (78 x 128 identity matrix, numeric projection, dataset-specific categorical embeddings, availability and missingness representations, shared encoder giving `Z_s` of dimension 128, dataset-specific adapters giving `Z_d` of dimension 128).
- `pytest tests/test_dual_encoder.py -v` gives 8 passed.
- Real-data forward checks passed for both datasets (`Z_s (8,128)`, `Z_d (8,128)`, tokens `(8,78,128)`).
- Model training has NOT been performed.

Open items to verify before training (Section 6.0 gate): whether the unseen-readiness features in Section 5.4 exist in the current code.

These are progress-note claims, not substitutes for fresh verification. Report discrepancies if code or artifacts differ.

---

## 3. FINAL architecture

| # | Stage | Content |
|---|---|---|
| 1 | Heterogeneous Input Datasets | UNSW-NB15, NSL-KDD, new datasets |
| 2 | Automated Dataset Onboarding | Profiling, semantic feature matching, confidence and compatibility validation |
| 3 | Dataset-Aware Preprocessing and Feature Separation | Cleaning, normalization, label harmonization, multi-level feature mapping, feature identity and availability masking |
| 4 | Unified Variable-Feature Representation | X=(V,I,M) |
| 5A | Shared Encoder (parallel) | `Z_s = E_s(X)` |
| 5B | Lightweight Dataset Adapter (parallel) | `Z_d = A_d(X)` |
| 6 | Regularized Residual Information Learning | `R = f(Z_s, Z_d)` |
| 7 | Lightweight Adaptive Residual Fusion | `G = sigmoid(W_g[Z_s;R]+b_g)`, `Z_f = Z_s + G (.) R` |
| 8 | Compatibility-Aware Selective Alignment | Align compatible information, retain incompatible information |
| 9 | Unified Representation | Shared + gated residual + selectively aligned information |
| 10 | Feature Tokenization and Transformer Encoder | Learns feature interactions |
| 11 | Classification + OOD Evaluation | Multi-class classifier and unfamiliar-traffic detection |
| 12 | Final Output | Normal / Known attack category / Uncertain / Unfamiliar |

Stages 5A and 5B are parallel branches. Both receive Stage 4 output and both feed Stage 6.

Permitted training-only helpers (not stages): the Stage 5B reconstruction decoder, auxiliary heads, adversarial heads, and probe heads used only for training or diagnostics. They must be removable at inference.

---

## 4. Data

### 4.1 Sources and storage

- NSL-KDD from `https://github.com/defcom17/NSL_KDD` and UNSW-NB15 from `https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15` are already saved (Google Drive and the local project). Do NOT download data from the internet. If a file is missing, report which one and stop.
- All paths come from one config file. Support Windows local paths and Google Colab (mounted Drive).
- Original raw files are never modified.

### 4.2 Splits and leakage

- Use the official train and test partitions, and carve validation from the TRAINING partition only (stratified, fixed seed, valid for tiny classes: report class counts and ask the user before merging labels).
- **UNSW-NB15 partition check [GO]:** The profile shows 82,332 rows as train and 175,341 as test, the reverse of the official convention (train 175,341, test 82,332). Some Kaggle copies swap file names. Verify file names and row counts, report them, and ask the user which partition is used for training. Do not silently swap or silently keep.
- **NSL-KDD overlap:** Keep the original files, report the full original `KDDTest+` result, and report a separate result with the 610 overlapping records excluded. Confirm that validation does not contain test duplicates.
- NSL-KDD `KDDTest+` contains attack types absent from training. Treat them as natural unfamiliar-attack evidence for OOD evaluation, not as errors.
- Fit every preprocessing statistic on training data only.

### 4.3 Labels

- Preserve original labels and mappings. Do not invent equivalences or silently merge labels across datasets.
- UNSW-NB15 `attack_cat`: Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms.
- NSL-KDD: normal plus fine attack labels grouped by the dataset into DoS, Probe, R2L, U2R. Verify against the files.
- **Taxonomy [GO]:** Propose a harmonized taxonomy with a justification per class. Likely cross-dataset candidates: Normal, DoS, and Probe with Reconnaissance. Classes without a justified counterpart (Fuzzers, Generic, Shellcode, Worms, Analysis, Backdoor, R2L, U2R) remain dataset-specific classes. Using NSL-KDD's own five families for the Stage 5 head is a within-dataset grouping that also needs approval.
- The taxonomy contains Normal and known attack classes. "Uncertain" and "Unfamiliar" are decided at inference, not training classes, unless the user approves otherwise.

---

## 5. Stages 1 to 4: onboarding, preprocessing, representation

### 5.1 Onboarding (Stage 2)

- Profiling (record counts, dtypes, missing values, duplicates, infinities, unique values, ranges, distributions, and class distribution) is saved per dataset.
- Semantic feature matching uses names, descriptions, units, dtypes, statistics, and network meaning with lightweight local methods (no paid APIs).
- A feature registry stores identity, description, dtype, unit, semantics, candidates, confidence, and compatibility status.
- Confidence score:

  `C = w1*S_sem + w2*S_type + w3*S_unit + w4*S_range + w5*S_calc + w6*S_behav`

  Weights and thresholds must be documented. High confidence, medium confidence requiring validation, and low/incompatible relationships must be distinguished. Low-confidence relationships must not be forced.
- Compatibility validation and a reusable, reproducible configuration per dataset are required.
- Feature mapping must never use labels.
- A new dataset must not need source-code changes.

### 5.2 Preprocessing (Stage 3), approved methods

- Genuine missing measurements: training-derived median imputation; NaN, +inf, and -inf are converted to missing first. The same medians are reused for test data.
- Numeric scaling: feature-wise Robust Scaling (median and IQR from training data), with log1p for skewed non-negative features where justified. Each dataset uses its own fitted scaler. Real zeros are preserved.
- Categorical features: dataset-specific vocabularies built from training data, with an explicit UNKNOWN id for unseen categories. Use embeddings in the model, never continuous numbers.
- Structurally unavailable features are never filled with zero, mean, or median. They receive learnable missing-feature representations and an availability mask.
- Dataset-specific features are always preserved. A dataset's own specific features have mask 1 and real values; they are unavailable only for rows of the other dataset.

### 5.3 Two masks

- `M_avail`: the dataset has this feature (structural availability).
- `M_miss`: the feature exists but this value is NaN or infinite (measurement missingness).

Both reach the model. Do not mix them.

### 5.4 Unseen-dataset readiness (required design; verify and implement if missing)

1. **Computable feature identity:** `I_f = P(d_f) + b_f`. `d_f` is a semantic descriptor from name tokens, description, dtype, unit, semantic group, and unlabeled statistics (log-quantiles, skewness, fraction of zeros, number of distinct values, binary or integer flag). `P` is a small shared learnable projection; `b_f` is a learnable offset only for source features (zero for unseen ones). A never-seen feature gets an identity by the same code path. Text encoding is a lightweight local method (a small local sentence-embedding model if available, otherwise TF-IDF character n-grams plus hand-built metadata). Check installed packages before installing anything.
2. **Three-level feature mapping:** Level 1 exact or equivalent (the 5 common features share one identity); Level 2 semantic group (related, not equivalent; a coarse group label with a learnable group embedding added to each member token; groups to review against real metadata: duration/timing, byte volume, packet counts, rates/load, inter-packet timing/jitter, TTL, TCP flags/state, TCP handshake/RTT/window, connection-history window counters such as NSL-KDD `count`, `srv_count`, `dst_host_*` and UNSW-NB15 `ct_*`, error/rejection rates, login/privilege/content features, application-layer features); Level 3 dataset-specific (identity only). Each assignment stores evidence, a confidence score, and a relationship type (`exact`, `equivalent`, `partial`, `group_only`, `dataset_specific`, `unverified`). Level 2 is never recorded as `equivalent`. The user reviews the group table **[GO]**. Unseen features are assigned automatically by descriptor similarity above a threshold; otherwise Level 3.
3. **Variable-length token input:** X is a set of feature tokens. No layer has a width fixed at 78. An unseen dataset adds extension tokens after the 78 source slots; matched features reuse identities.
4. **Value tokens:** Numeric tokens are `value_proj(v_f)` conditioned on `I_f`; categorical tokens use embeddings. An unseen categorical feature with no source vocabulary uses UNKNOWN plus identity.
5. **Value masking:** During training, randomly set a small fraction (5 to 10 percent) of available values to measurement-missing and replace their token with a shared learned missing-value embedding, so that embedding receives real gradients.
6. **Unseen scaling [GO]:** Source statistics do not exist for unseen-only features. Option A (recommended): fit the scaler on the unseen dataset's own unlabeled features (no labels, no gradient updates); report this as "zero-shot with unsupervised per-dataset normalization", not strict zero-shot. Option B (strict): only Level 1 features are scaled with source statistics. State the headline choice before seeing unseen results, and report both when allowed. Unseen NaN or infinite values use `M_miss` plus the missing-value embedding, not a filled number.
7. **Mask shortcut control:** The mask pattern identifies the dataset. Use feature-set dropout (randomly hide subsets of dataset-specific features, sometimes all of them) so the model works with partial feature sets, and measure the dataset-identity probe (Section 6.4).

---

## 6. Stages 5A and 5B: representation learning

### 6.0 Gate before any training (report, do not assume)

Inspect the code and report, item by item, which of Section 5.4 items 1 to 5 exist (and how `feature_identity_vectors.npy` was built, and how the 5B adapter would produce `Z_d` for an unseen dataset). Also confirm the UNSW-NB15 and NSL-KDD split decisions in Section 4.2.

If items are missing, implement them first (this is Stage 3 and 4 and input-side work, not new stages) and rerun the Milestone 2 validation, including all 14 approved checks and the Section 12 checks, before training.

### 6.1 Training objectives

**A. Shared supervision:** A classification head on `Z_s` per dataset, with classes derived dynamically from each dataset's training labels. Use class-weighted cross-entropy with weights from training labels only: inverse square-root frequency, normalized, clipped to about 10 to 20 times between largest and smallest. Report per-class training counts and weights. Optional extra head over approved harmonized classes **[GO]** (gives a label-aligned cross-dataset signal).

**B. Dataset-specific preservation:** A reconstruction decoder (training only; removable at inference) on `Z_d`.

- Masked-feature modelling: hide 15 to 30 percent of available dataset-specific values at the input and compute the loss on hidden positions (plus a small weight on visible ones). Validation uses a fixed seeded mask.
- Token-wise and identity-conditioned: predict feature `f` from `Z_d` and `I_f`.
- Numeric loss: Huber on robust-scaled values. Categorical loss: cross-entropy over the correct dataset vocabulary; UNKNOWN is not a target.
- Only dataset-specific positions (not the five common features); unavailable or missing positions are excluded; NaN at inactive positions cannot reach the loss.
- Average numeric and categorical terms separately over valid entries; configurable weights chosen on validation data only; log every term.

### 6.2 5B for an unseen dataset (the adapter strategy)

Per-dataset adapter weights cannot exist for an unseen dataset. Use one shared lightweight adapter conditioned on a dataset context vector `c` (for example FiLM: `gamma(c)` and `beta(c)` modulate the adapter's hidden layer).

`c` is computed only from onboarding output: pooled descriptor vectors and unlabeled summary statistics (mean of `P(d_f)`, fraction of available features per semantic group, mean skewness). No labels and no gradient steps on the new data.

Feature-set dropout and random feature-subset sampling create many pseudo-datasets so that `c` varies. Source and unseen data use the same code path. Test it with leave-one-dataset-out runs (Section 11.5). If it does not work well, report numbers and propose an alternative; do not claim zero-shot works.

### 6.3 Training loop

- Draw one equal-size sub-batch from each dataset in every step; each uses its own head and decoder.
- LayerNorm, not BatchNorm; AdamW; warm-up then cosine learning-rate decay; gradient clipping 1.0; fixed seeds; mixed precision on the RTX 3050 with CPU fallback; tensors preloaded on the device and no DataLoader workers (Windows).
- Early stopping and model selection on validation macro-F1 averaged over the two datasets (not accuracy), with per-dataset curves. Test partitions are never read.
- Save best and last checkpoints with optimizer state, label encoders and class mappings, configuration, loss history, and best-checkpoint metadata. Write to a temporary file and rename.
- The project is under OneDrive: write checkpoints and logs to a local non-synced directory (for example `C:\NOVEL_IDS_runs\`) and copy final artifacts into the project afterwards, or pause sync.
- Keep `dual_encoder.py` focused on the encoders. Put the training code in `src/novel_ids/representation/stage5_training.py`.

### 6.4 Diagnostics (report numbers; none are stages)

1. Dataset-identity probe on frozen `Z_s` (logistic regression). Report accuracy honestly.
2. Common-features-only evaluation of the `Z_s` heads.
3. Label probe on frozen `Z_d`.
4. Masked reconstruction error versus the mean (numeric) or mode (categorical) baseline.
5. Classical baseline (scikit-learn, for example HistGradientBoosting) per dataset on validation data.
6. Gradient check: non-zero gradients on missing-feature representations (on the 5A/5B path) and the missing-value embedding. Report zero gradients instead of hiding them.
7. NaN and infinity checks across inputs, embeddings, and losses.

**Conditional additions:**

- If the label probe shows `Z_d` lacks attack information, add a small classification head on `Z_d`.
- If the dataset-identity probe on `Z_s` is high, add a dataset-adversarial head (gradient reversal) with weight chosen on validation data and keep it only if the probe falls without hurting validation macro-F1.

Any conditional addition must be justified, configuration-switchable, tested, and included in the ablation record.

### 6.5 Stage 5 exit criteria

Finite losses; best checkpoint reloads with identical validation metrics; per-dataset validation macro-F1 reported against the classical baseline and any large gap explained; reconstruction beats the baseline; probes are reported; all focused tests pass; `PROGRESS.md` updated.

---

## 7. Stages 6 to 9

Recommended defaults in this section require review where specified. Approval of a decision family does not approve every implementation default.

### 7.1 Stage 6: residual learning — Decisions 7, 7b and 8 approved; implementation specification pending user review

The user approved:

- **Decision 7 — residual extraction:** Predict the dataset-specific representation from the shared representation and encode the prediction error.
- **Decision 7b — objective families:** Use magnitude stability, shared/residual redundancy reduction, and reconstruction-based information preservation.
- **Decision 8 — redundancy estimator:** Use Option A, a standardized cross-covariance penalty with all seven safeguards below.

Approval of Decisions 7, 7b, and 8 does not approve every implementation default below or resolve any other `[GO]` item.

Do not implement or train Stage 6 until the user approves this specification, the Section 6.0 gate, and Stage 5 exit criteria.

#### Inputs, outputs and shapes

For source dataset sub-batch `k` of size `B_k`, Stage 5 supplies:

- `Z_s,k ∈ R^(B_k × d_s)`
- `Z_d,k ∈ R^(B_k × d_d)`

Current Stage 5A/5B code emits dimensions 128 and 128. Recommend `d_s=d_d=d_r=128`.

Stage 6 returns:

- `Zhat_d,k ∈ R^(B_k × d_d)`
- `E_k ∈ R^(B_k × d_d)`
- `R_k ∈ R^(B_k × d_r)`

It does not replace, concatenate around, or reorder the twelve stages. Stage 7 consumes `R` in its existing gated-fusion position.

#### Predictor `g` and residual encoder `h` (proposed defaults)

Use a small predictor `g_φ: R^d_s → R^d_d`, proposed as:

`Linear(128,128) → GELU → LayerNorm → Linear(128,128)`.

Train it only with:

`Zhat_d = g_φ(StopGrad(Z_s))`

`L_pred,k = ||StopGrad(Z_d) - Zhat_d||²_F / (B_k d_d)`.

Both the predictor input and target are detached for `L_pred`, so this loss updates `φ` only.

For residual use, detach the fitted predictor result as well:

`Zhat_d_sg = StopGrad(Zhat_d)`

Form:

`E_d = Z_d - Zhat_d_sg`

`R = h_θ(E_d)`.

Recommend `h` as:

`Linear(d_d,128) → GELU → LayerNorm → Linear(128,d_r)`

with no output normalization that would erase residual magnitude.

The final projection dimension is 128 unless a future approved fusion design requires another size.

#### Decision 8: standardized cross-covariance penalty

For each source dataset independently, stop gradients on `Z_s`; center `Z_s` and `R` by their own sub-batch means and standardize each dimension by its own sub-batch standard deviation.

Let `I_s` and `I_r` contain dimensions with finite standard deviation at or above the configured floor.

For active dimensions:

`C_k = (Zs_tilde_k[:, I_s]^T R_tilde_k[:, I_r]) / (B_k - 1)`

`L_red,k = ||C_k||²_F / (|I_s| |I_r|)`

`L_red = mean_k L_red,k`

The mean is taken over source datasets with valid active dimensions.

The per-dataset value is zero/skipped with a logged reason if there are too few rows or no active dimensions. Implementation must not divide by zero.

This is mean squared linear cross-correlation, not a test of all dependence.

**Safeguards required in implementation and tests:**

1. **Stop-gradient on `Z_s`** within `L_red`; it regularizes the residual, not the shared encoder.
2. **Per-dataset statistics:** Calculate centering, variances, and cross-covariance within each dataset sub-batch, then average valid losses. Never pool datasets before covariance estimation.
3. **Variance floor:** Mask dimensions whose standard deviation is below a configured threshold; log masked dimension counts per dataset and step. Proposed numerical epsilon is `1e-6`; proposed standard-deviation floor is `1e-4` in the unstandardized representation units. Both are reviewable defaults.
4. **Adequate batch size:** Propose `B_k=256` per source dataset for covariance estimation, report the actual size, and check stability across batches. If memory prevents this, use an explicitly documented EMA of per-dataset statistics; gradient accumulation alone does not increase covariance sample count.
5. **Warm-up:** Ramp `lambda_red` linearly from zero to its validation-selected target over the first 10% of planned Stage 6 optimizer steps (configurable).
6. **No hard decorrelation constraint:** Keep `L_info` active throughout training so redundancy reduction cannot be satisfied by collapsing `R` to a constant or zero.
7. **Float32 covariance:** Compute standardization and covariance in float32 even under mixed precision.

The proposed diagnostic reports mean absolute active cross-correlation.

Decision 8 only penalizes linear dependence. Separately, for nonlinear dependence, freeze the trained Stage 6 representations, fit a small MLP to predict `R` from `Z_s` on a deterministic 80% split of each source validation partition, and report held-out `R²` on the remaining 20%. This is a diagnostic only, not a loss; it must not use test or unseen data.

#### Magnitude stability (proposed defaults)

Let:

`rms(X)=sqrt(mean(X²)+epsilon)`

`rho=rms(R)/(rms(E_d)+epsilon)`

`s_j=std_b(R[:,j])/(rms(E_d)+epsilon)`

Recommend a dimensionless range `rho ∈ [0.1,1.0]` and per-dimension floor `s_j ≥ 0.01` as initial, validation-reviewable targets:

`L_mag = max(0,0.1-rho)² + max(0,rho-1.0)² + mean_j max(0,0.01-s_j)²`.

This penalizes global collapse, extreme residual energy, and dead dimensions without forcing the residual to have a particular direction.

These target values are proposals, not measured optima. Tune only on source validation partitions, and log `R`/`E_d` RMS, per-dimension standard deviations, and active covariance dimensions.

Risks include forcing noise into legitimately inactive dimensions or allowing an overly large residual; inspect per-class and per-dataset values.

#### Reconstruction-based information preservation — approved objective

The approved Stage 6 information-preservation objective reconstructs the prediction error from the residual representation:

`E_d = Z_d - StopGrad(Zhat_d)`

`R = h(E_d)`

`E_d_hat = dec_R(R)`

`L_info = MSE(E_d_hat, StopGrad(E_d))`

This is the primary Stage 6 `L_info`.

It is distinct from the Stage 5 masked-feature reconstruction objective. It does not require the Stage 5B feature decoder `D_5B`.

With an unbottlenecked residual representation, this reconstruction objective is mainly an anti-collapse/information-preservation term: it encourages `R` to retain enough information to reconstruct `E_d`. It does not, by itself, establish that `R` contains useful attack-discriminative information.

Residual usefulness for detection must be evaluated using validation-only probes comparing frozen `Z_s` against frozen `[Z_s, R]`. Report per-dataset macro-F1 and class support for UNSW-NB15 and NSL-KDD, using identical validation partitions and evaluation conditions. Also report the approved residual-quality metrics.

**Gradient interpretation:** The target branch `StopGrad(E_d)` does not receive gradients. The prediction branch passes through `dec_R` and `h`. Because `E_d = Z_d - StopGrad(Zhat_d)` is not itself detached before entering `h`, gradients can also reach the Stage 5 path that produces `Z_d` unless Stage 5 representations are frozen. Gradients do not pass through `Zhat_d` or predictor `g`. The exact Stage 5 freezing policy and complete detach configuration remain subject to review and approval.

#### Optional Stage 6 masked-feature reconstruction ablation

The Stage 5 masked-feature objective remains:

`L5 = L_classification + lambda * L_reconstruction`

It is not Stage 6 `L_info`.

An optional, configuration-switchable Stage 6 ablation may test whether adding `R` improves the existing Stage 5 masked-feature reconstruction path:

`Xhat_base,f = D_5B(StopGrad(Zhat_d), I_f, dataset)`

`Xhat_full,f = D_5B(StopGrad(Zhat_d) + P_R(R), I_f, dataset)`

This optional ablation requires the frozen Stage 5B decoder `D_5B` and may only be considered after the Stage 5 checkpoint is accepted.

Status: **proposed, not approved**. It must be OFF by default and must not replace, modify, or be conflated with the approved MSE `L_info`.

#### Total objective and coefficient selection

For each dataset `k`, calculate its terms separately; average across the two source datasets only after each term is computed per dataset:

`L_6 = lambda_mag L_mag + lambda_red(t) L_red + lambda_info L_info + lambda_pred L_pred`.

Recommend initial `lambda_mag=lambda_info=lambda_pred=1`; compare `lambda_red,target ∈ {0.01, 0.1, 1.0}` using source validation only.

These are starting proposals, not approved final values. Log every per-dataset loss and coefficient. Do not tune on official test or unseen data. Use identical splits, seeds, and compute budgets for Stage 6 ablations.

#### Gradient flow and stage interaction

- `L_pred` updates `g` only; `Z_s` and `Z_d` are detached for this objective.
- `L_red` stops gradients at `Z_s`; it may update `h` through `R`. Because `R=h(E_d)` and `E_d` contains `Z_d`, gradients may also reach the Stage 5B path unless the Stage 5 representations are frozen. Float32 covariance does not alter the stop-gradient boundary.
- `L_mag` updates the residual encoder `h` through `R`. Gradients may also reach the Stage 5B path through `E_d` unless Stage 5 representations are frozen. The primary Stage 6 objective does not use `P_R`; therefore, do not state that `L_mag` updates `P_R` as part of the primary objective.
- `L_info` uses the approved reconstruction objective: `L_info = MSE(E_d_hat, StopGrad(E_d))`, where `E_d = Z_d - StopGrad(Zhat_d)` and `E_d_hat = dec_R(R)`, with `R = h(E_d)`. Gradients flow through `dec_R` and `h`, and potentially through the Stage 5B path producing `Z_d` unless frozen. They do not flow through `Zhat_d` or predictor `g`. The target `E_d` is detached in the target branch. The complete detach configuration and exact gradient-flow behavior across all Stage 6 objectives remain subject to final implementation specification and approval.
- The Stage 6 information-preservation objective does not require the Stage 5 decoder. Using the frozen Stage 5B decoder for residual-assisted masked-feature reconstruction is an optional ablation, proposed but not approved, and must remain disabled by default. Whether Stage 5B parameters receive Stage 6 gradients is unresolved. The recommended initial approach is to fit Stage 6 against frozen Stage 5 representations. Any later Stage 5–11 joint fine-tuning requires the separate Section 9 `[GO]` decision.
- Stage 5 per-dataset classification and reconstruction objectives remain intact. Stage 6 adds no new label input and does not replace Stage 5 heads.
- Stage 6 outputs `R` to Stage 7; it does not implement or train the Stage 7 gate. Backpropagation through Stage 7 into Stages 5–6 is governed by the later approved joint-training plan and must not be assumed here.

#### Residual quality reporting

Report per dataset and training epoch:

- Active-dimension mean absolute cross-correlation.
- `R` variance by dimension.
- Effective rank: `rank_eff=(tr(Σ_R))²/tr(Σ_R²)`.
- `R`/`E_d` norm and RMS distributions.
- Residual norms by class.
- Primary residual reconstruction MSE.
- On held-out source validation rows, nonlinear MLP prediction `R²`.

Masked-feature reconstruction losses with and without `R` belong only to the optional Stage 6 masked-feature reconstruction ablation. Do not present them as the primary `L_info` result.

If classification is later evaluated with and without `R`, use a frozen common checkpoint and identical data/budget. Do not describe synthetic tensor tests as research results.

#### Unseen inference path

The approved Option A onboarding path supplies descriptor-based `Z_s` and context-conditioned shared-adapter `Z_d` from the same Stage 5 code path.

At unseen inference, apply frozen `g` and `h` to produce `Zhat_d`, `E_d`, and `R`, then pass `R` to the approved later stages.

Use no unseen labels, test outcomes, model updates, gradient steps, or threshold/loss selection. Describe the normalization protocol as **zero-shot with unsupervised per-dataset normalization**.

No unseen-data fitting beyond the already approved unlabeled normalization is introduced by Stage 6.

#### Synthetic tests required before Stage 6 implementation

Implement these tests only after approval, using synthetic tensors and no research data:

1. Independent random `Z_s`,`R` gives near-zero `L_red` within a documented finite-sample tolerance; a linear `R` function of `Z_s` gives a larger value.
2. `L_red` is invariant to positive or negative rescaling away from the variance floor.
3. No gradient reaches `Z_s` through `L_red`.
4. Low-variance dimensions are masked, counted, and produce finite loss; constant `R` does not produce NaN/Inf.
5. A dataset-shift fixture proves per-dataset statistics are used rather than pooled statistics.
6. Batch-size policy rejects or explicitly skips an undersized covariance batch and accepts the configured minimum; if EMA is used, a synthetic chunked-statistics test checks it against full-batch statistics within tolerance.
7. The `lambda_red` warm-up begins at zero and reaches its target at the configured step; `L_info` remains active while `lambda_red=0`, and collapsed `R` receives a magnitude penalty.
8. Under an autocast/mixed-precision test context, standardization/covariance remain float32 and finite.
9. `L_pred` updates only `g`.
10. The magnitude floor detects collapsed `R`; all losses/gradients are finite and gradients reach `h`. Any gradient reaching the Stage 5B path must match the explicitly selected freeze policy.
11. The approved MSE reconstruction objective computes `MSE(E_d_hat, StopGrad(E_d))`, produces finite loss, and sends gradients through `dec_R` and `h` without sending gradients through `Zhat_d` or `g`.
12. If the optional masked-feature ablation is separately approved and enabled, test its masking and exclusion rules: unmasked, missing, unavailable, and UNKNOWN categorical targets must be excluded according to the approved Stage 5 reconstruction rules.
13. The unseen forward path returns finite `R` without labels, gradient steps, or parameter updates.

#### Stage 6 choices still requiring review

Decision 8's cross-covariance method and safeguards are approved.

Before implementation, obtain user review/approval of the complete specification, including:

- Proposed predictor and residual architectures.
- `d_r`.
- Frozen-versus-fine-tuned Stage 5 parameters.
- Magnitude targets.
- Variance floor.
- Batch/EMA policy.
- Warm-up.
- Reconstruction decoder architecture `dec_R`.
- Reconstruction baseline and any skill/improvement metric.
- Residual dimension choices.
- Coefficients and schedules.
- Exact detach list and gradient-flow configuration.
- Whether the optional Stage 6 masked-feature reconstruction ablation is approved.

Do not treat these proposed numeric defaults as additional approved decisions. Decisions for Stages 7–11 and Section 9 joint training remain separate and unresolved.

### 7.2 Stage 7: adaptive fusion

`G = sigmoid(W_g[Z_s;R] + b_g)`

`Z_f = Z_s + G (.) R`

Project `R` to the dimension of `Z_s` if needed. Initialize `b_g` negative (for example, -2) so the gate starts conservative.

Gate values must remain in (0,1); the residual must not overwhelm shared information. Log gate mean, standard deviation, and histogram per class and per dataset every epoch.

Exact Stage 7 implementation choices require review before implementation.

### 7.3 Stage 8: compatibility-aware selective alignment [GO]

Recommended: compute a compatibility weight `C` in (0,1) per dataset pair and per sample from:

(a) onboarding match confidence of the features present; and  
(b) a statistical distance between datasets in representation space.

Aligned part:

`A = W_a Z_f`

Train with a distribution-alignment loss (MMD or CORAL between datasets) weighted by `C`, so only compatible information is pulled together.

Retained part:

`U = W_u Z_d`

Weight it by `(1-C)`.

Define the exact loss, explain how incompatible information is retained, and never use blind concatenation.

### 7.4 Stage 9: unified representation

`Z_u = Z_s + G (.) R + C (.) A + (1 - C) (.) U`

Adapt the tensor operations without changing the structure. Save intermediate outputs and configuration; keep dimensions, sample correspondence, and feature-identity correspondence consistent.

---

## 8. Stages 10 to 12

### 8.1 Stage 10: tokenization and Transformer [GO]

`Z_u` is a pooled vector while the Transformer needs feature tokens.

Recommended:

`T_f = token_f + W_t Z_u`

The Stage 5 token embeddings include identity, value, group, and mask information, conditioned on `Z_u`, plus a CLS token initialized from `Z_u`.

Components:

- Input embedding.
- Feature-identity information.
- Multi-head self-attention (Q, K, V, scaled dot-product).
- Feed-forward network.
- Residual connections.
- Layer normalization.
- GELU.
- Dropout.
- Output representation `Z_t`.

Use `M_avail` as the attention key-padding mask so unavailable tokens never take part in attention.

Unit test: changing an unavailable token changes no logits.

Use CLS or masked mean pooling.

Size for 6 GB VRAM (for example, 2 to 4 layers, 4 to 8 heads, hidden 128, dropout 0.1); report parameter count and memory.

Do not replace it with an MLP or CNN.

### 8.2 Stage 11: classification and OOD [GO]

- Multi-class classifier on the approved taxonomy: `softmax(W_c Z_t + b_c)`, class-weighted cross-entropy (optionally label smoothing).
- Calibrate confidence with temperature scaling fitted on validation data only.
- Unfamiliar-traffic score from model outputs and embeddings, for example Mahalanobis distance on `Z_t` (class-conditional Gaussians with shared covariance, fitted on source training data), energy score, and k-NN distance.
- Recommended: Mahalanobis plus energy, combined and thresholded on validation data.
- Softmax confidence alone is not evidence that traffic is familiar.
- Threshold calibration uses only source validation data.
- A development protocol that holds one or two attack classes out of training and calibrates on them (never on test or unseen data) is allowed with approval.

### 8.3 Stage 12: final output

Per sample:

- Normal traffic.
- Attack detected with category and confidence.
- Uncertain (low confidence).
- Unfamiliar (high OOD score).

Human-readable text such as "Attack detected: DoS (confidence 0.93)."

Never output a category outside the learned taxonomy.

---

## 9. Overall training strategy [GO]

Initialize from the Stage 5 checkpoint and train Stages 5 to 11 jointly at a lower learning rate with auxiliary losses at small weights (recommended).

Alternative: freeze Stage 5 and train later stages only.

Every ablation variant starts from the same Stage 5 checkpoint with the same training budget.

Total loss:

`L = L_cls + a*L_rec + b*L_res + c*L_align + d*L_gate_reg (+ optional adversarial term)`

Weights are chosen on validation data only and documented.

The exact joint-training plan, including which stages are frozen or updated and the gradient paths between them, requires an explicit `[GO]` decision.

---

## 10. Performance and efficiency plan (how to get good, honest metrics)

1. **Baselines for context:** Validation and test, each dataset separately: HistGradientBoosting or Random Forest on the preprocessed features; a plain MLP; and a plain Transformer on the common feature set. The proposed model is judged against these. A result worse than a simple baseline must be reported and investigated, not hidden.
2. **Tuning on validation data only:** A small seeded random search over learning rate, hidden size, depth, dropout, loss weights, feature-set dropout rate, and masking rate. Budget it for the 6 GB GPU. Record every trial.
3. **Imbalance handling:** Capped class weights; optional per-class focal or logit adjustment if validation macro-F1 on rare classes stays poor. Report per-class recall for rare classes. No synthetic oversampling across partitions.
4. **Multiple seeds:** At least 3 seeds for the final model and the ablation variants; report mean and standard deviation.
5. **Calibration and thresholds:** Temperature scaling, Uncertain threshold, and OOD threshold on validation data only.
6. **Efficiency measurements:** Parameter count, peak GPU memory, training time per epoch, inference throughput (flows per second), and CPU fallback speed.
7. **Support values:** The support column in a classification report is the number of true samples per class. It describes the dataset and is not improved by the model. Report it with every per-class table, and interpret per-class scores with small-support classes (for example, NSL-KDD U2R and R2L, UNSW-NB15 Worms and Analysis) with caution. The model can only improve recall and F1 on those classes, and this must be measured, not assumed.
8. **Sanity expectations:** NSL-KDD `KDDTest+` accuracy is usually well below validation accuracy because of test-only attacks; multi-class UNSW-NB15 accuracy is usually lower than binary accuracy. These are general literature expectations, approximate and not targets. Do not tune to chase a number. Large unexplained gaps between validation and test are a leakage or distribution-shift investigation trigger.

---

## 11. Evaluation protocol

### 11.1 Anti-gaming rules

Test partitions and the unseen dataset are never used for gradient updates, preprocessing fitting, model selection, loss-weight selection, threshold selection, hyperparameter tuning, or label mapping.

Results are evaluated once per final configuration.

### 11.2 Source evaluation (each dataset separately, never merged into one score)

- UNSW-NB15 test.
- NSL-KDD `KDDTest+` (full original).
- NSL-KDD `KDDTest+` without the 610 overlapping records.
- Per-class results.

### 11.3 Metrics

Confusion matrix and normalized confusion matrix (multi-class), accuracy, precision, recall, F1, macro F1, weighted F1, per-class precision, recall, F1 and support, ROC-AUC and PR-AUC where appropriate (one-vs-rest, documented), False Positive Rate, False Negative Rate, Detection Rate, plus OOD metrics (AUROC, AUPR, FPR at 95 percent TPR) where ground truth permits, the Uncertain rate, and accuracy on accepted (non-uncertain) predictions.

Document averaging methods.

### 11.4 Unseen dataset (for example CIC-DDoS2019; supplied later, never touched before)

Onboard with no source-code changes, then infer with the trained model, with no supervised retraining or fine-tuning and no use of unseen labels for matching, preprocessing, or prediction.

Evaluate with labels separately.

A ground-truth class outside the trained taxonomy is not silently mapped: report it as unsupported, report known-class performance separately, and evaluate unfamiliar-traffic detection where ground truth permits.

Without labels, provide predictions and uncertainty statistics only.

Label the run accurately (strict, or with unsupervised normalization).

### 11.5 Development-time pseudo-unseen check

Train on NSL-KDD only and test on UNSW-NB15, and the reverse, on classes present in both approved taxonomies.

These are diagnostics. The real unseen dataset must not influence any design choice.

Expect limited zero-shot performance with only two sources and five exact common features, and report what is measured.

### 11.6 Graphs

Training and validation curves, gate statistics, confusion matrices, per-class bars, OOD score histograms, and reliability diagrams.

---

## 12. Validation checklist (all must pass before the dependent milestone)

The 14 approved preprocessing checks:

1. 41 NSL-KDD and 42 UNSW-NB15 inputs preserved.
2. 78 identities retained.
3. Original measurements unchanged.
4. Training-derived medians for genuine missing values.
5. Scalers and vocabularies fitted on training only.
6. Unavailable features get learnable representations, not fabricated values.
7. Identity and mask dimensions match.
8. No NaN or infinity in model inputs.
9. Consistent dimensions.
10. Finite forward pass and loss.
11. Gradients valid for learnable missing representations.
12. Train/test separation preserved.
13. Original multi-class labels preserved.
14. All required preprocessing artifacts and mappings are reproducible.

Additional checks:

- New-schema onboarding on a synthetic dataset (different names, order, extra features, an all-NaN column, an infinite column, an unknown category; synthetic data only).
- Identity for an unseen feature is finite and distinct.
- Every Level 1 and Level 2 assignment has evidence and confidence, and Level 2 is never `equivalent`.
- Input path accepts token counts other than 78.
- Changing an unavailable token changes no logits.
- Gradients reach the missing-feature representations and missing-value embedding.
- Dataset-identity probe runs.
- Leakage scan shows no fitted statistic from test or unseen data (the approved unseen normalization is tagged `unsupervised_calibration`).
- No label is used as an input.
- Checkpoint save and load reproduce metrics.
- Every added component has a config switch.

Tests also cover: profiling, semantic matching, confidence scoring, compatibility validation, configuration generation, imputation, scaling, categorical encoding, label harmonization, feature separation, identity generation, masking, unified representation, both Stage 5 branches, residual and its regularizer, gating, alignment, tokenization, Transformer forward pass, classification, OOD scoring, onboarding and inference for a new dataset, and metric calculations.

Stage-specific tests must not be implemented ahead of their milestone approval.

---

## 13. Ablation study (controlled: same splits, budget, seeds, Stage 5 starting checkpoint)

1. Shared encoder only.
2. Shared encoder + residual information.
3. Shared encoder + residual + adaptive gating.
4. Full NOVEL-IDS with compatibility-aware selective alignment (primary model).

Additional controlled experiments:

- Common-features-only model.
- Full model without feature-set dropout.
- With and without the approved Stage 6 residual reconstruction loss.
- With and without the optional Stage 6 masked-feature reconstruction ablation, only if separately approved.
- Adapter conditioning on and off for the zero-shot path.

Measure accuracy, macro F1, weighted F1, per-class recall, cross-dataset generalization, residual contribution (norm and share of the fused representation), and gate behaviour.

Report negative results honestly. Do not remove components from the final implementation; use config switches.

---

## 14. Environment

- Windows with VS Code, a Python virtual environment, NVIDIA RTX 3050 with 6 GB VRAM (batch size and model dimensions tuned to fit, mixed precision, gradient accumulation if needed), CPU fallback; Google Colab or Kaggle as options with Drive checkpoints.
- Fixed seeds, saved configs, resumable training.
- No paid APIs.
- Avoid new dependencies unless justified, and check installed packages first.

---

## 15. Milestones remaining

1. **Gate (Section 6.0):** Report on unseen-readiness features and split decisions; implement what is missing; rerun validation. Stop for user review.
2. **Stage 5 training:** Implement `stage5_training.py`, tests, run training, diagnostics, exit criteria (Section 6.5). Stop.
3. **Stages 6 and 7:** Plans for `[GO]` items, implementation, residual quality and gate analysis. Stop.
4. **Stages 8 and 9:** Selective alignment and unified representation, evidence that compatible information is aligned and incompatible information retained. Stop.
5. **Stages 10 and 11:** Transformer, classifier, calibration, OOD; train jointly (Section 9). Stop.
6. **Source evaluation (11.2, 11.3, 11.6):** Run with the baselines of Section 10.
7. **Unseen-dataset evaluation (11.4):** Begin only after the user supplies the dataset.
8. **Ablation and finalization:** Section 13, documentation, README with exact run commands, final results summary, reproducibility check.

Project structure follows the existing layout (`src/novel_ids/...`, `data/`, `configs/`, `onboarding/`, `preprocessing/`, `models/`, `training/`, `evaluation/`, `inference/`, `saved_models/`, `reports/`, `tests/`, `PROGRESS.md`, `README.md`, `requirements.txt`).

Keep `PROGRESS.md` current so work can resume exactly.

Final deliverables: source code, checkpoints, onboarding configs, feature registry, profiling reports, preprocessing configs, training logs, validation and test metrics, unseen metrics, confusion matrices, classification reports with support, ablation results, graphs, equations for residual, gating and alignment, and README.

---

## 16. Strict rules

1. Follow the architecture; never silently redesign it. Added components need justification, a config switch, and an ablation entry.
2. Never remove a component or skip a stage.
3. Multi-class classification is primary; never replace it with binary classification.
4. Never discard uncommon features. Never force incompatible feature mappings. Never replace semantic matching with column-name matching.
5. Never zero-fill, mean-fill, or median-fill structurally unavailable features.
6. Never introduce leakage. Never use test or unseen data, or unseen labels, for any decision.
7. Never claim zero-shot if any supervised adaptation or unreported calibration was used.
8. Never fabricate metrics. Never report synthetic-data results as real.
9. Never claim an unseen attack is recognized if it was not learned. Never claim a feature is transferable without validation.
10. Never invent attack-category equivalences or silently merge labels.
11. Never download datasets from the internet. Never modify raw files.
12. Never overwrite working code without checking the existing implementation. Never install unnecessary dependencies.
13. Never declare the project complete until the real source-dataset and unseen-dataset evaluations have been performed.
14. If a requirement is technically impossible under the architecture, explain exactly why and ask for a decision.

**Priority order:** Architecture preservation, scientific correctness, genuine unseen-dataset generalization, reproducibility, then efficiency, above any shortcut.



---

## Approval addendum 2026-11-26

The project owner has approved each [GO] item listed in `docs/APPROVALS_2026-11-26.md`. That register is the authoritative approval record for the listed Stage 6 final configuration, Section 9 frozen Stage 5/6 and train-Stages-7-to-11-only policy, Stages 7–12, baselines, ablations, diagnostics, one-shot source-test evaluation, and the unseen-data protocol. For any listed item whose original specification line still displays `[GO]`, treat the item as approved and follow the exact decision in the approval register. This addendum is append-only and does not authorize any other change to this master specification. All hard stops, anti-leakage requirements, and protected Stage 5/data/split/preprocessing constraints remain in force.


## Approval addendum 2026-11-26

The project owner has approved every [GO] item listed in `docs/APPROVALS_2026-11-26.md`. That register is the authoritative record of the approved decisions, source-message references, and status. For any listed item, this addendum supersedes a remaining [GO] marker in the earlier specification text. Stage 5 and Stage 6 remain frozen; the approved Section 9 policy trains Stages 7–11 only and does not authorize joint fine-tuning. The Stage 6 original-rule result and the approved revised-criterion Candidate 09 / lambda_decorr=1.0 selection must both be reported, with the required qualification wording and disclosure that the revision was made after results were seen, on logical grounds. All hard stops, data-leakage protections, protected-file constraints, and one-shot test rules remain binding.
