# NOVEL-IDS inference API

## Public API

from novel_ids.inference import load_model, onboard_csv, predict_csv, evaluate_predictions

model = load_model(r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43")
info = model.model_info()
onboarding = onboard_csv("new_traffic.csv", model=model, label_column="Label")
predictions, summary = predict_csv("new_traffic.csv", model, onboarding=onboarding, batch_size=256, chunk_size=8192)

- load_model(checkpoint_dir, device=None): loads frozen primary checkpoint, Stage 5/6 artifacts, Stage 7-11 model, taxonomy, calibration and source-training OOD references.
- model.model_info(): returns version, date, seed, checkpoint/protected hashes, taxonomy, device and frozen thresholds.
- onboard_csv(path, model=None, label_column=None, normalization_path=None, chunksize=8192, calibration_rows=50000): profiles the CSV in bounded chunks and fits Option A median/IQR normalization on a deterministic, uniformly sampled, unlabeled calibration sample (up to 50,000 rows).
- predict_csv(path, model, onboarding=None, label_column=None, batch_size=256, chunk_size=8192): rereads the CSV in chunks and performs batched inference using frozen onboarding descriptors and calibration statistics.
- evaluate_predictions(predictions, labels, classes, label_map=None): optional post-inference evaluation. Labels are passed only here, never into onboarding or prediction.

## Input contract

Input must be a non-empty CSV with unique, stable column names/order. Raw numeric and categorical features are accepted. NaN and infinities are treated as missing; all-NaN columns remain available but marked missing. Pass the outcome column explicitly with label_column / --labels-column so it is removed before profiling, calibration and prediction.

A separate normalization CSV must contain no label/class/attack/target column and must have exactly the same feature names/order. The deterministic calibration sample is capped at calibration_rows; all rows are still predicted.

## Prediction columns

- row_index: zero-based original input row.
- decision: Normal, Known attack, Uncertain or Unfamiliar.
- predicted_category: category from the frozen taxonomy.
- confidence: temperature-calibrated maximum class probability.
- ood_score: frozen unfamiliarity score.
- message: human-readable decision explanation.
- source_branch: selected frozen NSL-KDD or UNSW-NB15 residual branch.
- prob_<class>: calibrated probability for each taxonomy class.

## Summary JSON

The summary JSON schema is:

```json
{
  "rows": 12,
  "decision_counts": {"Normal": 0, "Known attack": 0, "Uncertain": 0, "Unfamiliar": 12},
  "class_counts": {"Normal": 0, "DoS": 0},
  "source_branch_counts": {"nsl_kdd": 12},
  "thresholds": {"temperature": 1.0, "tau_ood": 0.0, "tau_conf": 0.0},
  "unseen_normalization": {},
  "onboarding": {"profile": {}, "feature_matches": [], "compatibility": {}, "config": {}},
  "model": {"version": "NOVEL-IDS-1.0", "seed": 43, "checkpoint_sha256": "..."},
  "streaming": {"enabled": true, "chunk_size": 8192, "batch_size": 256}
}
```

The values above illustrate field types and shape; use the accompanying validation example JSON for actual frozen values. `class_counts` reports taxonomy categories only; rejected `Uncertain`/`Unfamiliar` rows have no predicted category. The onboarding `config` records ordered feature columns, per-feature match confidence, compatibility, and the unlabeled Option A normalization scope/statistics.

## CLI

Set PYTHONPATH to src, then run:
python -m novel_ids.inference.predict --input new_traffic.csv --output predictions.csv --summary predictions_summary.json --labels-column Label --batch-size 256 --chunk-size 8192

An optional separate unlabeled calibration CSV can be supplied with --normalization-input. The CLI reports concise exception messages instead of stack-trace-only failures.

## Resource and safety behavior

CSV scanning and model-input construction are chunked. Only the bounded calibration sample and final prediction DataFrame are retained. Model execution is additionally split into batch_size batches. Inference stops below 2 GiB free RAM or above 6 GiB GPU allocation. CUDA is optional; use device='cpu' to force CPU execution. The final DataFrame grows with output rows because that is the public API contract.


## Validation-data example files

The input file contains 12 raw NSL-KDD rows sampled from the frozen, training-derived validation split (25,074 validation rows). It contains only the 41 raw feature columns; no label column or test partition was read.

- Input: docs/examples/inference_validation_input.csv
- Current matching source-schema API output: docs/examples/inference_validation_output_source_schema.csv
- Current matching source-schema summary: docs/examples/inference_validation_summary_source_schema.json
- Earlier generic-fallback output and summary are retained as inference_validation_output.csv and inference_validation_summary.json for historical comparison; they are not the current recognized-source example.

The current 12-row example is recognized as the exact NSL-KDD source schema (41/41 features), uses the stored training-fitted NSL-KDD preprocessor, and reports compatibility 1.0 / High trust. It produces 7 Known attack, 4 Normal, and 1 Unfamiliar decision. No labels are included in the input or used for prediction. This is an engineering/API smoke example, not a research accuracy result. The generic unknown-schema fallback remains separate and does not promote features into source slots solely because names resemble source features.

## Validation

Focused API tests (13 passing in the recorded contract suite) cover label exclusion, renamed/shuffled/extra input columns, NaN/Inf/all-NaN columns, unknown categories, deterministic bounded calibration sampling, empty and malformed CSV handling, missing requested label columns, mixed non-numeric values, a 100,000-row streaming scan with a 512-row calibration cap, the 2 GiB free-RAM hard stop, and Stage-12 decision precedence (all four output states). The later focused regression suite passed 21 tests, including recognized-source-schema and metadata-declared numeric-column coercion checks. A 37-row full API synthetic stress run completed with finite outputs; predictions were bit-identical with and without the label column. The synthetic rows were deliberately unmatched and all routed `Unfamiliar`, so this is engineering robustness evidence, not a detection metric. Synthetic tests are engineering checks, not research metrics. GPU source-validation replay passed the required 1e-5 probability tolerance: maximum probability difference was 3.58e-7 for NSL-KDD and 2.98e-7 for UNSW-NB15. See C:\\NOVEL_IDS_runs\\phaseE_validation_repro_20261126\\inference_api_validation\\validation_parity_gpu.json. CPU replay differences were about 4.7e-4 due CPU/GPU numerical differences.

## 2026-11-27 API additions and interpretation

### Per-row outputs
Every row now includes:
- decision: Normal, Known attack, Uncertain, or Unfamiliar.
- best_guess_class and best_guess_probability: the top class and its probability even when the selective decision is Uncertain or Unfamiliar.
- Existing predicted_category, confidence, ood_score, message, source_branch, and per-class probabilities retain their existing meanings.

### Dataset-level compatibility and trust
The summary includes compatibility with exact/semantic/group-only/dataset-specific counts and a score:
(matched_exact + matched_semantic + 0.5 * group_only) / inspected_external_features.
Level-2 mappings are always group_only; they are never represented as feature equivalences.
Trust thresholds are configuration-backed in config/inference_trust_policy.yaml:
- High: compatibility >= 0.75
- Medium: compatibility >= 0.40 and < 0.75
- Low: compatibility < 0.40

When no features are matched, the system will generally mark rows Unfamiliar, while still returning best-guess class/probability and a Low trust level.

### Optional external feature metadata
Any dataset can supply an optional YAML metadata file through onboard_csv(..., feature_metadata_path=...) or the CLI. Example:
python -m novel_ids.inference.predict --input cic.csv --output cic_predictions.csv --summary cic_summary.json --feature-metadata config/cic_feature_metadata.yaml --normalization-mode option_a_target

The metadata may declare descriptions, units, network meanings, data types, semantic groups, explicit unit/type compatibility, and value transforms. Level 1 requires an approved source semantic identity plus unit/type compatibility and descriptor similarity. Level 2 requires a reviewed related group and descriptor similarity, and remains group-only. Level 3 remains dataset-specific.
The generic matcher keeps the 78 model feature slots separate from the 83 detailed source-training descriptors used for matching. The detailed registry carries approved review status; the 78-slot schema alone did not.

### Unseen normalization modes
- option_a_target (headline): per-dataset median/IQR fitted only on a bounded, unlabeled target calibration sample (maximum 50,000 rows).
- option_b_strict_source_level1: for Level-1 numeric features only, use equal-dataset-mean medians/IQRs from the saved source-training feature descriptors; non-Level-1 numeric features continue to use unlabeled target statistics. This is a strict source-statistics comparison, not fine-tuning.

CLI flags:
- --feature-metadata path.yaml
- --normalization-mode option_a_target|option_b_strict_source_level1

### Validation-tuned class-prior decision layer
The shipped inference path now applies the single validation-tuned additive class-bias vector in config/class_prior_bias.json to calibrated log-probabilities. The API verifies its canonical SHA-256 and taxonomy order before use. The vector was fitted only on source validation predictions using the pre-registered -2.0 to 2.0 grid, 0.25 step, three coordinate-ascent passes, seed 42, and 0.002 squared-bias penalty. The validation acceptance gate passed. This adjustment was designed after the primary model's test results were seen; any adjusted test result is post-hoc and must be disclosed. The original pre-registered test result remains the headline.

### Source-validation parity scope
scripts/validate_inference_api.py checks the API-loaded frozen model's internal source-validation forward path against saved Phase-E validation distributions. This is distinct from raw CSV generic unseen-data onboarding; do not describe it as proof that the generic predict_csv route reproduces source-validation predictions.

Streaming inference emits progress messages at approximately 60-second intervals for long CSV jobs.


## Important source-schema versus generic unseen-route distinction

The generic onboard_csv() / predict_csv() route remains the fallback for unknown schemas and does not promote features into frozen source slots solely because a name resembles a source feature. However, an exact normalized fingerprint of the complete recognized NSL-KDD (41 features) or UNSW-NB15 (42 features) schema selects that dataset's stored training-fitted preprocessor. The 2026-11-27 source-schema correction verified both exact routes at compatibility 1.0 / High trust and raw validation probability parity within 1e-5. Earlier six generic-route demo summaries with zero compatibility are superseded for these two recognized source schemas; they remain valid examples of the generic unknown-schema fallback.

Do not interpret those demo outputs as source-validation metrics or proof of source-schema compatibility. The saved GPU parity check uses the separate internal source-validation forward path; it is not the same as generic raw-CSV onboarding. See `demo_samples/README_DEMO_SAMPLES.md` and `docs/STATUS_VERIFIED.md`.


## 2026-11-27 source-schema and numeric-column corrections

- Source fingerprints are computed from normalized column-name sets (BOM removal, trim/collapse whitespace, case-folding, documented aliases). Recognition requires an exact match to the stored NSL-KDD (41 features) or UNSW-NB15 (42 features) preprocessor schema after excluding label/outcome columns and the dataset metadata column.
- Recognized source uploads use the stored source-training-fitted DatasetPreprocessor.transform path. Training medians, scalers, and categorical vocabularies are reused; upload rows do not refit normalization. Compatibility is 1.0 and trust is High.
- Unknown schemas retain semantic matching and unlabeled target-only Option A median/IQR normalization. The strict Option B path remains available for external metadata mappings.
- For unknown schemas with external metadata, columns declared numeric/continuous/integer/binary are coerced consistently in both each prediction chunk and the unlabeled calibration frame before target normalization. Non-numeric tokens become missing values and are masked; this prevents object-dtype calibration chunks from failing numeric normalization.
- Regression tests cover NSL-KDD/UNSW exact source recognition, unknown-schema fallback, label invariance, and metadata-declared numeric columns with object-dtype calibration values.
- 500-row raw source validation parity: NSL-KDD max probability difference 2.47e-7; UNSW-NB15 5.73e-7, each against saved Phase-E probabilities with the shipped class-prior bias applied. Both pass the 1e-5 tolerance.
- CIC second run: both Option A and Option B completed on the same 137,312-row seed-42 sample. Both produced 137,312 Unfamiliar decisions (0% accepted coverage). Option B completed on a resource-safe retry with batch_size=32 and chunk_size=2048 after its first attempt hit the 2-GiB free-RAM guard. Forced best-guess scores are diagnostics only; accepted CIC transfer was not demonstrated. See the combined CIC second-run report.
