# NOVEL-IDS verified status — 2026-11-27

## Overall

The owner-authorized continuation has been executed as far as the explicit resource guard permits. Source-format demo recognition, raw validation parity, the S0 runtime audit, saved Phase C reconciliation, both CIC second-run inference options, and documentation updates are complete. Option B completed on a user-authorized resource-safe retry after its first attempt hit the RAM guard. No source-test inference was rerun.

## Phase C integrity

- Recovery manifest SHA-256 matches the value recorded in results.json.
- 25/26 recovery manifest entries match; the sole mismatch is scripts/phaseC_recovery_execute.py (recorded a63b65de...; current 5b20ab90...).
- No exact-hash original evaluator copy was found. Original version not recovered. See docs/PHASEC_INTEGRITY_NOTE.md.
- All 24 saved prediction CSV hashes match. Protected Stage 5 snapshot, Stage 6 checkpoint, primary checkpoint, taxonomy, evaluation label map, calibration and thresholds match their recorded hashes.
- Saved results predate the current evaluator modification and remain hash-verified. Exact replay using the current evaluator is not guaranteed.
- Phase C saved test predictions/results remain the headline. No test inference, test feature reopening, or metric rerun occurred.

## S0 runtime audit

PASS — validation-only runtime audit saved to docs/pipeline_truth_check.json.

- One 64-row raw NSL-KDD validation batch passed through Stage 5–11.
- Tensors were finite; shared/residual/fused/transformer representations were 128-dimensional; classifier logits were 12-dimensional.
- Stage 7 gate G: shape 64x128, mean 0.362738, std 0.365859, min 0.00001087, max 0.999992.
- Stage 8 alignment C: shape 64x1, mean 0.407726, std 0.025548, min 0.375792, max 0.480753.
- Stage 5 and Stage 6 parameters were frozen. Stage 7, Stage 8, Stage 10 and Stage 11 parameters retain requires_grad=True but were in eval mode and executed under torch.no_grad(); no gradients or parameter updates occurred. Stage 9 has no parameters.
- No test partitions or test features were loaded.

## Source-format demo fix

PASS.

- NSL-KDD raw validation sample: 41/41 source features matched; compatibility 1.0; trust High; stored NSL-KDD source preprocessor used.
- UNSW-NB15 raw validation sample: 42/42 source features matched; compatibility 1.0; trust High; stored UNSW-NB15 source preprocessor used.
- Both pairs have identical labeled/unlabeled predictions. Source validation parity against saved validation probabilities, with the already-shipped class-prior bias applied in log-probability space: NSL maximum absolute difference 2.47e-7; UNSW 5.73e-7 (tolerance 1e-5).
- NSL 500-row labeled demo: accuracy 1.000, macro-F1 1.000; decisions Normal 261, Known attack 227, Unfamiliar 12.
- UNSW 500-row labeled demo: accuracy 0.792, macro-F1 0.5907; decisions Known attack 244, Normal 142, Uncertain 79, Unfamiliar 35.
- These are validation-derived demo samples, not test results.
- All six CSVs were verified as raw values with original feature names; no processed-artifact regeneration was needed. Original sample CSVs and manifest were preserved.
- CIC 500-row labeled/unlabeled demo: compatibility 0.465116, trust Medium, all 500 Unfamiliar; labeled/unlabeled predictions identical.

## CIC-DDoS2019 second run

Option A and Option B COMPLETE (Option B completed on a resource-safe retry).

- Sample: 137,312 rows, 86 features, seed 42, maximum 15,000 rows per raw label. Free disk was 16.88 GiB before starting, so the full approved sample was used.
- Mapping SHA-256 6cccab77a7892d469731955a519405a6eb1d4b370a3cacfbc7aa6a31f957eb59 matches the pre-inference PROGRESS.md value. Feature and label files match the prepared manifest hashes.
- Option A target-only unlabeled normalization completed; compatibility 0.465116 (4 exact, 72 group-only, 10 dataset-specific); trust Medium; decisions: 137,312 Unfamiliar (0 accepted coverage).
- Forced best-guess attack-vs-normal: accuracy 0.568377, attack precision 0.974329, recall 0.529392, F1 0.686034; confusion matrix [[13294,1706],[57561,64751]].
- 1-P(Normal): AUROC 0.771223, AUPR 0.966248 — “useful signal”.
- OOD score: AUROC 0.589223, AUPR 0.870222 — “no usable signal”.
- The first Option B attempt stopped when the API RAM guard observed free RAM below 2 GiB. A user-authorized continuation completed all 137,312 rows using batch_size=32 and chunk_size=2048; predictions and metrics are saved and hash-verified.
- The earlier 162,276-row CIC run remains preserved. No training, fine-tuning, threshold change, or source-test access occurred.
- Aggregate sample lacks original source-file identity; per-source-file rates are unavailable.

## Tests and API

- Focused suite before the numeric-coercion regression test: 18 passed.
- Focused suite after the fix: 21 passed in 9.73s.
- New regression covers metadata-declared numeric features whose unlabeled calibration values have object dtype/malformed tokens.
- Unknown-schema semantic matching and target-only normalization remain the fallback. Recognized source schemas use source-fitted preprocessing; no source preprocessor or fitted artifact was changed.
- No package installation, Git operation, or training occurred.

## Permanent deviations and limitations

1. Phase C recovery contains five ablations, not the requested sixth.
2. HGB/MLP baseline aggregate metrics/confusion matrices exist, but baseline per-row prediction/probability/OOD/decision artifacts do not.
3. No plain Transformer baseline checkpoint or test numbers exist.
4. Phase C evaluator hash note and unrecovered original version are documented in docs/PHASEC_INTEGRITY_NOTE.md.
5. CIC Option B completed after a resource-safe retry; both Option A and Option B have measured second-run results. Both still reject all CIC rows as Unfamiliar (0% accepted coverage).
6. Both CIC Option A and Option B selective decision rules reject every row as Unfamiliar; forced class scores must not be described as accepted operational detection performance.
7. Stage 6 qualification remains: the original pre-registered criterion result is NO_QUALIFYING_STAGE6_ARM; Candidate 09 / lambda_decorr=1.0 uses a revised criterion applied after results were seen on logical grounds. The residual has measurable detection information beyond random features, but the learned encoder does not exceed a random projection of E_d; the original rank gate fails and the L_decorr effect is not distinguishable from noise on one seed.
8. KDD Cup 99 sample files were not supplied. S4 unseen-dataset selection/few-shot adaptation remains intentionally skipped.
9. The CIC aggregate does not retain per-source-file identity, so per-file attack rates cannot be calculated.

## Protected scope

No Stage 5 source file, Stage 5 snapshot, Stage 6 checkpoint, primary checkpoint, taxonomy, calibration, thresholds, dataset, label, split, or fitted preprocessing artifact was changed. No closed test partition was accessed. No Git or package installation occurred.


S0 inventory detail: the parameter-free Stage 9 fusion module retained its module training flag (training_mode=True) during the no-grad validation pass; it has zero parameters. This is explicitly recorded in pipeline_truth_check.json.

## 2026-11-27 — CIC Option B completed after resource-safe continuation

- The earlier Option B pass stopped at the API 2 GiB RAM guard. Following the user's request to continue pending work, Option B was rerun once with smaller batch_size=32 and chunk_size=2048; the same prepared 137,312-row seed-42 sample, frozen model, mapping, metadata, and thresholds were retained.
- Option B completed 137,312/137,312 rows. Normalization scope: source-training statistics for Level-1 features; target-unlabeled statistics for non-Level-1 features. All decisions were Unfamiliar (0% accepted coverage).
- Option B forced best-guess metrics (not accepted operational metrics): accuracy 0.544432, attack precision 0.961951, recall 0.508683, F1 0.665465. 1-P(Normal) AUROC/AUPR 0.794376/0.967844; OOD AUROC/AUPR 0.547361/0.853284.
- Option A remains unchanged: 137,312 Unfamiliar; 0% accepted coverage. Option A forced accuracy/F1 0.568377/0.686034; 1-P(Normal) AUROC 0.771223; OOD AUROC 0.589223.
- The API RAM guard remained enabled throughout. The successful Option B run ended with 2.681 GiB free RAM and 21.903 GiB free disk. No training, fine-tuning, threshold change, source-test access, or protected artifact change.
- Combined report and hash verification: C:\NOVEL_IDS_runs\cic\second_run_20261127\CIC_DDoS2019_SECOND_RUN_REPORT.md and cic_second_run_final_report.json. Both prediction CSVs and the separate evaluation-label file contain 137,312 rows; feature/label/mapping/metadata hashes match the frozen manifest.
- Interpretation: 1-P(Normal) ranking improves under Option B, but forced F1 decreases and OOD AUROC remains below 0.60. Since all rows remain Unfamiliar, CIC zero-shot transfer is not demonstrated. Do not claim accepted detection performance.


## 2026-11-27 ? Dashboard operational verification completed

- Streamlit 1.67.0 was already present in the project virtual environment; no package installation was performed.
- Streamlit AppTest loaded the approved frozen checkpoint on CPU and reported ?Frozen model loaded and verified?; checkpoint, Stage 5 and Stage 6 hashes matched the loaded model info.
- The dashboard uploaded docs/examples/inference_validation_input.csv (12 raw NSL-KDD validation feature rows; no labels), completed inference for 12 rows, and produced finite numeric outputs with no app exceptions.
- Exact source schema recognized as NSL-KDD: 41 available features, compatibility 1.0, trust High. Decisions: 7 Known attack, 4 Normal, 1 Unfamiliar. This is an engineering smoke test on validation data, not a new research metric.
- Prediction CSV and summary JSON serialized successfully (3,393 and 41,623 bytes); both corresponding download controls are wired in app.py.
- Headless Streamlit server returned HTTP 200 for /_stcore/health and /; the test server was stopped after verification.
- No training, fine-tuning, threshold change, package install, source-test access, or protected artifact modification.

## 2026-11-27 ? Finish-today prompt re-audit and current API example

- Re-read the exact 189-line Pasted markdown(4).md finish-today prompt, Sections 0?6, and compared each requirement with current saved evidence.
- Re-ran the 12-row validation CSV through public load_model/onboard_csv/predict_csv on CPU after exact source-schema recognition was implemented. NSL-KDD 41/41 features, compatibility 1.0, High trust; 7 Known attack, 4 Normal, 1 Unfamiliar; all outputs finite.
- New paired artifacts: docs/examples/inference_validation_output_source_schema.csv (SHA-256 ccdb8f06694f8da918751ce5af6e1bad7b0168b7864e88a3d46fd3ef9969e932) and docs/examples/inference_validation_summary_source_schema.json (SHA-256 f697a430a42481c8e181df7c8e74ad4ed8d5b4d726bb59e6fb02ca7602752589). Existing generic-fallback output files were preserved.
- Updated docs/INFERENCE_API.md and README.md to distinguish current recognized-source inference from generic unseen-schema fallback and to remove stale CIC/S0 statements.
- Permanent Phase C omissions and S4/KDD Cup 99 absence remain documented; no closed test data was reopened, no training/threshold/protected artifact changes, no package install, and no Git operation.
