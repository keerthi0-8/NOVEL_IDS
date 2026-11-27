# NOVEL-IDS

A cross-dataset network intrusion detection research implementation using UNSW-NB15 and NSL-KDD, with frozen Stage 5 shared encoder, Stage 6 residual branch, Stage 7 adaptive gate, Stage 8 selective alignment, Stage 9 unified representation, Stage 10 Transformer and Stage 11 unified classification/OOD decision logic.

## Current research status

**Source-test evaluation and partial CIC-DDoS2019 evaluation are complete. The inference API is implemented and its source-validation GPU parity check passes. Cross-dataset generalization is not established.**

- Phase A/B: primary seed-43 model and validation ablations completed; common-only comparison used the reduced 1,000-step, batch-64, two-seed budget.
- Phase C: one frozen source-test evaluation completed for the primary model, five validation-selected ablations, HGB and MLP baselines. NSL-KDD primary independent accuracy/macro-F1: 0.789671/0.627273; UNSW-NB15: 0.761692/0.460816. HGB exceeds the primary model on both independent comparisons (NSL unified HGB macro-F1 0.640996; UNSW native HGB macro-F1 0.548124). NSL family-level and native label-space results are not directly comparable. Test partitions are closed. Phase C audit: baseline per-row prediction files were not saved; primary metrics omit explicit uncertain-rate and accepted-accuracy fields; only five ablation variants (six configurations including primary) are in the frozen result. These cannot be repaired after the one-shot closure.
- Phase D: CIC-DDoS2019 was only partially available (13 CSVs). The initial 162,276-row run and the later 137,312-row second run are documented separately. Both Option A (target-only unlabeled normalization) and Option B (source statistics for Level-1 features, target-unlabeled statistics for non-Level-1 features) on the second sample assigned every row Unfamiliar (0% accepted coverage). Forced best-guess scores are diagnostic only; accepted classification metrics are undefined. Missing exact filenames in the initial inventory: NetBIOS.csv, TFTP.csv, and UDP.csv. This is a negative/limited transfer result, not successful zero-shot detection.
- Inference API: chunked onboarding and prediction, bounded unlabeled calibration sample, CPU fallback, CLI, model metadata and robustness handling implemented. GPU source-validation parity passes <=1e-5. The raw validation example pair is included under `docs/examples/`.
- Last recorded full test suite: 182 passed, 1 failed. The sole failure is a protected Stage 5 synthetic test expecting an unknown_category_embedding gradient on a fixture that does not exercise it. The later focused regression suite passed 21 tests; Stage 5 was not modified.

See `docs/FINAL_REPORT_2026-11-26.md`, `docs/DECISION_RECORD.md`, and `docs/INFERENCE_API.md`.

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
- Official source-test evaluation completed once from the frozen prediction files; `TEST PARTITIONS CLOSED ? 2026-11-26`. Do not reopen or rerun the test partitions.
- CIC-DDoS2019 evaluation used only the available files and the frozen, label-blind onboarding map. The initial partial run covered 162,276 rows; the second run covered 137,312 rows under both Option A and Option B. Both second-run options returned all Unfamiliar (0% accepted coverage). No training, fine-tuning, threshold selection, or use of unseen labels for prediction occurred.
- Test and unseen labels are never used for training, preprocessing fitting, model selection or threshold tuning.

## Validation-only reproduction command (PowerShell)

From the project root:

    $env:PYTHONPATH = "$PWD\src;$PWD\scripts"
    .\.venv\Scripts\python.exe scripts\phaseE_validation_repro.py --run-dir C:\NOVEL_IDS_runs\phaseE_validation_repro_new --checkpoint C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43\best.pt

The output directory must be new. This is validation-only and does not access test/unseen partitions.

## One-shot and unseen status

The Phase C source-test evaluation was completed once under the frozen recovery protocol. The prediction CSVs and authoritative `results.json` are archived under `C:\NOVEL_IDS_runs\phaseC_recovery_20261126`; `PROGRESS.md` records `TEST PARTITIONS CLOSED ? 2026-11-26`. Do not rerun, tune, or reopen those test partitions.

CIC-DDoS2019 partial evaluation is complete for the available files. Both second-run normalization options processed 137,312 rows and assigned all rows Unfamiliar (0% accepted coverage). This does not demonstrate successful zero-shot transfer.

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

- Cross-dataset validation transfer is weak: NSL-KDD?UNSW-NB15 macro-F1 0.1312 and UNSW-NB15?NSL-KDD 0.2968 on shared classes.
- CIC-DDoS2019 second-run Option A and Option B both had 0% accepted coverage (all 137,312 sampled rows were Unfamiliar); no successful zero-shot transfer is demonstrated.
- NOVEL-IDS does not outperform the HGB baselines on the reported independent source comparisons.
- Stage 6 is accepted only with qualifications: measurable residual detection information beyond random features, but the learned encoder does not exceed a random projection of E_d; the original rank gate fails; L_decorr's effect is not distinguishable from noise on a single seed.
- The full test suite retains one protected Stage 5 synthetic gradient-expectation failure. No Stage 5 source change is authorized or made.
- The generic unseen route and its source-branch selection policy require further empirical validation; the CIC result is negative.


## Inference API

See `docs/INFERENCE_API.md`. Load the frozen model and predict a new CSV:

```powershell
python -m novel_ids.inference.predict --input new_traffic.csv --output predictions.csv --summary summary.json
```

The inference API supports CPU fallback and batched prediction. For a labeled evaluation-only CSV, pass `--labels-column Label`; labels are excluded before inference.


## CIC-DDoS2019 zero-shot evaluation

The completed partial-coverage evaluation report and artifact inventory are documented at C:\NOVEL_IDS_runs\cic\second_run_20261127\CIC_DDoS2019_SECOND_RUN_REPORT.md. Both Option A and Option B processed 137,312 rows; every row was Unfamiliar (0% accepted coverage). Forced best-guess metrics are diagnostics only, not accepted operational detections.


 
## Inference API (2026-11-27)

Activate the existing environment from the project root:
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

Focused API tests:
python -m pytest -q tests\test_inference_onboarding.py tests\test_inference_streaming.py tests\test_stage12_decision.py

Full test suite:
python -m pytest -q

Streaming inference:
python -m novel_ids.inference.predict --input new_traffic.csv --output predictions.csv --summary predictions_summary.json --labels-column Label --batch-size 256 --chunk-size 8192

Optional separate unlabeled calibration file:
python -m novel_ids.inference.predict --input cic_evaluation.csv --normalization-input cic_unlabeled_calibration.csv --output cic_predictions.csv --summary cic_summary.json --labels-column " Label"

Source-validation parity replay:
python scripts\validate_inference_api.py

The API scans CSVs in 8,192-row chunks, fits Option A median/IQR using a deterministic unlabeled sample capped at 50,000 rows, and performs model inference in batches of 256 by default. CUDA is preferred; CPU can be forced with load_model(..., device="cpu"). Free RAM below 2 GiB or GPU allocation above 6 GiB triggers a hard stop.

Hardware used: Intel i7-13700HX, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM. The final prediction DataFrame is retained in memory by the API return contract even though CSV scanning and tensor inference are chunked.

Known limitation: the full suite has one existing Stage 5 synthetic gradient-expectation failure. Stage 5 is protected and unchanged. The raw validation input/output example pair is now provided in `docs/examples/` and documented in `docs/INFERENCE_API.md`.


Validation-data API example: `docs/examples/inference_validation_input.csv` and matching `docs/examples/inference_validation_output.csv`; see `docs/INFERENCE_API.md`.


## Inference dashboard

The project includes an optional inference-only dashboard in app.py, connected to the frozen Stage 5–11 inference API (not the separate legacy NOVEL_IDS_V2 demo pipeline). It supports CSV upload, optional label-column exclusion, optional unlabeled calibration CSV, streaming onboarding/inference, Stage 12 decision summaries, feature compatibility details, model provenance, and CSV/JSON downloads.

The dashboard dependency is intentionally separate from the captured backend environment. See docs/DASHBOARD.md; install requirements-dashboard.txt only when the UI is needed. Launch from the project root:

    $env:PYTHONPATH = "$PWD\src"
    .\.venv\Scripts\python.exe -m streamlit run app.py

CPU is the default for conservative resource use; CUDA is optional. The dashboard is inference-only and does not reopen the closed source-test partitions. Runtime verification on 2026-11-27 passed: frozen checkpoint hash checks, a 12-row raw NSL-KDD validation CSV upload/inference smoke test, finite outputs, and HTTP 200 from the headless Streamlit app. See docs/DASHBOARD.md for the exact evidence and limitations.


## 2026-11-27 continuation audit

The current saved-artifact evaluation is in `docs/EVALUATION_REPORT.md`; the older `docs/EVALUATION_REPORT.txt` is marked non-authoritative because its row counts are stale. The Markdown report was rebuilt only from saved Phase C prediction CSVs and `results.json`; no source-test inference was rerun.

The six demo sample CSVs (500 rows each) pass their manifest SHA-256 and row-count checks. Their saved generic unseen-route outputs are all Unfamiliar with zero compatibility and Low trust, including the NSL-KDD/UNSW validation samples. This is not source-validation performance; see `demo_samples/README_DEMO_SAMPLES.md`.

S0 validation-only runtime audit passed on 64 raw NSL-KDD validation rows. The recovery evaluator hash mismatch is documented separately; the original evaluator was not recovered, but the approved saved results remain hash-verified and no source-test replay was performed.
