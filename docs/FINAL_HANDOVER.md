# NOVEL-IDS final handover

## Delivered project components

- Frozen Stage 5–11 primary model and Stage 12 decision policy; no training changes in the finalization pass.
- One-shot Phase C source evaluation and archived prediction/metric artifacts.
- Phase D partial CIC-DDoS2019 inference and evaluation report.
- Public backend inference API and CLI for onboarding and chunked CSV prediction.
- Validation-data example input/output and summary under docs/examples/.
- Optional inference-only Streamlit dashboard in app.py.
- Backend/API documentation, decision record, progress log, final report, and requirements files.

## Final verification performed

- Protected Phase C freeze manifest: 26/26 hashes matched at the latest recorded audit.
- Phase C prediction files: 24/24 hashes matched at the latest recorded audit.
- Focused inference/onboarding/streaming/Stage12 tests re-run during finalization: 13 passed in 4.87 seconds.
- Python bytecode compilation passed for the dashboard and inference modules.
- No Stage 5/6 training, model changes, test partition reopening, threshold tuning, package installation, or Git operation occurred during finalization.

## Research findings that must remain explicit

- Stage 6 is accepted with qualifications; its learned encoder did not exceed a random projection of the residual target under the reported tests, the original rank gate failed, and the decorrelation effect was not distinguishable from noise on a single seed.
- LODO cross-dataset transfer was weak: NSL-KDD→UNSW-NB15 macro-F1 0.1312; UNSW-NB15→NSL-KDD macro-F1 0.2968.
- HGB outperformed the primary Transformer on the reported independent comparisons: NSL-KDD unified macro-F1 0.640996 vs 0.627273; UNSW-NB15 native macro-F1 0.548124 vs 0.460816.
- CIC-DDoS2019 remains a negative/limited transfer result. The initial partial run covered 13 available files and 162,276 rows; the second run evaluated 137,312 rows under both Option A and Option B. Both second-run options assigned every row Unfamiliar (0% accepted coverage). Forced best-guess scores are diagnostics only; accepted zero-shot transfer was not demonstrated.
- Phase C prompt omissions remain: five ablation variants rather than six additional variants, no baseline per-row prediction artifacts, and no explicit primary uncertain_rate / accuracy_on_accepted fields. No plain Transformer baseline artifact was found. These cannot be retroactively repaired because the one-shot test partitions are closed.
- The full test suite's known protected failure remains tests/test_stage5_training.py::test_trainer_finite_losses_and_gradients (182 passed, 1 failed in the recorded full-suite run). The Stage 5 code was not modified to force this synthetic gradient expectation to pass.

## Dashboard runtime note

app.py was operationally verified on 2026-11-27 using Streamlit 1.67.0 already present in the project environment. Streamlit AppTest loaded and hash-verified the frozen checkpoint on CPU, uploaded the 12-row raw NSL-KDD validation example, and completed finite-output inference with no app exceptions. The source schema was recognized at 1.0 compatibility / High trust; decisions were 7 Known attack, 4 Normal, 1 Unfamiliar. Prediction CSV and summary JSON payloads serialized, both download controls are wired, and a headless Streamlit server returned HTTP 200 for the health and app endpoints. No package was installed. A fresh environment can install the optional UI dependency with:

    .\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt

Then follow docs/DASHBOARD.md. The separate NOVEL_IDS_V2 dashboard uses a different model pipeline and should not be presented as the current frozen model UI.

## Overall status

The trained model, backend API, two-option CIC second-run evaluation, documentation, and inference dashboard are delivered. The dashboard passed AppTest upload/inference and headless HTTP verification; the browser download click path was not manually automated, though both export controls and non-empty serialized payloads were verified. The research goal of demonstrated cross-dataset generalization is not achieved by the current CIC-DDoS2019 result. The Phase C omissions remain final protocol deviations, not unfinished work that can safely be rerun.

## 2026-11-27 ? Complete-prompt re-audit addendum

The original 189-line finish-today prompt (Sections 0?6) was re-read and checked against the current artifacts. The remaining Phase C omissions are still permanent protocol deviations because the one-shot source-test partitions are closed: five ablations rather than six, no baseline per-row prediction/probability/OOD/decision files, no plain Transformer baseline artifact, and no explicit primary uncertain-rate / accepted-accuracy fields. They were not recreated from test data.

The 12-row validation API example was refreshed through the public API using the recognized exact NSL-KDD schema and CPU inference. The new source-schema output/summary files are linked in docs/INFERENCE_API.md; 41/41 features recognized, compatibility 1.0 / High trust, decisions 7 Known attack, 4 Normal, 1 Unfamiliar. The older generic-fallback example files are preserved and explicitly identified as historical fallback output. This is engineering verification, not a research metric.

The latest focused regression suite remains 21 passed. Current resource check: 4.902 GB free RAM and 21.898 GB free disk; no Streamlit test server or Python/Streamlit process remained active. No test partition was reopened, and no training, threshold change, package install, Git operation, or protected artifact modification occurred.
