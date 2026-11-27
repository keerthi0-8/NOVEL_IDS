# NOVEL-IDS inference dashboard

## Scope

The root `app.py` is a multi-page, inference-only Streamlit frontend for the frozen NOVEL-IDS Stage 5–11 model and Stage 12 decision policy. It uses the public API in `src/novel_ids/inference/api.py`; it does not duplicate model inference or preprocessing.

Pages:
- **Dashboard** — latest-session metrics, decision donut, attack-category distribution and recent session history.
- **Traffic Analysis** — dataset/schema selection, CSV/XLSX upload, file metadata, preview, optional label selection and analysis launch.
- **Analysis Results** — Stage 12 decision counts, confidence and attack-probability display, category chart, optional ground-truth evaluation, confusion matrix, searchable/filterable/sortable/paginated table, downloads and a Back to Dashboard action for starting another analysis.
- **Prediction History** — session-local previous analyses and ability to reopen their results.

## Model and inference safety

- Loads the approved frozen checkpoint through `load_model()` on the first Analyze action when not already loaded; the loader verifies the protected Stage 5, Stage 6 and primary checkpoint hashes. The sidebar also allows preloading the model. The frontend wraps this loader in Streamlit's process-wide resource cache, so browser sessions/reruns reuse one verified model instead of repeatedly rebuilding it.
- CPU is the default; CUDA is optional.
- No training, fine-tuning, threshold fitting, source-test access or model artifact changes occur in the UI.
- The app preserves Stage 12 outcomes: Normal, Known attack, Uncertain and Unfamiliar. Uncertain and Unfamiliar remain rejection outcomes; best-guess classes are not shown as accepted categories.
- The confidence-review slider is a display-only review flag. It does not change frozen Stage 12 thresholds or prediction decisions.
- Ground-truth labels are excluded from onboarding and prediction. They are read again only after inference for optional evaluation. Metrics and confusion matrices are calculated only when labels map to the approved source taxonomy or supported binary labels; rejected rows are excluded and accepted coverage is shown.
- Attack probability is displayed as (1-P(\mathrm{Normal})), summed across the model's non-Normal taxonomy probabilities. Confidence is the model's calibrated maximum class probability; neither is accuracy.

## Input formats

- Upload limit is configured to 1 GB in .streamlit/config.toml. CSV uploads are copied from the upload buffer to temporary disk without creating an additional full-size bytes copy, then passed through the existing chunked onboarding and prediction API. CSV is recommended for very large files.
- XLSX is read from its first worksheet and converted to a temporary CSV for the existing CSV-streaming API. The workbook is not changed. Large XLSX files can require substantially more RAM during parsing/conversion; use CSV for files approaching 1 GB.
- Source datasets are recognized only when their normalized feature schema matches the saved NSL-KDD (41-feature) or UNSW-NB15 (42-feature) preprocessor contract. A manually selected source dataset that does not match is rejected before prediction.
- Auto-detect is recommended. Unknown schemas continue through the existing generic unseen-schema route.
- The selected label/outcome column is removed before onboarding and inference.
- Prediction history is session-local and retains the latest 20 analyses. It shares each analysis DataFrame with the active result instead of storing a second full copy.
- CPU inference defaults are tuned to model batch size 128 and CSV scan chunk size 8192 (both adjustable in the UI). These reduce per-batch and per-chunk overhead; they do not change model weights, outputs, calibration or Stage 12 thresholds.
- Analysis results display separate elapsed times for file copy/preview, onboarding scan and model inference so the dominant cost can be identified on the user's hardware.
- The default results table skips a full-data sort when showing the natural row order. Large CSV exports are prepared only when the user requests them, rather than being serialized on every Streamlit rerun.
- The CSV pipeline still performs a full onboarding/profile scan followed by a prediction scan. A true single-pass rewrite would require a larger backend API change and has not been made because it could alter calibration/profile semantics.

## Environment and launch

The dashboard dependencies are intentionally separate from the captured backend requirements.

PowerShell from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8765
```

Dashboard dependencies:
- Streamlit
- openpyxl (XLSX reading)
- Matplotlib (donut and confusion-matrix visualizations)

The frozen checkpoint default is:
`C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43`

## Verification

- Python syntax compilation: passed after the performance-focused frontend changes.
- Streamlit AppTest: dashboard renders with 0 exceptions; results-to-dashboard navigation passed in the prior UI verification.
- Current local Streamlit health endpoint: HTTP 200; one listener on port 8765.
- Upload configuration confirmed at 1024 MB for both maxUploadSize and maxMessageSize.
- A real 300 MB/1 GB upload benchmark was not run in this change; the updated batch/chunk defaults are performance tuning, not a measured runtime guarantee.
- Streamlit AppTest empty Dashboard: 0 exceptions; expected navigation and model-load controls rendered.
- Streamlit AppTest Traffic Analysis navigation: 0 exceptions; upload control rendered.
- Streamlit AppTest Results page: 0 exceptions with synthetic decision rows, charts, evaluation metrics, confusion matrix and download controls; a separate all-rejected fixture also rendered without errors.
- CSV conversion smoke test: passed (sample 3-row file retained columns and rows).
- XLSX conversion smoke test: passed (sample first worksheet converted to CSV with columns and rows retained).
- Approved UNSW taxonomy evaluation helper smoke test: passed on a synthetic 4-row fixture (accuracy/F1 1.0); NSL-KDD labels with trailing periods, external binary labels, and zero accepted coverage were also tested. These are code-path tests, not NOVEL-IDS performance metrics.
- Live Streamlit health and root endpoints: HTTP 200.
- The full frozen-model upload-to-prediction flow was not rerun during this frontend-only change to avoid duplicate model loading and unnecessary memory pressure. The previously recorded 12-row source-schema inference smoke test remains the backend integration evidence.

## Known limitations

- XLSX uses the first worksheet only.
- Prediction history is retained in the active Streamlit session, not persisted to a database across server restarts.
- PDF export is not included; complete/filtered CSV and JSON summary downloads are available.
- Evaluation is unavailable when labels are absent or cannot be mapped to the approved source taxonomy.
- The generic CIC-DDoS2019 evaluation remains negative/limited: the documented second run returned 0% accepted coverage. The dashboard does not change or conceal this research limitation.

## Upload configuration

The repository-level .streamlit/config.toml sets both server.maxUploadSize and server.maxMessageSize to 1024 MB. Restart Streamlit after changing this configuration. Hosting providers may impose their own request, memory or execution limits, independent of this app setting.
