from pathlib import Path
import shutil

root = Path(r"C:\Users\VaishnaviP\OneDrive\Desktop\Documents\NOVEL_IDS_Project")
app_path = root / "app.py"
api_path = root / "src" / "novel_ids" / "inference" / "api.py"

app = app_path.read_text(encoding="utf-8")
api = api_path.read_text(encoding="utf-8")

# App: add a size + available-RAM aware chunk selector.
needle = 'def run_analysis(uploaded, extension: str, dataset_choice: str, label_column: str | None,\n                 batch_size: int, chunk_size: int) -> None:\n'
replacement = '''def choose_csv_chunk_size(file_size_bytes: int, free_ram_gb: float) -> int:
    """Choose a conservative CSV chunk from upload size and currently free RAM."""
    size_mb = max(0, int(file_size_bytes)) / (1024 * 1024)
    if size_mb <= 50:
        size_based = 8192
    elif size_mb <= 200:
        size_based = 4096
    elif size_mb <= 500:
        size_based = 2048
    elif size_mb <= 1024:
        size_based = 1024
    else:
        size_based = 512
    if free_ram_gb >= 6:
        ram_cap = 8192
    elif free_ram_gb >= 4:
        ram_cap = 4096
    elif free_ram_gb >= 3:
        ram_cap = 2048
    elif free_ram_gb >= 2:
        ram_cap = 1024
    else:
        ram_cap = 512  # inference will still enforce the 2 GiB hard stop
    return min(size_based, ram_cap)


def run_analysis(uploaded, extension: str, dataset_choice: str, label_column: str | None,
                 batch_size: int, chunk_size: int | str) -> None:
'''
if needle not in app:
    raise SystemExit("Could not find run_analysis signature; no app changes written.")
app = app.replace(needle, replacement, 1)

needle = '''    if uploaded is None:
        st.error("Choose a CSV or XLSX file first.")
        return
    started = datetime.now().astimezone()
'''
replacement = '''    if uploaded is None:
        st.error("Choose a CSV or XLSX file first.")
        return
    if chunk_size == "auto":
        free_ram_gb = float(model.resource_module.free_ram_gb())
        effective_chunk_size = choose_csv_chunk_size(int(uploaded.size), free_ram_gb)
        st.info(
            f"Automatic streaming selected {effective_chunk_size:,} rows per CSV chunk "
            f"for this {uploaded.size / (1024 * 1024):.1f} MB file "
            f"({free_ram_gb:.1f} GiB RAM currently free)."
        )
    else:
        effective_chunk_size = int(chunk_size)
    started = datetime.now().astimezone()
'''
if needle not in app:
    raise SystemExit("Could not find upload guard; no app changes written.")
app = app.replace(needle, replacement, 1)
app = app.replace('chunksize=int(chunk_size), calibration_rows=50000,',
                  'chunksize=effective_chunk_size, calibration_rows=50000,', 1)
app = app.replace('batch_size=int(batch_size), chunk_size=int(chunk_size),',
                  'batch_size=int(batch_size), chunk_size=effective_chunk_size,', 1)
app = app.replace('''                def update_inference_progress(done: int, total: int) -> None:
                    fraction = min(1.0, done / max(total, 1))
                    progress_bar.progress(fraction, text=f"Running frozen inference: {done:,} / {total:,} rows ({fraction:.0%})")
                    progress_text.caption("Predictions are being generated in bounded chunks. Keep this tab open until completion.")
''', '''                def update_inference_progress(done: int, total: int) -> None:
                    fraction = min(1.0, done / max(total, 1))
                    elapsed = max(time.perf_counter() - phase_started, 0.001)
                    remaining = max(0.0, elapsed * (max(total - done, 0) / max(done, 1)))
                    progress_bar.progress(
                        fraction,
                        text=f"Running frozen inference: {done:,} / {total:,} rows ({fraction:.0%})",
                    )
                    progress_text.caption(
                        f"Elapsed {elapsed / 60:.1f} min · estimated remaining {remaining / 60:.1f} min · "
                        f"chunk {effective_chunk_size:,} · batch {batch_size}"
                    )
''', 1)
needle = '''        chunk_size = st.select_slider("CSV scan chunk size", options=[1024, 2048, 4096, 8192], value=4096,
                                      help="A 4,096-row chunk reduces repeated CSV and preprocessing overhead while keeping model batches independently bounded.")
        st.caption("Balanced defaults: batch 32 and CSV chunk 4,096. The inference API retains its 2 GiB free-RAM hard stop.")
'''
replacement = '''        chunk_choice = st.selectbox(
            "CSV scan chunk size",
            options=["auto", 512, 1024, 2048, 4096, 8192, 16384],
            index=0,
            format_func=lambda value: "Automatic (recommended)" if value == "auto" else f"{value:,} rows",
            help="Automatic mode considers uploaded file size and currently free RAM. Manual values are available for controlled testing.",
        )
        st.caption("Automatic mode reduces chunk size as files grow or free RAM falls. Model batch size stays independently bounded; the 2 GiB RAM hard stop remains active.")
'''
if needle not in app:
    raise SystemExit("Could not find inference settings block; no app changes written.")
app = app.replace(needle, replacement, 1)
app = app.replace('run_analysis(uploaded, extension, dataset, label_column, batch_size, chunk_size)',
                  'run_analysis(uploaded, extension, dataset, label_column, batch_size, chunk_choice)', 1)

# API: spool prediction chunks to disk rather than retaining every chunk and then
# allocating a second full copy during pd.concat. The returned DataFrame API stays intact.
if "import tempfile\n" not in api:
    api = api.replace("import sys\n", "import sys\nimport tempfile\n", 1)
if "import shutil\n" not in api:
    api = api.replace("import re\n", "import re\nimport shutil\n", 1)

needle = '''    outputs, branch_counts = [], {}
    offset = 0
    progress_started = time.monotonic()
    progress_last = progress_started
    try:
        for chunk in pd.read_csv(input_path, chunksize=chunk_size):
'''
replacement = '''    outputs_path = None
    branch_counts = {}
    offset = 0
    progress_started = time.monotonic()
    progress_last = progress_started
    # Prediction output is much wider than the input. Spool each completed
    # chunk to disk so a large run does not retain all output chunks in RAM.
    disk = shutil.disk_usage(input_path.parent)
    if disk.free < 10 * 1024**3:
        raise RuntimeError("HARD STOP: free disk space below 10 GiB before streaming prediction output.")
    output_temp = tempfile.TemporaryDirectory(prefix="novel_ids_prediction_spool_")
    outputs_path = Path(output_temp.name) / "predictions.csv"
    wrote_output_header = False
    try:
        for chunk in pd.read_csv(input_path, chunksize=chunk_size):
'''
if needle not in api:
    raise SystemExit("Could not find prediction accumulation block; no API changes written.")
api = api.replace(needle, replacement, 1)
needle = '''            batch_output, branches = _predict_input_batch(model, inputs, batch_size, offset)
            outputs.append(batch_output)
            for name, count in pd.Series(branches).value_counts().items():
'''
replacement = '''            batch_output, branches = _predict_input_batch(model, inputs, batch_size, offset)
            batch_output.to_csv(
                outputs_path,
                mode="a" if wrote_output_header else "w",
                header=not wrote_output_header,
                index=False,
            )
            wrote_output_header = True
            del batch_output, inputs, feature_frame, chunk
            gc.collect()
            for name, count in pd.Series(branches).value_counts().items():
'''
if needle not in api:
    raise SystemExit("Could not find output append block; no API changes written.")
api = api.replace(needle, replacement, 1)
needle = '''    except pd.errors.EmptyDataError as exc:
        raise ValueError("Input CSV is empty.") from exc
    if offset != report.rows or not outputs:
        raise RuntimeError(f"Streaming inference processed {offset} rows; onboarding recorded {report.rows}.")
    output = pd.concat(outputs, ignore_index=True)
'''
replacement = '''    except pd.errors.EmptyDataError as exc:
        output_temp.cleanup()
        raise ValueError("Input CSV is empty.") from exc
    except Exception:
        output_temp.cleanup()
        raise
    if offset != report.rows or not wrote_output_header:
        output_temp.cleanup()
        raise RuntimeError(f"Streaming inference processed {offset} rows; onboarding recorded {report.rows}.")
    if model.resource_module.free_ram_gb() < 2.0:
        output_temp.cleanup()
        raise RuntimeError("HARD STOP: free RAM below 2 GiB before loading prediction results.")
    try:
        output = pd.read_csv(outputs_path)
    finally:
        output_temp.cleanup()
'''
if needle not in api:
    raise SystemExit("Could not find concat/finalization block; no API changes written.")
api = api.replace(needle, replacement, 1)

# Record the effective chunk in the returned summary (existing field).
app_path.write_text(app, encoding="utf-8")
api_path.write_text(api, encoding="utf-8")
print("PATCHED app.py and inference/api.py")
