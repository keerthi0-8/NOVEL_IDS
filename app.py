from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from novel_ids.inference import load_model, onboard_csv, predict_csv


@st.cache_resource(show_spinner=False)
def get_frozen_model(checkpoint: str, device: str):
    """Share one verified frozen model across Streamlit sessions and reruns."""
    return load_model(checkpoint, device=device)


DEFAULT_CHECKPOINT = os.environ.get(
    "NOVEL_IDS_CHECKPOINT",
    r"C:\NOVEL_IDS_runs\stage7_11\coral_screen_lambda0p05_seed43",
)
PAGES = ["Dashboard", "Traffic Analysis", "Analysis Results", "Prediction History"]
DECISIONS = ["Normal", "Known attack", "Uncertain", "Unfamiliar"]
COLORS = {
    "Normal": "#27c79a",
    "Known attack": "#ff5d73",
    "Uncertain": "#f5b942",
    "Unfamiliar": "#8b9bb4",
}

st.set_page_config(
    page_title="NOVEL-IDS | Security Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --bg:#080f1c; --panel:#101c2d; --panel2:#142238; --line:#263951;
            --muted:#91a4bd; --text:#e8f0fa; --cyan:#42c7e8; }
    .stApp { background: radial-gradient(circle at 75% -10%, #142c46 0, #080f1c 42%); color:var(--text); }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.02em; }
    h1 { font-size:1.85rem !important; }
    h2 { font-size:1.25rem !important; }
    h3 { font-size:1.02rem !important; }
    [data-testid="stSidebar"] { background:#0b1626; border-right:1px solid #20334a; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b6c6d9; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,#14243a,#101b2c);
      border:1px solid #263b55; padding:14px 16px; border-radius:13px; }
    div[data-testid="stMetricLabel"] { color:#9eb1c8; font-size:.82rem; }
    div[data-testid="stMetricValue"] { color:#edf5ff; font-family:'Space Grotesk',sans-serif; }
    div[data-testid="stMetricDelta"] { font-size:.75rem; }
    .hero { padding:18px 20px; border:1px solid #29435f; border-radius:16px;
      background:linear-gradient(110deg,rgba(27,58,88,.78),rgba(15,29,47,.92)); margin-bottom:18px; }
    .hero-kicker { color:#62d3ed; text-transform:uppercase; letter-spacing:.14em; font-size:.7rem; font-weight:700; }
    .hero-title { color:#f2f7ff; font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; margin-top:5px; }
    .hero-copy { color:#a9bbcf; font-size:.9rem; margin-top:4px; }
    .section-note { color:#8ea4bd; font-size:.82rem; }
    .status-pill { display:inline-block; padding:4px 9px; border-radius:20px; font-size:.72rem;
      border:1px solid #31516d; color:#7ddcf0; background:#10283b; }
    .stButton>button[kind="primary"] { background:linear-gradient(90deg,#087fa8,#16a6c8); border:0; color:white; font-weight:700; }
    .stButton>button { border-radius:9px; }
    [data-testid="stDataFrame"] { border:1px solid #263b55; border-radius:10px; overflow:hidden; }
    div[data-testid="stExpander"] { border:1px solid #263b55; border-radius:10px; }
    hr { border-color:#263b55 !important; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "page": "Dashboard",
        "novel_ids_model": None,
        "novel_ids_model_info": None,
        "novel_ids_load_error": None,
        "novel_ids_predictions": None,
        "novel_ids_summary": None,
        "novel_ids_history": [],
        "novel_ids_analysis_meta": None,
        "novel_ids_filtered_export": None,
        "novel_ids_filtered_export_signature": None,
        "novel_ids_complete_export": None,
        "review_threshold": 0.50,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">Network intelligence / NOVEL-IDS</div>'
        f'<div class="hero-title">{title}</div><div class="hero-copy">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def safe_int(mapping: dict, key: str) -> int:
    try:
        return int(mapping.get(key, 0))
    except (TypeError, ValueError):
        return 0


def decision_counts(summary: dict) -> dict[str, int]:
    raw = summary.get("decision_counts", {})
    return {name: safe_int(raw, name) for name in DECISIONS}


def make_donut(counts: dict[str, int], title: str) -> None:
    labels = [x for x in DECISIONS if counts.get(x, 0) > 0]
    values = [counts[x] for x in labels]
    if not values or sum(values) == 0:
        st.info("No analyzed traffic is available yet.")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=120)
    fig.patch.set_facecolor("#101c2d")
    ax.set_facecolor("#101c2d")
    colors = [COLORS[x] for x in labels]
    wedges, _ = ax.pie(values, startangle=90, counterclock=False, colors=colors,
                       wedgeprops={"width": 0.34, "edgecolor": "#101c2d", "linewidth": 2})
    ax.text(0, 0.06, f"{sum(values):,}", ha="center", va="center",
            color="#f1f6fc", fontsize=18, fontweight="bold")
    ax.text(0, -0.18, "records", ha="center", va="center", color="#91a4bd", fontsize=8)
    legend = ax.legend(wedges, [f"{x}  ·  {counts[x]:,}" for x in labels],
                       loc="center left", bbox_to_anchor=(1.0, .5), frameon=False,
                       labelcolor="#c6d4e4", fontsize=8)
    for text_item in legend.get_texts():
        text_item.set_color("#c6d4e4")
    ax.set_title(title, color="#e8f0fa", fontsize=10, loc="left", pad=12)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def make_confusion(cm: list[list[int]], labels: list[str]) -> None:
    arr = np.asarray(cm, dtype=int)
    if arr.size == 0:
        st.info("Confusion matrix is unavailable.")
        return
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * .7), max(3.8, len(labels) * .55)), dpi=120)
    fig.patch.set_facecolor("#101c2d")
    ax.set_facecolor("#101c2d")
    image = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", color="#c6d4e4", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, color="#c6d4e4", fontsize=8)
    ax.set_xlabel("Predicted", color="#9eb1c8")
    ax.set_ylabel("Actual", color="#9eb1c8")
    ax.set_title("Confusion matrix · accepted predictions only", color="#e8f0fa", loc="left")
    threshold = arr.max() / 2 if arr.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:,}", ha="center", va="center",
                    color="#081321" if arr[i, j] > threshold else "#e8f0fa", fontsize=8)
    ax.spines[:].set_visible(False)
    fig.colorbar(image, ax=ax, fraction=.04, pad=.04)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)




def prepare_input(uploaded, extension: str, temp_dir: str) -> tuple[Path, pd.DataFrame, list[str]]:
    if not getattr(uploaded, "size", 0):
        raise ValueError("The uploaded file is empty.")
    input_path = Path(temp_dir) / "uploaded_traffic.csv"
    if extension == "csv":
        # Copy the in-memory upload buffer directly to disk without creating
        # another 1 GB Python bytes object. The inference API then streams CSV.
        uploaded.seek(0)
        with input_path.open("wb") as output_file:
            output_file.write(uploaded.getbuffer())
        preview = pd.read_csv(input_path, nrows=12)
        if preview.empty and len(preview.columns) == 0:
            raise ValueError("The CSV has no readable columns.")
    else:
        uploaded.seek(0)
        preview = pd.read_excel(uploaded, engine="openpyxl", nrows=12)
        if len(preview.columns) == 0:
            raise ValueError("The Excel worksheet has no columns.")
        # XLSX parsing/conversion is inherently memory-intensive. The first
        # worksheet is converted for the existing CSV-streaming inference API.
        uploaded.seek(0)
        full_frame = pd.read_excel(uploaded, engine="openpyxl")
        full_frame.to_csv(input_path, index=False)
        preview = full_frame.head(12)
    if preview.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported. Rename duplicates and upload again.")
    columns = [str(x) for x in preview.columns]
    return input_path, preview, columns


def choose_csv_chunk_size(file_size_bytes: int, free_ram_gb: float) -> int:
    """Prefer 4K/8K CSV chunks; lower only when free RAM requires it."""
    # The uploader currently caps files at 1 GiB, so file size alone should
    # not force tiny chunks. The model's 2 GiB hard stop remains authoritative.
    if free_ram_gb >= 5.0:
        return 8192
    if free_ram_gb >= 3.0:
        return 4096
    if free_ram_gb >= 2.0:
        return 2048
    return 1024  # inference will still stop below the protected 2 GiB floor


def run_analysis(uploaded, extension: str, dataset_choice: str, label_column: str | None,
                 batch_size: int, chunk_size: int | str) -> None:
    model = st.session_state.get("novel_ids_model")
    if model is None:
        preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            with st.spinner(f"Loading and verifying the approved frozen model on {preferred_device.upper()}…"):
                model = get_frozen_model(str(Path(DEFAULT_CHECKPOINT).expanduser()), preferred_device)
                st.session_state["novel_ids_model"] = model
                st.session_state["novel_ids_model_info"] = model.model_info()
                st.session_state["novel_ids_load_error"] = None
        except Exception as exc:
            st.error(f"Frozen model could not be loaded on {preferred_device.upper()} ({type(exc).__name__}): {exc}")
            if preferred_device == "cuda":
                st.warning("GPU loading failed. Select CPU in Model Status and load again; no checkpoint or model weights were changed.")
            return
    if uploaded is None:
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
    run_started = time.perf_counter()
    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="novel_ids_frontend_") as temp_dir:
        try:
            with st.spinner("Validating the file, onboarding features and running frozen NOVEL-IDS inference…"):
                phase_started = time.perf_counter()
                input_path, preview, columns = prepare_input(uploaded, extension, temp_dir)
                timings["file_copy_and_preview"] = time.perf_counter() - phase_started
                if len(columns) == 0:
                    raise ValueError("No feature columns were found.")
                if label_column is not None and label_column not in columns:
                    raise ValueError(f"Selected label column {label_column!r} is not present.")
                phase_started = time.perf_counter()
                onboarding = onboard_csv(
                    input_path, model=model, label_column=label_column,
                    chunksize=effective_chunk_size, calibration_rows=50000,
                )
                timings["onboarding_scan"] = time.perf_counter() - phase_started
                detected = onboarding.source_dataset
                requested = {
                    "UNSW-NB15": "unsw_nb15",
                    "NSL-KDD": "nsl_kdd",
                }.get(dataset_choice)
                if requested and detected != requested:
                    raise ValueError(
                        f"The selected dataset is {dataset_choice}, but the uploaded feature schema "
                        f"was recognized as {detected or 'an external/unseen schema'}. "
                        "Choose Auto-detect or upload the matching source schema. No predictions were run."
                    )
                phase_started = time.perf_counter()
                progress_bar = st.progress(0, text=f"Running frozen inference: 0 / {onboarding.rows:,} rows")
                progress_text = st.empty()
                def update_inference_progress(done: int, total: int) -> None:
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
                predictions, summary = predict_csv(
                    input_path, model, onboarding=onboarding, label_column=label_column,
                    batch_size=int(batch_size), chunk_size=effective_chunk_size,
                    progress_callback=update_inference_progress,
                )
                progress_bar.empty()
                progress_text.empty()
                timings["model_inference"] = time.perf_counter() - phase_started
                meta = {
                    "dataset": detected or "External / unseen schema",
                    "dataset_choice": dataset_choice,
                    "filename": uploaded.name,
                    "file_size": int(uploaded.size),
                    "timestamp": started.isoformat(timespec="seconds"),
                    "rows": int(len(predictions)),
                    "columns": int(len(columns)),
                    "label_column": label_column,
                    "timings_seconds": {**{k: round(v, 2) for k, v in timings.items()},
                                        "total_before_results": round(time.perf_counter() - run_started, 2)},
                }
                st.session_state["novel_ids_predictions"] = predictions
                st.session_state["novel_ids_summary"] = summary
                st.session_state["novel_ids_analysis_meta"] = meta
                st.session_state["novel_ids_filtered_export"] = None
                st.session_state["novel_ids_filtered_export_signature"] = None
                st.session_state["novel_ids_complete_export"] = None
                history_item = {
                    "meta": meta, "summary": summary,
                    # Keep one in-memory prediction frame; history references it instead of duplicating it.
                    "predictions": predictions,
                }
                st.session_state["novel_ids_history"] = (
                    [history_item] + st.session_state.get("novel_ids_history", [])
                )[:20]
                st.session_state["page"] = "Analysis Results"
        except Exception as exc:
            st.error(f"Analysis failed ({type(exc).__name__}): {exc}")
            return
    st.success(f"Analysis completed for {len(st.session_state['novel_ids_predictions']):,} records.")
    st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🛡️ NOVEL-IDS")
        st.caption("Network Intrusion Detection System")
        st.markdown('<span class="status-pill">● INFERENCE CONSOLE</span>', unsafe_allow_html=True)
        st.divider()
        current = st.session_state.get("page", "Dashboard")
        selected = st.radio("WORKSPACE", PAGES, index=PAGES.index(current), label_visibility="visible")
        if selected != current:
            st.session_state["page"] = selected
            st.rerun()
        st.divider()
        st.markdown("**MODEL STATUS**")
        if st.session_state.get("novel_ids_model") is not None:
            info = st.session_state.get("novel_ids_model_info", {})
            st.success("Frozen model verified")
            st.caption(f"{info.get('version', 'NOVEL-IDS')} · {info.get('device', 'CPU')}")
            st.caption(f"Seed {info.get('seed', '—')}")
            if str(info.get("device", "cpu")).startswith("cpu") and torch.cuda.is_available():
                if st.button("Switch inference to GPU", width="stretch"):
                    try:
                        with st.spinner("Verifying protected hashes and loading the frozen model onto CUDA…"):
                            gpu_model = get_frozen_model(str(Path(DEFAULT_CHECKPOINT).expanduser()), "cuda")
                            st.session_state["novel_ids_model"] = gpu_model
                            st.session_state["novel_ids_model_info"] = gpu_model.model_info()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"GPU model load failed ({type(exc).__name__}): {exc}")
        else:
            st.warning("Model not loaded")
            st.caption("CUDA is preferred when available. One verified frozen model is cached and shared across browser sessions.")
            checkpoint = st.text_input("Frozen checkpoint", value=DEFAULT_CHECKPOINT)
            device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
            device = st.selectbox("Inference device", device_options, index=0)
            if st.button("Load frozen model", type="primary", width="stretch"):
                try:
                    with st.spinner("Verifying protected hashes and loading model…"):
                        model = get_frozen_model(str(Path(checkpoint).expanduser()), device)
                        st.session_state["novel_ids_model"] = model
                        st.session_state["novel_ids_model_info"] = model.model_info()
                        st.session_state["novel_ids_load_error"] = None
                    st.rerun()
                except Exception as exc:
                    st.session_state["novel_ids_load_error"] = f"{type(exc).__name__}: {exc}"
            if st.session_state.get("novel_ids_load_error"):
                st.error(st.session_state["novel_ids_load_error"])
        st.divider()
        st.caption("Frozen inference only · no training or threshold tuning")


def render_dashboard() -> None:
    page_header("Security overview", "A live summary of traffic analyzed by the frozen NOVEL-IDS inference pipeline.")
    if st.button("＋ Analyze another dataset", type="primary", key="dashboard_new_analysis"):
        st.session_state["page"] = "Traffic Analysis"
        st.rerun()
    predictions = st.session_state.get("novel_ids_predictions")
    summary = st.session_state.get("novel_ids_summary")
    meta = st.session_state.get("novel_ids_analysis_meta") or {}
    if predictions is None or summary is None:
        st.info("No traffic has been analyzed in this session. Start with Traffic Analysis to generate real predictions.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Analyzed records", "—")
        c2.metric("Known attacks", "—")
        c3.metric("Uncertain / unfamiliar", "—")
        if st.button("Open Traffic Analysis", type="primary"):
            st.session_state["page"] = "Traffic Analysis"
            st.rerun()
        return
    counts = decision_counts(summary)
    total = int(summary.get("rows", len(predictions)))
    cols = st.columns(4)
    cols[0].metric("Total records", f"{total:,}")
    cols[1].metric("Normal traffic", f"{counts['Normal']:,}")
    cols[2].metric("Known attacks", f"{counts['Known attack']:,}")
    cols[3].metric("Uncertain / unfamiliar", f"{counts['Uncertain'] + counts['Unfamiliar']:,}")
    left, right = st.columns([1, 1])
    with left:
        make_donut(counts, "Traffic decision distribution")
    with right:
        attack_rows = predictions[predictions["decision"] == "Known attack"]
        attack_counts = attack_rows["predicted_category"].fillna("Unspecified").value_counts()
        st.subheader("Attack category distribution")
        if attack_counts.empty:
            st.info("No accepted attack categories in this analysis.")
        else:
            st.bar_chart(attack_counts.rename_axis("Category"), color="#ff7183")
    st.subheader("Recent analysis sessions")
    recent_rows = []
    for item in st.session_state.get("novel_ids_history", [])[:8]:
        item_meta, item_summary = item["meta"], item["summary"]
        item_counts = decision_counts(item_summary)
        recent_rows.append({
            "Dataset": item_meta.get("dataset", "—"),
            "File": item_meta.get("filename", "—"),
            "Records": item_meta.get("rows", 0),
            "Normal": item_counts["Normal"],
            "Known attacks": item_counts["Known attack"],
            "Uncertain": item_counts["Uncertain"],
            "Unfamiliar": item_counts["Unfamiliar"],
            "Analyzed at": item_meta.get("timestamp", "—"),
        })
    st.dataframe(pd.DataFrame(recent_rows), width="stretch", hide_index=True)
    st.caption(f"Recognized dataset: {meta.get('dataset', '—')} · Trust: {summary.get('trust_level', '—')} · "
               f"Compatibility: {summary.get('compatibility', {}).get('compatibility_score', 0):.3f}")


def render_traffic_analysis() -> None:
    page_header("Traffic analysis", "Upload a source dataset or an external traffic file and run the existing frozen inference pipeline.")
    if st.session_state.get("novel_ids_model") is None:
        st.warning("The frozen model loads automatically on the first analysis if it has not already been loaded.")
    dataset = st.selectbox(
        "Dataset / schema", ["Auto-detect", "UNSW-NB15", "NSL-KDD"],
        help="Auto-detect is recommended. The backend confirms the exact feature schema before prediction.",
    )
    uploaded = st.file_uploader("Upload network traffic (maximum 1 GB)", type=["csv", "xlsx"], key="traffic_upload")
    if uploaded is None:
        st.markdown('<p class="section-note">Supported formats: CSV and XLSX, up to 1 GB. CSV is strongly recommended for very large files because it can be streamed with lower memory use. Excel conversion requires substantially more RAM.</p>', unsafe_allow_html=True)
        return
    extension = Path(uploaded.name).suffix.lower().lstrip(".")
    try:
        uploaded.seek(0)
        if extension == "csv":
            preview = pd.read_csv(uploaded, nrows=10)
            uploaded.seek(0)
            header = pd.read_csv(uploaded, nrows=0)
            columns = list(header.columns)
        else:
            preview = pd.read_excel(uploaded, engine="openpyxl", nrows=10)
            uploaded.seek(0)
            header = pd.read_excel(uploaded, engine="openpyxl", nrows=0)
            columns = list(header.columns)
        uploaded.seek(0)
        if len(columns) == 0:
            st.error("No columns were detected in this file.")
            return
        if len(set(columns)) != len(columns):
            st.error("Duplicate column names detected. Rename duplicate columns before analysis.")
            return
    except Exception as exc:
        st.error(f"Unable to read the uploaded file ({type(exc).__name__}): {exc}")
        return
    file_mb = int(uploaded.size) / (1024 * 1024)
    a, b, c = st.columns(3)
    a.metric("File", uploaded.name)
    b.metric("Columns", f"{len(columns):,}")
    c.metric("File size", f"{file_mb:.2f} MB")
    st.subheader("Data preview")
    st.dataframe(preview, width="stretch", height=250)
    label_options = ["(No ground-truth label)"] + [str(x) for x in columns]
    label_choice = st.selectbox(
        "Ground-truth label column (excluded from model input)",
        label_options,
        help="Select an actual outcome/category column only when the uploaded file contains ground truth. It is never passed to the model.",
    )
    label_column = None if label_choice == "(No ground-truth label)" else label_choice
    with st.expander("Inference settings"):
        batch_size = st.select_slider("Model batch size", options=[16, 32, 64, 128], value=32,
                                      help="The default is conservative to reduce peak memory use. Increase only if the machine has sufficient free RAM.")
        chunk_choice = st.selectbox(
            "CSV scan chunk size",
            options=["auto", 512, 1024, 2048, 4096, 8192, 16384],
            index=0,
            format_func=lambda value: "Automatic (recommended)" if value == "auto" else f"{value:,} rows",
            help="Automatic mode considers uploaded file size and currently free RAM. Manual values are available for controlled testing.",
        )
        st.caption("Automatic mode targets 4,096 rows per chunk, or 8,192 when at least 5 GiB RAM is free. It reduces to 2,048 under memory pressure; the 2 GiB RAM hard stop remains active.")
    st.markdown("---")
    if st.button("ANALYZE TRAFFIC", type="primary", width="stretch",
):
        run_analysis(uploaded, extension, dataset, label_column, batch_size, chunk_choice)


def render_results() -> None:
    page_header("Analysis results", "Prediction outcomes, confidence and traffic classification from the latest analysis.")
    if st.button("← Back to Dashboard", key="results_back_dashboard"):
        st.session_state["page"] = "Dashboard"
        st.rerun()
    predictions = st.session_state.get("novel_ids_predictions")
    summary = st.session_state.get("novel_ids_summary")
    meta = st.session_state.get("novel_ids_analysis_meta") or {}
    if predictions is None or summary is None:
        st.info("There are no results yet. Upload a traffic file and select ANALYZE TRAFFIC.")
        if st.button("Go to Traffic Analysis", type="primary"):
            st.session_state["page"] = "Traffic Analysis"
            st.rerun()
        return
    counts = decision_counts(summary)
    total = max(int(summary.get("rows", len(predictions))), 1)
    cols = st.columns(4)
    cols[0].metric("Records analyzed", f"{total:,}")
    cols[1].metric("Normal", f"{counts['Normal']:,}", f"{counts['Normal']/total:.1%}")
    cols[2].metric("Known attacks", f"{counts['Known attack']:,}", f"{counts['Known attack']/total:.1%}")
    cols[3].metric("Uncertain / unfamiliar", f"{counts['Uncertain']+counts['Unfamiliar']:,}",
                   f"{(counts['Uncertain']+counts['Unfamiliar'])/total:.1%}")
    st.caption(f"{meta.get('filename', 'Uploaded file')} · {meta.get('dataset', 'Unknown schema')} · "
               f"{meta.get('timestamp', '')} · Trust: {summary.get('trust_level', '—')}")
    timings = meta.get("timings_seconds", {})
    if timings:
        st.caption("Processing time (seconds): " + " · ".join(
            f"{name.replace('_', ' ')}: {value:.2f}" for name, value in timings.items()
        ))
    left, right = st.columns(2)
    with left:
        make_donut(counts, "Prediction distribution")
    with right:
        st.subheader("Attack category distribution")
        attacks = predictions[predictions["decision"] == "Known attack"]["predicted_category"].dropna().value_counts()
        if attacks.empty:
            st.info("No accepted attack categories were produced.")
        else:
            st.bar_chart(attacks.rename_axis("Attack category"), color="#ff7183")
    st.subheader("Confidence distribution")
    if "confidence" in predictions and len(predictions):
        bins = np.linspace(0, 1, 11)
        hist, edges = np.histogram(predictions["confidence"].astype(float), bins=bins)
        chart = pd.DataFrame({"Confidence range": [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(hist))],
                              "Records": hist}).set_index("Confidence range")
        st.bar_chart(chart, color="#43c7e8")
        st.caption("Confidence is the model's calibrated maximum class probability. It is not model accuracy.")
    st.subheader("Prediction table")
    threshold = st.slider("Additional UI review flag · confidence below", 0.0, 1.0,
                          step=0.05, key="review_threshold",
                          help="Display-only flag. It does not modify Stage 12 decisions or frozen thresholds.")
    # Add display-only columns once to the session frame; avoid duplicating a large result table.
    table = predictions
    if "prob_Normal" in table.columns:
        prob_cols = [c for c in table.columns if c.startswith("prob_")]
        table["attack_probability"] = 1.0 - table["prob_Normal"].astype(float)
    else:
        table["attack_probability"] = np.nan
    table["ui_review_flag"] = np.where(
        table["confidence"].astype(float) < threshold, "Review", "—"
    )
    table["binary_classification"] = np.select([table["decision"] == "Normal", table["decision"] == "Known attack"], ["Normal", "Attack"], default="Rejected")
    display_columns = ["row_index"]
    display_columns += ["binary_classification", "decision", "predicted_category", "best_guess_class",
                        "confidence", "attack_probability", "ood_score", "ui_review_flag"]
    display_columns = [x for x in display_columns if x in table.columns]
    controls = st.columns([2, 1, 1])
    query = controls[0].text_input("Search rows", placeholder="Search category, decision, row ID…")
    decision_filter = controls[1].selectbox("Decision filter", ["All", *DECISIONS])
    page_size = controls[2].selectbox("Rows per page", [25, 50, 100, 250], index=1)
    filtered = table
    if decision_filter != "All":
        filtered = filtered[filtered["decision"] == decision_filter]
    if query.strip():
        mask = filtered.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]
    sort_col = st.selectbox("Sort by", [x for x in ["row_index", "confidence", "attack_probability", "ood_score"] if x in filtered.columns])
    ascending = st.checkbox("Ascending", value=True)
    # The inference output is already in row_index order; skip a full-table sort on the default view.
    if sort_col != "row_index" or not ascending:
        filtered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")
    max_page = max(1, int(np.ceil(len(filtered) / page_size)))
    page_num = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1)
    start = (int(page_num) - 1) * page_size
    page_frame = filtered.iloc[start:start + page_size]
    st.dataframe(page_frame[display_columns], width="stretch", height=430, hide_index=True)
    st.caption(f"Showing {len(page_frame):,} of {len(filtered):,} filtered rows · {len(table):,} total.")
    export_signature = (query, decision_filter, sort_col, bool(ascending))
    if st.session_state.get("novel_ids_filtered_export_signature") != export_signature:
        st.session_state["novel_ids_filtered_export"] = None
        st.session_state["novel_ids_filtered_export_signature"] = export_signature
    export_left, export_right = st.columns(2)
    with export_left:
        if st.button("Prepare filtered CSV", key="prepare_filtered_csv"):
            st.session_state["novel_ids_filtered_export"] = filtered.to_csv(index=False).encode("utf-8")
        if st.session_state.get("novel_ids_filtered_export") is not None:
            st.download_button("Download filtered CSV", st.session_state["novel_ids_filtered_export"],
                               "novel_ids_filtered_predictions.csv", "text/csv")
    with export_right:
        if st.button("Prepare complete results CSV", key="prepare_complete_csv"):
            st.session_state["novel_ids_complete_export"] = table.to_csv(index=False).encode("utf-8")
        if st.session_state.get("novel_ids_complete_export") is not None:
            st.download_button("Download complete results CSV", st.session_state["novel_ids_complete_export"],
                               "novel_ids_predictions.csv", "text/csv")
    st.download_button("Download analysis summary JSON",
                       json.dumps({"metadata": meta, "summary": summary},                                  indent=2, allow_nan=False, default=str).encode("utf-8"),
                       "novel_ids_analysis_summary.json", "application/json")
    with st.expander("Feature compatibility and onboarding"):
        onboarding = summary.get("onboarding", {})
        st.json({
            "recognized_dataset": onboarding.get("recognized_dataset"),
            "inference_mode": onboarding.get("inference_mode"),
            "compatibility": summary.get("compatibility"),
            "trust_level": summary.get("trust_level"),
            "profile": onboarding.get("profile"),
            "normalization": onboarding.get("config", {}).get("normalization"),
        })
    with st.expander("Model provenance and frozen thresholds"):
        st.json(summary.get("model", {}))
        st.json(summary.get("thresholds", {}))
    st.caption("Uncertain and Unfamiliar are Stage 12 rejection decisions. The UI review flag is display-only. "
               "Best-guess classes for rejected rows are not accepted classifications.")


def render_history() -> None:
    page_header("Prediction history", "Analysis sessions retained in this Streamlit session. History clears when the session resets.")
    history = st.session_state.get("novel_ids_history", [])
    if not history:
        st.info("No prediction sessions have been recorded yet.")
        return
    rows = []
    for index, item in enumerate(history):
        meta, summary = item["meta"], item["summary"]
        counts = decision_counts(summary)
        rows.append({
            "Session": index + 1, "Dataset": meta.get("dataset"),
            "File": meta.get("filename"), "Records": meta.get("rows"),
            "Normal": counts["Normal"], "Known attacks": counts["Known attack"],
            "Uncertain": counts["Uncertain"], "Unfamiliar": counts["Unfamiliar"],
            "Timestamp": meta.get("timestamp"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    choice = st.selectbox("Open a previous analysis", list(range(len(history))),
                          format_func=lambda i: f"{history[i]['meta'].get('filename')} · {history[i]['meta'].get('timestamp')}")
    if st.button("Open selected results", type="primary"):
        item = history[choice]
        st.session_state["novel_ids_predictions"] = item["predictions"].copy()
        st.session_state["novel_ids_summary"] = item["summary"]
        st.session_state["novel_ids_analysis_meta"] = item["meta"]
        st.session_state["page"] = "Analysis Results"
        st.rerun()


def main() -> None:
    init_state()
    render_sidebar()
    page = st.session_state.get("page", "Dashboard")
    if page == "Dashboard":
        render_dashboard()
    elif page == "Traffic Analysis":
        render_traffic_analysis()
    elif page == "Analysis Results":
        render_results()
    elif page == "Prediction History":
        render_history()
    else:
        st.session_state["page"] = "Dashboard"
        st.rerun()


if __name__ == "__main__":
    main()
