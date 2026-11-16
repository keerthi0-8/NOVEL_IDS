
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import torch
import joblib

# Project paths
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

sys.path.insert(0, str(FRONTEND_DIR))

from model_loader import NOVELIDSModel, AttackCategoryClassifier
from inference import predict_uploaded_data


# --------------------------------------------------
# STEP 1: PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NOVEL-IDS | Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# STEP 2: LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_project_models():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    saved_models = torch.load(
        ARTIFACT_DIR / "novel_ids_two_stage_models.pt",
        map_location=device,
        weights_only=True
    )

    binary_model = NOVELIDSModel().to(device)
    binary_model.load_state_dict(saved_models["binary_model"])
    binary_model.eval()

    unsw_category_model = AttackCategoryClassifier(64, 9).to(device)
    unsw_category_model.load_state_dict(
        saved_models["unsw_category_model"]
    )
    unsw_category_model.eval()

    nsl_category_model = AttackCategoryClassifier(64, 22).to(device)
    nsl_category_model.load_state_dict(
        saved_models["nsl_category_model"]
    )
    nsl_category_model.eval()

    preprocessing = joblib.load(
        ARTIFACT_DIR / "preprocessing_bundle.joblib"
    )

    return (
        binary_model,
        unsw_category_model,
        nsl_category_model,
        preprocessing,
        device
    )


# --------------------------------------------------
# STEP 3: WEBSITE HEADER
# --------------------------------------------------

st.title("🛡️ NOVEL-IDS")
st.subheader("Network Intrusion Detection System")

st.write(
    "An AI-based intrusion detection system using "
    "shared feature encoding, dataset-specific learning "
    "and a two-stage Transformer classification approach."
)

st.divider()


# --------------------------------------------------
# STEP 4: DATASET SELECTION
# --------------------------------------------------

st.header("1. Select Dataset")

dataset = st.selectbox(
    "Choose the dataset format of your uploaded file:",
    ["UNSW", "NSL"]
)

if dataset == "UNSW":
    st.info(
        "Upload a CSV or Excel file containing the original "
        "UNSW-NB15 feature columns."
    )
else:
    st.info(
        "Upload a CSV or Excel file containing the original "
        "NSL-KDD feature columns."
    )


# --------------------------------------------------
# STEP 5: FILE UPLOAD
# --------------------------------------------------

st.header("2. Upload Network Data")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully.")

        st.write("Uploaded records:", len(input_df))
        st.write("Available columns:", len(input_df.columns))

        with st.expander("Preview uploaded data"):
            st.dataframe(input_df.head(10), use_container_width=True)

        # --------------------------------------------------
        # STEP 6: RUN PREDICTIONS
        # --------------------------------------------------

        st.header("3. Detect Network Intrusions")

        if st.button(
            "Run NOVEL-IDS Detection",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Loading models and analyzing network records..."
            ):

                try:
                    (
                        binary_model,
                        unsw_category_model,
                        nsl_category_model,
                        preprocessing,
                        device
                    ) = load_project_models()

                    results = predict_uploaded_data(
                        input_df,
                        dataset,
                        binary_model,
                        unsw_category_model,
                        nsl_category_model,
                        preprocessing,
                        device
                    )

                    output_df = pd.concat(
                        [
                            input_df.reset_index(drop=True),
                            results.reset_index(drop=True)
                        ],
                        axis=1
                    )

                    st.session_state["novel_ids_results"] = output_df
                    st.session_state["novel_ids_dataset"] = dataset

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

    except Exception as error:
        st.error(f"Unable to read the uploaded file: {error}")


# --------------------------------------------------
# STEP 7: DISPLAY RESULTS
# --------------------------------------------------

if "novel_ids_results" in st.session_state:

    output_df = st.session_state["novel_ids_results"]
    result_dataset = st.session_state["novel_ids_dataset"]

    st.divider()
    st.header("4. Detection Results")

    total_records = len(output_df)

    normal_count = (
        output_df["Binary_Prediction"] == "Normal"
    ).sum()

    attack_count = (
        output_df["Binary_Prediction"] == "Attack"
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Normal Records", f"{normal_count:,}")
    col3.metric("Attack Records", f"{attack_count:,}")

    st.subheader("Prediction Distribution")

    chart_data = pd.DataFrame({
        "Prediction": ["Normal", "Attack"],
        "Records": [normal_count, attack_count]
    })

    st.bar_chart(
        chart_data.set_index("Prediction")
    )

    st.subheader("Prediction Details")

    st.dataframe(
        output_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Download Results")

    csv_data = output_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Results (CSV)",
        data=csv_data,
        file_name=f"NOVEL_IDS_{result_dataset}_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "Predictions are generated using the saved NOVEL-IDS "
        "binary model and dataset-specific attack-category classifier."
    )


# --------------------------------------------------
# STEP 8: FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "NOVEL-IDS | Academic Network Intrusion Detection Project"
)
