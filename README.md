# NOVEL-IDS Web Application

A Streamlit-based web interface for the NOVEL-IDS
network intrusion detection project.

## Features

- Upload UNSW-NB15 or NSL-KDD datasets.
- CSV and Excel file support.
- Binary classification: Normal or Attack.
- Attack-category prediction using dataset-specific classifiers.
- Attack probability.
- Prediction summary and downloadable CSV results.

## Model Architecture

The application uses:

1. Shared Feature Encoder
2. Dataset-Specific Adapters
3. Adaptive Residual Fusion
4. Transformer Binary Classifier
5. Dataset-Specific Attack Category Classifiers

## Supported Datasets

- UNSW-NB15
- NSL-KDD

Uploaded files must contain the required original dataset features.
Arbitrary network traffic files and CIC-DDoS2019 are not currently supported.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Deployment

Deploy this folder using Streamlit Community Cloud.

Main application file: `app.py`

## Important Notes

- The application uses saved trained model weights.
- It does not retrain models when users upload files.
- Uploaded executable files are not supported or executed.
- Predictions are research-project outputs, not a guarantee of
  production-grade network security.
