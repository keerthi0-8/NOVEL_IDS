
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]

with open(ROOT / "config" / "datasets.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Read NSL-KDD feature names and training data.
nsl_metadata = pd.read_csv(
    config["datasets"]["nsl_kdd"]["metadata"],
    header=None,
)
nsl_names = nsl_metadata.iloc[:, 0].astype(str).str.strip().tolist()

nsl = pd.read_csv(
    config["datasets"]["nsl_kdd"]["train"],
    header=None,
)
nsl.columns = nsl_names + ["attack_label", "difficulty"]

# Read UNSW-NB15 training data.
unsw = pd.read_csv(config["datasets"]["unsw_nb15"]["train"])

mappings = [
    ("flow_duration", "duration", "dur"),
    ("transport_protocol", "protocol_type", "proto"),
    ("network_service", "service", "service"),
    ("source_to_destination_bytes", "src_bytes", "sbytes"),
    ("destination_to_source_bytes", "dst_bytes", "dbytes"),
]

print("\n" + "=" * 65)
print("COMMON FEATURE VALUE COMPATIBILITY AUDIT")
print("Training data only — no preprocessing or model fitting")
print("=" * 65)

for semantic_id, nsl_col, unsw_col in mappings:
    print(f"\n--- {semantic_id} ---")

    for dataset_name, dataframe, column in [
        ("NSL-KDD", nsl, nsl_col),
        ("UNSW-NB15", unsw, unsw_col),
    ]:
        series = dataframe[column]

        print(f"\n{dataset_name}: {column}")
        print("Data type:", series.dtype)
        print("Missing values:", series.isna().sum())
        print("Unique values:", series.nunique())

        if pd.api.types.is_numeric_dtype(series):
            print(
                series.describe(
                    percentiles=[0.25, 0.50, 0.75, 0.95]
                ).to_string()
            )
        else:
            print("Category frequencies:")
            print(series.value_counts(dropna=False).head(30).to_string())

    nsl_values = set(nsl[nsl_col].dropna().astype(str).unique())
    unsw_values = set(unsw[unsw_col].dropna().astype(str).unique())

    if not (
        pd.api.types.is_numeric_dtype(nsl[nsl_col])
        or pd.api.types.is_numeric_dtype(unsw[unsw_col])
    ):
        common = nsl_values & unsw_values
        print("\nCategory vocabulary comparison:")
        print("Shared category values:", len(common))
        print("NSL-KDD-only categories:", sorted(nsl_values - unsw_values))
        print("UNSW-NB15-only categories:", sorted(unsw_values - nsl_values))

print("\n" + "=" * 65)
print("AUDIT COMPLETE")
print("=" * 65)
