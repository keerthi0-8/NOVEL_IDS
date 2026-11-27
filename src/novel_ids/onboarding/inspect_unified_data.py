
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

files = [
    "data/interim/nsl_kdd_unified.joblib",
    "data/interim/unsw_nb15_unified.joblib",
]

for filename in files:
    filepath = PROJECT_ROOT / filename
    obj = joblib.load(filepath)
    X = obj["X_train"]

    print("\n" + "=" * 70)
    print("DATASET:", obj["dataset_name"])
    print("=" * 70)

    print("\nFeature counts by data type:")
    print(X.dtypes.astype(str).value_counts())

    print("\nFeature details:")
    for column in X.columns:
        series = X[column]
        print(
            f"{column} | dtype={series.dtype} "
            f"| missing={series.isna().sum()} "
            f"| unique={series.nunique(dropna=True)}"
        )

        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            print("  Categories:", series.dropna().unique()[:8].tolist())
        elif pd.api.types.is_numeric_dtype(series):
            print(
                "  Min:", series.min(),
                "| Max:", series.max(),
                "| Skew:", round(series.skew(), 3)
                if series.nunique(dropna=True) > 1 else "constant"
            )

    print("\nAvailability mask:")
    print("Available:", int(np.sum(obj["availability_mask"])))
    print("Unavailable:", int(len(obj["availability_mask"]) -
                              np.sum(obj["availability_mask"])))

    print("\nLabel examples:", obj["y_train"].unique()[:10].tolist())
