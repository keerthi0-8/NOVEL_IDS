
from pathlib import Path

import joblib
import pandas as pd
import yaml

from novel_ids.onboarding.load_source_datasets import validation_split_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLIT_DIR = PROJECT_ROOT / "data" / "interim" / "validation_splits"
CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"
OUTPUT_PATH = PROJECT_ROOT / "results" / "validation_split_audit.txt"


def audit_dataset(name, config):
    split_path = validation_split_path({"datasets": {name: config}}, name)
    data = joblib.load(split_path)

    X_train = data["X_train"]
    X_val = data["X_validation"]
    y_train = data["y_train"]
    y_val = data["y_validation"]

    train_hashes = set(
        pd.util.hash_pandas_object(X_train, index=False)
    )
    val_hashes = set(
        pd.util.hash_pandas_object(X_val, index=False)
    )

    overlap = train_hashes.intersection(val_hashes)

    train_classes = set(y_train.unique())
    val_classes = set(y_val.unique())

    missing_val_classes = sorted(train_classes - val_classes)

    test_path = config["test"]
    header = None if config["format"] == "csv_no_header" else 0
    original_test = pd.read_csv(test_path, header=header)

    lines = [
        "=" * 65,
        name.upper(),
        "=" * 65,
        f"Training records: {len(X_train)}",
        f"Validation records: {len(X_val)}",
        f"Total source training records: {len(X_train) + len(X_val)}",
        f"Validation percentage: {len(X_val) / (len(X_train) + len(X_val)) * 100:.2f}%",
        f"Original independent test records: {len(original_test)}",
        f"Training classes: {len(train_classes)}",
        f"Validation classes: {len(val_classes)}",
        f"Classes absent from validation: {missing_val_classes}",
        f"Identical feature groups crossing partitions: {len(overlap)}",
        "",
        "TRAINING CLASS COUNTS:",
        y_train.value_counts().to_string(),
        "",
        "VALIDATION CLASS COUNTS:",
        y_val.value_counts().to_string(),
        "",
        f"Duplicate-group check: {'PASS' if not overlap else 'FAIL'}",
    ]

    return lines


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)["datasets"]

    report = [
        "NOVEL-IDS VALIDATION SPLIT AUDIT",
        "Random seed: 42",
        "Split strategy: StratifiedGroupKFold",
        "Requested split: approximately 80:20",
        "Grouping: identical input feature rows kept together",
        "Validation generated from original training files only.",
        "Original test files were kept separate.",
        "",
    ]

    for name in ["nsl_kdd", "unsw_nb15"]:
        report.extend(audit_dataset(name, config[name]))
        report.append("")

    report_path = OUTPUT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("\nAudit report saved to:", report_path)


if __name__ == "__main__":
    main()
