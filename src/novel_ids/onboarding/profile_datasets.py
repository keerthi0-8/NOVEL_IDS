
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "datasets.yaml"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def profile_dataset(name, settings):
    print(f"\n{'=' * 60}")
    print(f"DATASET: {name.upper()}")
    print("=" * 60)

    train_path = Path(settings["train"])
    test_path = Path(settings["test"])

    for split, path in [("TRAIN", train_path), ("TEST", test_path)]:
        print(f"\n--- {split} ---")
        print(f"File: {path}")

        if not path.is_file():
            print("ERROR: Dataset file not found.")
            continue

        if settings["format"] == "csv_no_header":
            data = pd.read_csv(path, header=None)
        else:
            data = pd.read_csv(path)

        print(f"Rows: {data.shape[0]:,}")
        print(f"Columns: {data.shape[1]}")
        print(f"Missing values: {data.isna().sum().sum():,}")
        print(f"Duplicate rows: {data.duplicated().sum():,}")
        print(f"Memory usage: {data.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

        if split == "TRAIN":
            print("\nColumn names:")
            print(data.columns.tolist())

        del data


def main():
    config = load_config()

    for name, settings in config["datasets"].items():
        profile_dataset(name, settings)


if __name__ == "__main__":
    main()
