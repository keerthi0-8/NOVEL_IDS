
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INVENTORY_PATH = PROJECT_ROOT / "results" / "feature_inventory.csv"

NSL_METADATA = Path(
    r"G:\My Drive\NSL-KDD\NSL_KDD-master\Field Names.csv"
)

UNSW_METADATA = Path(
    r"G:\My Drive\UNSW-NB15\NUSW-NB15_features.csv"
)


def normalize(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )


ALIASES = {
    "smean": "smeansz",
    "dmean": "dmeansz",
    "response_body_len": "res_bdy_len",
    "sinpkt": "sintpkt",
    "dinpkt": "dintpkt",
    "ct_src_ltm": "ct_src_ltm",
}


def check_nsl(inventory):
    metadata = pd.read_csv(
        NSL_METADATA,
        header=None,
        names=["feature", "type"]
    )

    available = {
        normalize(name): name
        for name in metadata["feature"]
    }

    print("\n" + "=" * 65)
    print("NSL-KDD METADATA MATCHING")
    print("=" * 65)

    for feature in inventory[
        inventory["dataset"] == "NSL-KDD"
    ]["feature"]:

        key = normalize(feature)

        if key in available:
            print(f"MATCH: {feature} -> {available[key]}")
        else:
            print(f"NOT FOUND: {feature}")


def check_unsw(inventory):
    metadata = pd.read_csv(
        UNSW_METADATA,
        encoding="cp1252"
    )

    metadata["normalized_name"] = (
        metadata["Name"].astype(str).map(normalize)
    )

    available = {
        row["normalized_name"]: row
        for _, row in metadata.iterrows()
    }

    print("\n" + "=" * 65)
    print("UNSW-NB15 METADATA MATCHING")
    print("=" * 65)

    for feature in inventory[
        inventory["dataset"] == "UNSW-NB15"
    ]["feature"]:

        alias = ALIASES.get(feature, feature)
        key = normalize(alias)

        if key in available:
            row = available[key]

            print(
                f"MATCH: {feature} -> {row['Name']} "
                f"| Type: {row['Type ']} "
                f"| Description: {row['Description']}"
            )
        else:
            print(f"NOT FOUND: {feature}")


def main():
    inventory = pd.read_csv(INVENTORY_PATH)

    check_nsl(inventory)
    check_unsw(inventory)

    print("\nMETADATA MATCHING CHECK COMPLETE.")


if __name__ == "__main__":
    main()
