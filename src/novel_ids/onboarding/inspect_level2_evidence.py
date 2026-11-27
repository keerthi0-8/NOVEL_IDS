
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]

FILE = (
    ROOT / "src" / "novel_ids" / "onboarding"
    / "feature_descriptors.yaml"
)

GROUPS_TO_REVIEW = {
    "authentication_activity",
    "host_and_privilege_activity",
    "application_protocol_activity",
    "connection_history_count",
    "connection_context_ratio",
    "packet_loss_or_retransmission",
    "state_ttl_condition_count",
    "application_content_size",
}


def main():

    with open(FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    features = data["features"]

    print("=" * 75)
    print("LEVEL 2 SEMANTIC EVIDENCE REVIEW")
    print("=" * 75)

    for group in sorted(GROUPS_TO_REVIEW):

        records = [
            f for f in features
            if f.get("level") == 2
            and f.get("semantic_group") == group
        ]

        print(f"\nGROUP: {group}")
        print("-" * 75)

        for f in records:
            print(f"\nFeature: {f['dataset']}::{f['original_feature_name']}")
            print("Relationship:", f.get("relationship"))
            print("Confidence:", f.get("confidence"))
            print("Description:", f.get("description"))
            print("Description source:", f.get("description_source"))
            print("Evidence:", f.get("evidence"))
            print("Unit:", f.get("unit"))
            print("Unit status:", f.get("unit_status"))

    print("\nEVIDENCE INSPECTION COMPLETE")
    print("No files were modified.")


if __name__ == "__main__":
    main()
