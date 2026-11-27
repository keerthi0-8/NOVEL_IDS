
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ONBOARDING_DIR = (
    PROJECT_ROOT / "src" / "novel_ids" / "onboarding"
)

DESCRIPTOR_PATH = ONBOARDING_DIR / "feature_descriptors.yaml"
STATISTICS_PATH = ONBOARDING_DIR / "feature_statistics.yaml"


def main():
    with open(DESCRIPTOR_PATH, encoding="utf-8") as file:
        descriptor_data = yaml.safe_load(file)

    with open(STATISTICS_PATH, encoding="utf-8") as file:
        statistics_data = yaml.safe_load(file)

    attached = 0

    for descriptor in descriptor_data["features"]:
        dataset = descriptor["dataset"]
        feature = descriptor["original_feature_name"]

        try:
            statistics = statistics_data["datasets"][dataset][feature]
        except KeyError as error:
            raise ValueError(
                f"Missing statistics for {dataset}::{feature}"
            ) from error

        descriptor["statistics"] = statistics
        attached += 1

    if attached != 83:
        raise ValueError(
            f"Expected 83 attached descriptors, got {attached}"
        )

    output_path = ONBOARDING_DIR / "feature_descriptors_with_stats.yaml"

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            descriptor_data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print("Descriptors processed:", attached)
    print("Statistics attached:", attached)
    print("Output:", output_path)
    print("STATISTICS ATTACHMENT PASSED.")


if __name__ == "__main__":
    main()
