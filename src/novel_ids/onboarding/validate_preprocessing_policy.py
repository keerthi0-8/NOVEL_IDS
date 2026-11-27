
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

POLICY_PATH = PROJECT_ROOT / "config" / "preprocessing_policy.yaml"


def main():
    with open(POLICY_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    policy = config["preprocessing_policy"]

    checks = {
        "Training median for missing numeric values":
            policy["numeric_features"]["missing_values"]
            == "training_median",

        "Infinite values handled before imputation":
            policy["numeric_features"]["infinite_values"]
            == "replace_with_missing",

        "Log transformation selected using training data":
            policy["numeric_features"]["transformations"]
            ["skewed_non_negative"]["selection"]
            == "training_statistics_and_feature_semantics",

        "Robust Scaling enabled":
            policy["numeric_features"]["scaling"]["method"]
            == "robust_scaling",

        "Scaling fitted on source training data only":
            policy["numeric_features"]["scaling"]["fit_on"]
            == "training_partition_only",

        "Dataset-specific categories preserved":
            policy["categorical_features"]
            ["preserve_dataset_specific_vocabularies"] is True,

        "Feature availability mask enabled":
            policy["feature_availability"]["availability_mask"]
            == "M_avail"
            and policy["feature_availability"]["measurement_missingness_mask"]
            == "M_miss"
            and policy["feature_availability"]["distinguish_absent_features_from_missing_measurements"] is True,

        "Uncommon features preserved":
            policy["dataset_preservation"]
            ["discard_uncommon_features"] is False,

        "Unseen Option A uses unlabeled values without changing source fitting":
            policy["data_leakage_prevention"]
            ["unseen_data_used_for_source_fitting"] is False
            and policy["data_leakage_prevention"]
            ["unseen_labels_used_for_normalization"] is False
            and policy["data_leakage_prevention"]
            ["unseen_predictions_metrics_or_feedback_used_for_normalization"] is False
            and policy["missing_value_handling"]["unseen_dataset"]
            ["numeric_normalization"]["approved_option"] == "A"
            and policy["missing_value_handling"]["unseen_dataset"]
            ["numeric_normalization"]["fit_on"]
            == "unseen_unlabeled_available_finite_feature_values_only",

        "Original feature names preserved":
            policy["dataset_preservation"]
            ["preserve_original_feature_names"] is True,
    }

    print("\n" + "=" * 60)
    print("NOVEL-IDS PREPROCESSING POLICY VALIDATION")
    print("=" * 60)

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    print("\n" + "=" * 60)

    if all(checks.values()):
        print(f"ALL {len(checks)} POLICY CHECKS PASSED")
    else:
        print("POLICY VALIDATION FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()
