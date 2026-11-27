import numpy as np
import pandas as pd

from novel_ids.inference import onboard_csv


def test_onboarding_excludes_labels_and_handles_nan_inf_unknown_categories(tmp_path):
    n = 12
    frame = pd.DataFrame({
        "numeric": np.arange(n, dtype=float),
        "all_nan": np.full(n, np.nan),
        "with_inf": np.where(np.arange(n) % 2 == 0, np.inf, np.arange(n, dtype=float)),
        "protocol": ["tcp", "udp", "new_proto"] * 4,
        "Label": ["Normal", "DoS"] * 6,
    })
    path = tmp_path / "unseen.csv"
    frame.to_csv(path, index=False)
    first = onboard_csv(path, label_column="Label")
    assert first.rows == n
    assert "Label" not in first.feature_names
    assert first.model_inputs["numeric_values"].shape[0] == n
    assert np.isfinite(first.model_inputs["numeric_values"]).all()
    assert np.all(first.model_inputs["source_feature_indices"] == -1)
    assert first.profile["all_missing_columns"] == ["all_nan"]
    assert first.profile["infinite_numeric_values"] == 6
    # The 78 source slots are unavailable; each incoming feature is available.
    assert np.all(first.model_inputs["M_avail"][:, :78] == 0)
    assert np.all(first.model_inputs["M_avail"][:, 78:] == 1)
    assert first.compatibility["level1_promotion_allowed"] is False


def test_label_values_do_not_change_unseen_feature_tensors(tmp_path):
    n = 8
    frame = pd.DataFrame({"duration": np.arange(n, dtype=float), "proto": ["tcp", "udp"] * 4, "Label": ["Normal"] * n})
    path = tmp_path / "labels.csv"
    frame.to_csv(path, index=False)
    a = onboard_csv(path, label_column="Label")
    frame["Label"] = ["attack-X", "attack-Y"] * 4
    frame.to_csv(path, index=False)
    b = onboard_csv(path, label_column="Label")
    for key in ("numeric_values", "categorical_values", "M_avail", "M_miss"):
        assert np.array_equal(a.model_inputs[key], b.model_inputs[key])
