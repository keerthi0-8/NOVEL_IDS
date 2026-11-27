import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from novel_ids.inference.api import _registry
from novel_ids.onboarding.unseen_schema import build_unseen_model_inputs


def test_external_metadata_supports_level1_level2_and_level3_without_labels():
    root = Path(__file__).resolve().parents[1]
    metadata = yaml.safe_load((root / "config/cic_feature_metadata.yaml").read_text(encoding="utf-8"))["features"]
    encoder, slots, matching = _registry()
    level1 = [name for name, item in metadata.items() if item["mapping"]["level"] == 1]
    level2 = [name for name, item in metadata.items() if item["mapping"]["level"] == 2]
    level3 = [name for name, item in metadata.items() if item["mapping"]["level"] == 3]
    selected = level1 + [name for name in level2 if metadata[name]["mapping"].get("semantic_group") in {"packet_volume", "packet_size_measurement"}][:2] + level3[:1]
    values = {}
    for name in selected:
        transform = metadata[name].get("value_transform")
        if transform == "iana_protocol_to_name":
            values[name] = [6, 17, 1]
        else:
            values[name] = [1.0, 2.0, 3.0]
    frame = pd.DataFrame(values)
    output = build_unseen_model_inputs(
        frame,
        slots,
        encoder,
        normalization_frame=frame,
        matching_descriptors=matching,
        external_metadata={name: metadata[name] for name in selected},
        similarity_threshold=0.15,
        normalization_mode="option_b_strict_source_level1",
    )
    extension = output["feature_descriptors"][78:]
    assert sum(item["level"] == 1 for item in extension) == 4
    assert all(item["relationship"] == "equivalent" for item in extension if item["level"] == 1)
    assert all(item["relationship"] == "group_only" for item in extension if item["level"] == 2)
    assert all(item["relationship"] == "dataset_specific" for item in extension if item["level"] == 3)
    assert output["source_training_level1_normalization"]
    assert " Protocol" in output["declared_feature_transforms"]
def test_exact_source_training_feature_names_are_recognized_without_external_metadata():
    encoder, slots, matching = _registry()
    source_level1 = next(item for item in matching if item.get("level") == 1 and item.get("review_status") == "approved")
    source_level2 = next(item for item in matching if item.get("level") == 2 and item.get("review_status") == "approved")
    level1_name = source_level1["original_feature_name"]
    level2_name = source_level2["original_feature_name"]
    frame = pd.DataFrame({
        level1_name: [1.0, 2.0, 3.0],
        level2_name: [1.0, 2.0, 3.0],
        "project_custom_unmapped_feature": [4.0, 5.0, 6.0],
    })
    output = build_unseen_model_inputs(
        frame, slots, encoder, normalization_frame=frame,
        matching_descriptors=matching, similarity_threshold=1.0,
    )
    extension = output["feature_descriptors"][78:]
    assert extension[0]["level"] == 1
    assert extension[0]["semantic_id"] == source_level1["semantic_id"]
    assert extension[1]["level"] == 2
    assert extension[1]["relationship"] == "group_only"
    assert extension[2]["level"] == 3


def test_metadata_numeric_columns_coerce_object_calibration_values():
    root = Path(__file__).resolve().parents[1]
    metadata = yaml.safe_load((root / "config/cic_feature_metadata.yaml").read_text(encoding="utf-8"))["features"]
    encoder, slots, matching = _registry()
    name = "Total Length of Fwd Packets"
    frame = pd.DataFrame({name: [10.0, 20.0, 30.0]})
    calibration = pd.DataFrame({name: pd.Series(["10.0", "not-a-number", "30.0"], dtype=object)})
    output = build_unseen_model_inputs(
        frame, slots, encoder, normalization_frame=calibration,
        matching_descriptors=matching, external_metadata={name: metadata[name]},
        similarity_threshold=0.15, normalization_mode="option_a_target",
    )
    assert output["numeric_token_mask"][78]
    assert np.isfinite(output["numeric_values"]).all()
    assert output["M_miss"][:, 78].tolist() == [0.0, 0.0, 0.0]
