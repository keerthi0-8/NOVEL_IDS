import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from novel_ids.representation.dual_encoder import DualEncoder, CATEGORICAL_CONFIG
from novel_ids.representation.feature_identity import FeatureIdentityEncoder
from novel_ids.onboarding.unseen_schema import build_unseen_model_inputs
from novel_ids.preprocessing.numeric_preprocessor import NumericPreprocessor


def source_descriptors():
    descriptors = []
    for i in range(78):
        if i < 5:
            item = {
                "original_feature_name": f"common_{i}", "metadata_feature_name": f"common_{i}",
                "semantic_id": f"common_{i}", "level": 1, "semantic_group": None,
                "relationship": "equivalent", "confidence": "high", "review_status": "approved",
                "data_type": "continuous", "statistics": {"skewness": 0.1},
            }
        elif i == 6:
            item = {
                "original_feature_name": "flag", "description": "connection state flag",
                "level": 2, "semantic_id": None, "semantic_group": "connection_state",
                "relationship": "group_only", "confidence": "medium", "review_status": "approved",
                "data_type": "symbolic", "evidence": "Metadata describes connection status.",
                "statistics": {"skewness": 0.2},
            }
        else:
            item = {
                "original_feature_name": f"feature_{i}", "description": f"source feature {i}",
                "level": 3, "semantic_id": None, "semantic_group": None,
                "relationship": "dataset_specific", "confidence": "unassessed",
                "review_status": "dataset_specific", "data_type": "continuous",
                "statistics": {"skewness": 0.0},
            }
        descriptors.append(item)
    return descriptors


def make_model(**kwargs):
    descriptors = source_descriptors()
    identity_encoder = FeatureIdentityEncoder()
    identity_encoder.fit(descriptors)
    vectors = identity_encoder.encode(descriptors).detach().cpu().numpy()
    kwargs.setdefault("feature_set_dropout_rate", 0.0)
    kwargs.setdefault("feature_set_dropout_all_probability", 0.0)
    model = DualEncoder(
        vectors,
        feature_identity_encoder=identity_encoder,
        source_descriptors=descriptors,
        **kwargs,
    )
    return model, descriptors


def make_batch(dataset="nsl_kdd", batch_size=4, count=78, descriptors=None):
    config = CATEGORICAL_CONFIG[dataset]
    numeric_mask = torch.ones(count, dtype=torch.bool)
    categorical_mask = torch.zeros(count, dtype=torch.bool)
    for position in config["positions"]:
        numeric_mask[position] = False
        categorical_mask[position] = True
    numeric = torch.zeros(batch_size, count)
    categorical = torch.zeros(batch_size, count, dtype=torch.long)
    for position, size in zip(config["positions"], config["vocab_sizes"]):
        categorical[:, position] = torch.randint(1, size + 1, (batch_size,))
    available = torch.ones(batch_size, count)
    missing = torch.zeros(batch_size, count)
    return {
        "numeric_values": numeric,
        "categorical_values": categorical,
        "M_avail": available,
        "M_miss": missing,
        "numeric_token_mask": numeric_mask,
        "categorical_token_mask": categorical_mask,
        "dataset": dataset,
        "feature_descriptors": descriptors,
    }


def append_extension(batch, descriptors, kind="numeric"):
    count = batch["numeric_values"].shape[1]
    batch["numeric_values"] = torch.cat([batch["numeric_values"], torch.zeros(4, 1)], dim=1)
    batch["categorical_values"] = torch.cat([batch["categorical_values"], torch.zeros(4, 1, dtype=torch.long)], dim=1)
    batch["M_avail"] = torch.cat([batch["M_avail"], torch.ones(4, 1)], dim=1)
    batch["M_miss"] = torch.cat([batch["M_miss"], torch.zeros(4, 1)], dim=1)
    numeric_mask = torch.cat([batch["numeric_token_mask"], torch.tensor([kind == "numeric"])])
    categorical_mask = torch.cat([batch["categorical_token_mask"], torch.tensor([kind == "categorical"])])
    batch["numeric_token_mask"] = numeric_mask
    batch["categorical_token_mask"] = categorical_mask
    batch["source_feature_indices"] = torch.cat([torch.arange(count), torch.tensor([-1])])
    batch["feature_descriptors"] = descriptors + [{
        "original_feature_name": "never_seen_feature",
        "description": "previously unseen network field",
        "level": 3,
        "semantic_id": None,
        "semantic_group": None,
        "relationship": "dataset_specific",
        "confidence": "unassessed",
        "review_status": "dataset_specific",
        "data_type": "symbolic" if kind == "categorical" else "continuous",
        "statistics": {"skewness": 0.0},
    }]
    return batch


def test_78_token_backward_compatibility_and_finite_forward():
    model, _ = make_model(value_masking_enabled=False)
    model.eval()
    output = model(**make_batch())
    assert output["Z_s"].shape == output["Z_d"].shape == (4, 128)
    assert output["token_embeddings"].shape == (4, 78, 128)
    assert all(torch.isfinite(value).all() for value in output.values())

    # The original source-only call shape remains supported without descriptors.
    legacy = DualEncoder(model.source_identity_base.detach().cpu().numpy(), value_masking_enabled=False)
    legacy.eval()
    legacy_output = legacy(**make_batch())
    assert legacy_output["token_embeddings"].shape == (4, 78, 128)


def test_source_offsets_apply_only_to_source_specific_features():
    model, descriptors = make_model(value_masking_enabled=False)
    device = model.source_identity_base.device
    indices = torch.arange(78, device=device)
    with torch.no_grad():
        model.source_identity_offsets[6].fill_(0.25)
    identities, base = model._identity_tokens(78, indices, descriptors, device)
    assert torch.equal(identities[0], base[0])
    assert torch.allclose(identities[6] - base[6], torch.full((128,), 0.25))


@pytest.mark.parametrize("extension_type", ["numeric", "categorical"])
def test_variable_length_input_and_unseen_categorical_unknown(extension_type):
    model, descriptors = make_model(value_masking_enabled=False)
    model.eval()
    batch = append_extension(make_batch(descriptors=descriptors), descriptors, extension_type)
    if extension_type == "categorical":
        # An unseen categorical feature has no source vocabulary: only UNKNOWN is valid.
        batch["categorical_values"][:, -1] = 0
    output = model(**batch)
    assert output["token_embeddings"].shape == (4, 79, 128)
    assert torch.isfinite(output["Z_d"]).all()
    if extension_type == "categorical":
        expected = output["token_embeddings"][:, -1]
        assert torch.isfinite(expected).all()
        model.train()
        training_output = model(**batch)
        (training_output["Z_s"].square().mean() + training_output["Z_d"].square().mean()).backward()
        assert model.unknown_category_embedding.grad is not None
        assert torch.isfinite(model.unknown_category_embedding.grad).all()
        assert torch.count_nonzero(model.unknown_category_embedding.grad) > 0


def test_synthetic_onboarding_path_handles_reordered_and_new_schema():
    model, descriptors = make_model(value_masking_enabled=False)
    frame = pd.DataFrame({
        "new_category": ["unseen-A", "unseen-B"],
        "extra_bytes": [17.0, 29.0],
        "all_nan": [float("nan"), float("nan")],
        "infinite_measure": [float("inf"), float("-inf")],
    })
    # A different order must preserve each frame column's descriptor/value pairing.
    frame = frame[["infinite_measure", "new_category", "all_nan", "extra_bytes"]]
    batch = build_unseen_model_inputs(frame, descriptors, model.feature_identity_encoder)
    assert batch["onboarded_feature_names"] == list(frame.columns)
    assert batch["source_feature_indices"].tolist() == [-1] * 82
    assert [d["original_feature_name"] for d in batch["feature_descriptors"][-4:]] == list(frame.columns)
    assert batch["numeric_token_mask"][-4:].tolist() == [True, False, True, True]
    assert batch["categorical_token_mask"][-4:].tolist() == [False, True, False, False]
    assert np.equal(batch["M_avail"][:, 78:], 1).all()
    assert batch["M_miss"][:, 78:].tolist() == [[1, 0, 1, 0], [1, 0, 1, 0]]
    assert batch["categorical_values"][:, 79].tolist() == [0, 0]
    assert batch["numeric_values"][:, 81].tolist() == [-1.0, 1.0]
    stats = batch["unseen_numeric_normalization"]["statistics"]["extra_bytes"]
    assert stats["median"] == 23.0 and stats["iqr"] == 6.0
    identities = model.feature_identity_encoder.encode(batch["feature_descriptors"][-4:])
    assert identities.shape == (4, 128) and torch.isfinite(identities).all()
    model.eval()
    model_batch = {
        key: value for key, value in batch.items()
        if key not in {"onboarded_feature_names", "unseen_numeric_normalization"}
    }
    with torch.no_grad():
        output = model(**model_batch)
    assert output["token_embeddings"].shape == (2, 82, 128)
    assert torch.isfinite(output["token_embeddings"]).all()
    assert torch.isfinite(output["Z_s"]).all() and torch.isfinite(output["Z_d"]).all()


def test_unseen_normalization_ignores_labels_and_is_reproducible():
    model, descriptors = make_model(value_masking_enabled=False)
    features = pd.DataFrame({"bytes_new": [2.0, 4.0, 10.0], "const": [7.0, 7.0, 7.0]})
    first = build_unseen_model_inputs(
        features.assign(attack_cat=["a", "b", "c"], label=[0, 1, 0]),
        descriptors,
        model.feature_identity_encoder,
        label_columns=("attack_cat", "label"),
    )
    changed_labels = build_unseen_model_inputs(
        features.assign(attack_cat=["changed", "changed", "changed"], label=[99, 99, 99]),
        descriptors,
        model.feature_identity_encoder,
        label_columns=("attack_cat", "label"),
    )
    without_labels = build_unseen_model_inputs(
        features, descriptors, model.feature_identity_encoder,
        label_columns=("attack_cat", "label"),
    )
    assert np.array_equal(first["numeric_values"], changed_labels["numeric_values"])
    assert np.array_equal(first["numeric_values"], without_labels["numeric_values"])
    assert first["unseen_numeric_normalization"] == changed_labels["unseen_numeric_normalization"]
    assert first["unseen_numeric_normalization"] == without_labels["unseen_numeric_normalization"]
    assert first["onboarded_feature_names"] == ["bytes_new", "const"]
    assert first["unseen_numeric_normalization"]["fit_scope"] == (
        "unseen_unlabeled_available_finite_values_only"
    )
    assert np.array_equal(first["numeric_values"], without_labels["numeric_values"])
    assert first["numeric_values"][:, 79].tolist() == [0.0, 0.0, 0.0]


def test_unseen_normalization_finite_missing_constant_and_unavailable_values():
    model, descriptors = make_model(value_masking_enabled=False)
    frame = pd.DataFrame({
        "measure": [1.0, np.nan, np.inf, -np.inf, 5.0],
        "all_missing": [np.nan, np.inf, -np.inf, np.nan, np.nan],
        "constant": [3.0, 3.0, 3.0, 3.0, 3.0],
    })
    batch = build_unseen_model_inputs(frame, descriptors, model.feature_identity_encoder)
    assert np.isfinite(batch["numeric_values"]).all()
    assert np.equal(batch["M_avail"][:, :78], 0).all()
    assert np.equal(batch["M_miss"][:, :78], 0).all()
    assert np.equal(batch["numeric_values"][:, :78], 0).all()
    assert np.equal(batch["M_avail"][:, 78:], 1).all()
    assert batch["M_miss"][:, 78:].tolist() == [
        [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0]
    ]
    assert batch["numeric_values"][:, 80].tolist() == [0.0] * 5
    stats = batch["unseen_numeric_normalization"]["statistics"]
    assert stats["all_missing"]["finite_count"] == 0
    assert stats["all_missing"]["scale"] == 1.0
    assert stats["constant"]["iqr"] == 0.0
    assert stats["constant"]["scale"] == 1.0


def test_unseen_normalization_does_not_refit_source_preprocessor():
    model, descriptors = make_model(value_masking_enabled=False)
    source_preprocessor = NumericPreprocessor().fit(
        pd.DataFrame({"source_measure": [1.0, 3.0, 9.0]})
    )
    medians_before = source_preprocessor.medians.copy(deep=True)
    center_before = source_preprocessor.scaler.center_.copy()
    scale_before = source_preprocessor.scaler.scale_.copy()
    state_before = source_preprocessor.fitted

    build_unseen_model_inputs(
        pd.DataFrame({"new_measure": [100.0, 200.0, np.nan]}),
        descriptors,
        model.feature_identity_encoder,
    )

    pd.testing.assert_series_equal(source_preprocessor.medians, medians_before)
    np.testing.assert_array_equal(source_preprocessor.scaler.center_, center_before)
    np.testing.assert_array_equal(source_preprocessor.scaler.scale_, scale_before)
    assert source_preprocessor.fitted is state_before is True
    model.eval()
    batch = append_extension(make_batch(descriptors=descriptors), descriptors, "categorical")
    batch["source_feature_indices"].fill_(-1)
    batch["categorical_values"].zero_()
    batch["numeric_values"][:, 10] = float("nan")
    batch["numeric_values"][:, 11] = float("inf")
    batch["M_miss"][:, 10:12] = 1
    batch["is_unseen"] = True
    with torch.no_grad():
        output = model(**batch)
    assert torch.isfinite(output["token_embeddings"]).all()
    assert torch.isfinite(output["Z_s"]).all()
    assert torch.isfinite(output["Z_d"]).all()
    assert output["M_avail"][:, 10:12].eq(1).all()
    assert output["M_miss"][:, 10:12].eq(1).all()


def test_padding_invariance_and_unavailable_token_exclusion():
    model, descriptors = make_model(value_masking_enabled=False)
    model.eval()
    base = make_batch(descriptors=descriptors)
    base_output = model(**base)

    padded = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in base.items()
    }
    padded["numeric_values"] = torch.cat([padded["numeric_values"], torch.full((4, 1), 12345.0)], 1)
    padded["categorical_values"] = torch.cat([padded["categorical_values"], torch.ones((4, 1), dtype=torch.long)], 1)
    padded["M_avail"] = torch.cat([padded["M_avail"], torch.zeros((4, 1))], 1)
    padded["M_miss"] = torch.cat([padded["M_miss"], torch.zeros((4, 1))], 1)
    padded["numeric_token_mask"] = torch.cat([padded["numeric_token_mask"], torch.tensor([True])])
    padded["categorical_token_mask"] = torch.cat([padded["categorical_token_mask"], torch.tensor([False])])
    padded["source_feature_indices"] = torch.cat([torch.arange(78), torch.tensor([-1])])
    padded["feature_descriptors"] = descriptors + [{
        "original_feature_name": "padding", "level": 3, "data_type": "continuous",
        "semantic_group": None, "statistics": {"skewness": 0.0},
    }]
    pad_output = model(**padded)
    assert torch.allclose(base_output["Z_s"], pad_output["Z_s"], atol=1e-6)
    assert torch.allclose(base_output["Z_d"], pad_output["Z_d"], atol=1e-6)

    unavailable = make_batch(descriptors=descriptors)
    unavailable["M_avail"][:, 10] = 0
    first = model(**unavailable)
    unavailable["numeric_values"][:, 10] = -99999.0
    second = model(**unavailable)
    assert torch.allclose(first["Z_s"], second["Z_s"], atol=1e-6)
    assert torch.allclose(first["Z_d"], second["Z_d"], atol=1e-6)


def test_5b_receives_pooled_not_shared():
    """Stage 5B must receive the Stage 4-derived pooled tensor, not Z_s.

    Replacing shared_encoder changes Z_s (shared) but must not change Z_d,
    because Z_d is computed from pooled directly.
    """
    model, descriptors = make_model(value_masking_enabled=False)
    model.eval()
    batch = make_batch(descriptors=descriptors)

    with torch.no_grad():
        baseline = model(**batch)

    # Replace shared_encoder with a zero-producing encoder.
    # This changes shared/Z_s but not pooled, which is computed before
    # shared_encoder is applied.
    wrong_encoder = torch.nn.Sequential(torch.nn.Linear(128, 128))
    with torch.no_grad():
        wrong_encoder[0].weight.zero_()
        wrong_encoder[0].bias.zero_()
    model.shared_encoder = wrong_encoder.to(model.source_identity_base.device)

    with torch.no_grad():
        modified = model(**batch)

    # Z_s must change because it flows through shared_encoder.
    assert not torch.allclose(baseline["Z_s"], modified["Z_s"], atol=1e-6)
    # Z_d must NOT change because it flows from pooled, not shared.
    assert torch.allclose(baseline["Z_d"], modified["Z_d"], atol=1e-6)
    # Output shapes are preserved.
    assert baseline["Z_s"].shape == modified["Z_s"].shape == (4, 128)
    assert baseline["Z_d"].shape == modified["Z_d"].shape == (4, 128)


def test_all_unavailable_input_is_stable():
    model, descriptors = make_model(value_masking_enabled=False)
    model.eval()
    batch = make_batch(descriptors=descriptors)
    batch["M_avail"].zero_()
    batch["M_miss"].zero_()
    output = model(**batch)
    assert torch.isfinite(output["Z_s"]).all()
    assert torch.isfinite(output["Z_d"]).all()


def test_descriptor_identity_and_reviewed_group_embedding():
    model, descriptors = make_model(value_masking_enabled=False)
    unseen = {
        "original_feature_name": "flag_like",
        "description": "connection status flag indicating connection state",
        "data_type": "symbolic",
        "statistics": {"skewness": 0.0},
    }
    mapped = model.feature_identity_encoder.map_unseen_descriptor(
        unseen, descriptors, similarity_threshold=0.0
    )
    assert mapped["level"] == 2
    assert mapped["relationship"] == "group_only"
    assert mapped["semantic_id"] is None
    assert mapped["review_status"] == "approved"
    assert mapped["semantic_group"] in model.group_to_index
    generated = model.feature_identity_encoder.encode([mapped])
    assert generated.shape == (1, 128)
    assert torch.isfinite(generated).all()
    assert not torch.allclose(generated[0], model.feature_identities[0])

    batch = make_batch(descriptors=descriptors)
    batch["feature_descriptors"] = descriptors.copy()
    batch["feature_descriptors"][6] = mapped
    model.eval()
    output = model(**batch)
    assert output["token_embeddings"].shape == (4, 78, 128)
    assert model.group_embeddings.weight.grad is None
    output["Z_s"].sum().backward()
    assert model.group_embeddings.weight.grad is not None
    assert torch.isfinite(model.group_embeddings.weight.grad).all()


def test_training_only_random_value_masking_and_missing_embedding_gradient():
    model, descriptors = make_model(value_masking_rate=0.10, value_masking_enabled=True)
    model.train()
    batch = make_batch(batch_size=64, descriptors=descriptors)
    output = model(**batch)
    assert output["value_mask_applied"].any()
    assert torch.equal(output["M_avail"], batch["M_avail"])
    assert torch.equal(output["M_miss"].bool(), output["value_mask_applied"])
    loss = output["Z_s"].square().mean() + output["Z_d"].square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.missing_value_embedding.grad is not None
    assert torch.isfinite(model.missing_value_embedding.grad).all()
    assert torch.count_nonzero(model.missing_value_embedding.grad) > 0
    assert model.feature_identity_encoder.projection[0].weight.grad is not None
    assert torch.isfinite(model.feature_identity_encoder.projection[0].weight.grad).all()

    model.eval()
    eval_output = model(**batch)
    assert not eval_output["value_mask_applied"].any()


def test_structural_missing_embedding_receives_gradients():
    model, descriptors = make_model(value_masking_enabled=False)
    model.train()
    batch = make_batch(descriptors=descriptors)
    batch["M_avail"][:, 10] = 0
    output = model(**batch)
    loss = output["Z_s"].square().mean() + output["Z_d"].square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    gradient = model.unavailable_embedding.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_feature_set_dropout_is_training_only_and_configurable():
    descriptors = source_descriptors()
    model, _ = make_model(
        value_masking_enabled=False,
        feature_set_dropout_rate=1.0,
        feature_set_dropout_all_probability=1.0,
    )
    batch = make_batch(descriptors=descriptors)
    model.train()
    output = model(**batch)
    specific = torch.tensor([d["level"] == 3 for d in descriptors])
    assert output["feature_dropout_applied"][:, specific].any()
    assert torch.all(output["M_avail"][:, specific] == 0)
    assert torch.all(output["M_avail"][:, :5] == 1)
    assert torch.all(output["M_miss"] == 0)
    model.eval()
    output = model(**batch)
    assert not output["feature_dropout_applied"].any()


def test_representation_augmentation_policy_controls_model_switches():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    with (root / "config" / "preprocessing_policy.yaml").open(encoding="utf-8") as file:
        policy = yaml.safe_load(file)["preprocessing_policy"]["representation_augmentation"]
    model, _ = make_model(augmentation_config=policy)
    assert model.value_masking_enabled is True
    assert model.value_masking_rate == 0.05
    assert model.feature_set_dropout_rate == 0.10
    assert model.feature_set_dropout_all_probability == 0.05


def test_unseen_context_adapter_is_shared_and_requires_no_grad_inference():
    model, descriptors = make_model(value_masking_enabled=False)
    first_batch = append_extension(make_batch(descriptors=descriptors), descriptors, "numeric")
    first_batch["source_feature_indices"].fill_(-1)
    first_batch["categorical_values"].zero_()
    first_batch["is_unseen"] = True
    model.eval()
    with pytest.raises(ValueError, match="torch.no_grad"):
        model(**first_batch)
    with torch.no_grad():
        first = model(**first_batch)
        second_batch = dict(first_batch)
        second_batch["M_avail"] = first_batch["M_avail"].clone()
        second_batch["M_avail"][:, -1] = 0
        second = model(**second_batch)
    assert first["Z_d"].shape == second["Z_d"].shape == (4, 128)
    assert not torch.allclose(first["feature_context"], second["feature_context"])
    assert not any(name.startswith("dataset_adapters") for name, _ in model.named_parameters())


def test_unseen_labels_and_test_fields_are_not_used_for_fitting():
    model, descriptors = make_model(value_masking_enabled=False)
    encoder = model.feature_identity_encoder
    unseen = {
        "original_feature_name": "new_counter",
        "description": "number of recent packet events",
        "data_type": "integer",
        "statistics": {"skewness": 1.2, "zero_fraction": 0.1},
    }
    with_label = dict(
        unseen,
        label="must_not_be_read",
        test_partition_values=[999999.0],
    )
    before = [p.detach().clone() for p in encoder.parameters()]
    left = encoder.encode([unseen])
    right = encoder.encode([with_label])
    assert torch.equal(left, right)
    assert all(torch.equal(old, new) for old, new in zip(before, encoder.parameters()))
    mapped = encoder.map_unseen_descriptor(unseen, descriptors, similarity_threshold=1.0)
    assert mapped["level"] == 3
    assert mapped["relationship"] == "dataset_specific"

    inference_batch = append_extension(make_batch(descriptors=descriptors), descriptors, "numeric")
    inference_batch["source_feature_indices"].fill_(-1)
    inference_batch["categorical_values"].zero_()
    inference_batch["feature_descriptors"][-1] = unseen
    inference_batch["is_unseen"] = True
    model.eval()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    with torch.no_grad():
        model(**inference_batch)
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_unseen_inference_rejects_training_mode():
    model, descriptors = make_model(value_masking_enabled=False)
    batch = append_extension(make_batch(descriptors=descriptors), descriptors, "numeric")
    batch["source_feature_indices"].fill_(-1)
    batch["categorical_values"].zero_()
    batch["is_unseen"] = True
    model.train()
    with torch.no_grad(), pytest.raises(ValueError, match="eval mode"):
        model(**batch)
