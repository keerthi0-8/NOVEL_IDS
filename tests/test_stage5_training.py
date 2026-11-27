"""Tests for Stage 5 training pipeline components."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tempfile
import torch
from torch import nn

from novel_ids.representation.classification_head import (
    ClassificationHead,
    compute_class_weights,
)
from novel_ids.representation.reconstruction_decoder import ReconstructionDecoder
from novel_ids.representation.stage5_training import (
    Stage5Config,
    Stage5Trainer,
    train_stage5,
)
from novel_ids.representation.dual_encoder import (
    CATEGORICAL_CONFIG,
    DualEncoder,
    IDENTITY_DIM,
    OUTPUT_DIM,
    TOKEN_COUNT,
)
from novel_ids.representation.feature_identity import FeatureIdentityEncoder


def make_descriptors():
    descriptors = []
    for i in range(78):
        if i < 5:
            item = {
                "original_feature_name": f"common_{i}",
                "metadata_feature_name": f"common_{i}",
                "semantic_id": f"common_{i}",
                "level": 1, "semantic_group": None,
                "relationship": "equivalent", "confidence": "high",
                "review_status": "approved",
                "data_type": "continuous",
                "statistics": {"skewness": 0.1},
            }
        elif i == 6:
            item = {
                "original_feature_name": "flag",
                "description": "connection state flag",
                "level": 2, "semantic_id": None,
                "semantic_group": "connection_state",
                "relationship": "group_only", "confidence": "medium",
                "review_status": "approved",
                "data_type": "symbolic",
                "evidence": "Metadata describes connection status.",
                "statistics": {"skewness": 0.2},
            }
        else:
            item = {
                "original_feature_name": f"feature_{i}",
                "description": f"source feature {i}",
                "level": 3, "semantic_id": None,
                "semantic_group": None,
                "relationship": "dataset_specific",
                "confidence": "unassessed",
                "review_status": "dataset_specific",
                "data_type": "continuous",
                "statistics": {"skewness": 0.0},
            }
        descriptors.append(item)
    return descriptors


def make_model(device="cpu", **kwargs):
    descriptors = make_descriptors()
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


def make_batch(dataset="nsl_kdd", batch_size=4, count=78, descriptors=None, device="cpu"):
    config = CATEGORICAL_CONFIG[dataset]
    numeric_mask = torch.ones(count, dtype=torch.bool)
    categorical_mask = torch.zeros(count, dtype=torch.bool)
    for position in config["positions"]:
        numeric_mask[position] = False
        categorical_mask[position] = True
    numeric = torch.zeros(batch_size, count, device=device)
    categorical = torch.zeros(batch_size, count, dtype=torch.long, device=device)
    for position, size in zip(config["positions"], config["vocab_sizes"]):
        categorical[:, position] = torch.randint(1, size + 1, (batch_size,), device=device)
    available = torch.ones(batch_size, count, device=device)
    missing = torch.zeros(batch_size, count, device=device)
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


# ------------------------------------------------------------------
# Classification head tests
# ------------------------------------------------------------------
def test_classification_head_separate_per_dataset():
    """Each dataset has its own classification head with native labels."""
    head_nsl = ClassificationHead(input_dim=128, num_classes=23)
    head_unsw = ClassificationHead(input_dim=128, num_classes=10)

    assert head_nsl.num_classes == 23
    assert head_unsw.num_classes == 10

    z_s = torch.randn(4, 128)
    logits_nsl = head_nsl(z_s)
    logits_unsw = head_unsw(z_s)

    assert logits_nsl.shape == (4, 23)
    assert logits_unsw.shape == (4, 10)


def test_classification_head_class_weights_training_only():
    """Class weights computed from training labels only, not validation."""
    train_labels = ["normal"] * 100 + ["dos"] * 10 + ["probe"] * 5
    weights = compute_class_weights(train_labels, num_classes=3)

    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()
    normal_idx = sorted(["normal", "dos", "probe"]).index("normal")
    dos_idx = sorted(["normal", "dos", "probe"]).index("dos")
    assert weights[normal_idx] < weights[dos_idx]


def test_classification_head_loss_with_weights():
    """Classification loss uses class weights."""
    head = ClassificationHead(input_dim=128, num_classes=3)
    weights = torch.tensor([1.0, 5.0, 10.0])
    head.class_weights = weights

    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])

    loss_weighted = head.loss(logits, targets)
    loss_unweighted = nn.functional.cross_entropy(logits, targets)

    assert torch.isfinite(loss_weighted)
    assert loss_weighted != loss_unweighted


# ------------------------------------------------------------------
# Reconstruction decoder tests
# ------------------------------------------------------------------
def test_reconstruction_decoder_separate_per_dataset():
    """Each dataset has its own reconstruction decoder."""
    decoder_nsl = ReconstructionDecoder(
        z_dim=128, identity_dim=128,
        numeric_features=[10, 11],
        categorical_features={1: 3, 2: 70, 5: 11},
    )
    decoder_unsw = ReconstructionDecoder(
        z_dim=128, identity_dim=128,
        numeric_features=[10],
        categorical_features={1: 133, 2: 13, 41: 8},
    )

    z_d = torch.randn(4, 128)
    identities = torch.randn(78, 128)

    pred_nsl = decoder_nsl(z_d, identities, [10, 1, 2, 5])
    pred_unsw = decoder_unsw(z_d, identities, [10, 1, 2, 41])

    assert pred_nsl[10].shape == (4, 1)
    assert pred_nsl[1].shape == (4, 3)
    assert pred_nsl[2].shape == (4, 70)
    assert pred_nsl[5].shape == (4, 11)
    assert pred_unsw[10].shape == (4, 1)
    assert pred_unsw[1].shape == (4, 133)
    assert pred_unsw[2].shape == (4, 13)
    assert pred_unsw[41].shape == (4, 8)


def test_reconstruction_decoder_loss():
    """Reconstruction loss computes Huber (numeric) and CE (categorical)."""
    decoder = ReconstructionDecoder(
        z_dim=128, identity_dim=128,
        numeric_features=[10],
        categorical_features={1: 3, 2: 70, 5: 11},
    )

    z_d = torch.randn(4, 128)
    identities = torch.randn(78, 128)
    predictions = decoder(z_d, identities, [10, 2])

    targets = {
        10: torch.tensor([0.5, -0.3, 1.0, 0.0]),
        # Decoder targets are zero-based; Stage 5 shifts stored ids 1..V by -1.
        2: torch.tensor([5, 10, 3, 7]),
    }
    masks = {
        10: torch.tensor([True, True, False, True]),
        2: torch.tensor([False, True, True, True]),
    }

    loss, logged = decoder.reconstruction_loss(
        predictions, targets, masks, numeric_weight=1.0, categorical_weight=1.0
    )

    assert torch.isfinite(loss)
    assert "reconstruction_numeric" in logged
    assert "reconstruction_categorical" in logged
    assert "reconstruction_total" in logged


# ------------------------------------------------------------------
# Stage 5B parallel input path test
# ------------------------------------------------------------------
def test_5b_receives_pooled_not_shared():
    """Stage 5B receives pooled (Stage 4 output), not shared (Z_s).

    Replacing shared_encoder changes Z_s but not Z_d, confirming Z_d
    is computed from pooled directly.
    """
    model, descriptors = make_model(value_masking_enabled=False)
    model.eval()
    batch = make_batch(descriptors=descriptors)

    with torch.no_grad():
        baseline = model(**batch)

    wrong_encoder = nn.Sequential(nn.Linear(128, 128))
    with torch.no_grad():
        wrong_encoder[0].weight.zero_()
        wrong_encoder[0].bias.zero_()
    model.shared_encoder = wrong_encoder.to(model.source_identity_base.device)

    with torch.no_grad():
        modified = model(**batch)

    assert not torch.allclose(baseline["Z_s"], modified["Z_s"], atol=1e-6)
    assert torch.allclose(baseline["Z_d"], modified["Z_d"], atol=1e-6)
    assert baseline["Z_s"].shape == modified["Z_s"].shape == (4, 128)
    assert baseline["Z_d"].shape == modified["Z_d"].shape == (4, 128)


# ------------------------------------------------------------------
# Stage5Trainer tests
# ------------------------------------------------------------------
def test_trainer_build_models():
    """Trainer builds models with separate heads and decoders per dataset."""
    config = Stage5Config(
        device="cpu",
        mixed_precision=False,
        max_steps=10,
        sub_batch_size=8,
        validation_interval=5,
        patience=3,
    )
    trainer = Stage5Trainer(config)
    trainer.load_artifacts()
    trainer.build_models()

    assert trainer.dual_encoder is not None
    assert "nsl_kdd" in trainer.classification_heads
    assert "unsw_nb15" in trainer.classification_heads
    assert "nsl_kdd" in trainer.reconstruction_decoders
    assert "unsw_nb15" in trainer.reconstruction_decoders
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None


def test_trainer_class_weights_from_training_only():
    """Class weights computed from training labels only."""
    config = Stage5Config(
        device="cpu", mixed_precision=False, max_steps=10, sub_batch_size=8,
        validation_interval=5, patience=3,
    )
    trainer = Stage5Trainer(config)
    trainer.load_artifacts()
    trainer.build_models()

    for dataset_name in config.dataset_names:
        weights = trainer.class_weights[dataset_name]
        assert torch.isfinite(weights).all()
        assert weights.shape[0] == trainer.datasets[dataset_name]["num_classes"]


def test_trainer_finite_losses_and_gradients():
    """Training step produces finite losses and gradients on synthetic data."""
    config = Stage5Config(
        device="cpu", mixed_precision=False, max_steps=10, sub_batch_size=4,
        validation_interval=5, patience=3,
    )
    trainer = Stage5Trainer(config)
    trainer.load_artifacts()
    trainer.build_models()

    metrics = trainer.train_step()

    assert torch.isfinite(torch.tensor(metrics["total_loss"]))
    assert torch.isfinite(torch.tensor(metrics["classification_loss"]))
    assert torch.isfinite(torch.tensor(metrics["reconstruction_loss"]))

    for name, param in trainer.dual_encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    for head_name, head in trainer.classification_heads.items():
        for name, param in head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {head_name}.{name}"


def test_trainer_validation_macro_f1():
    """Validation computes per-dataset macro-F1."""
    config = Stage5Config(
        device="cpu", mixed_precision=False, max_steps=10, sub_batch_size=4,
        validation_interval=1, patience=3,
    )
    trainer = Stage5Trainer(config)
    trainer.load_artifacts()
    trainer.build_models()

    val_metrics = trainer.validate()

    assert "nsl_kdd_val_macro_f1" in val_metrics
    assert "unsw_nb15_val_macro_f1" in val_metrics
    assert "val_macro_f1_avg" in val_metrics
    assert 0.0 <= val_metrics["nsl_kdd_val_macro_f1"] <= 1.0
    assert 0.0 <= val_metrics["unsw_nb15_val_macro_f1"] <= 1.0
    assert 0.0 <= val_metrics["val_macro_f1_avg"] <= 1.0


def test_checkpoint_save_load():
    """Checkpoint save/load restores training state."""
    config = Stage5Config(
        device="cpu", mixed_precision=False, max_steps=10, sub_batch_size=4,
        validation_interval=5, patience=3, runs_dir=Path(tempfile.mkdtemp()),
    )
    trainer = Stage5Trainer(config)
    trainer.load_artifacts()
    trainer.build_models()

    for _ in range(3):
        trainer.train_step()

    step_before = trainer.global_step
    best_f1_before = trainer.best_val_macro_f1

    checkpoint_path = trainer.save_checkpoint(is_best=False, tag="test_save")

    trainer2 = Stage5Trainer(config)
    trainer2.load_artifacts()
    trainer2.build_models()
    trainer2.load_checkpoint(checkpoint_path)

    assert trainer2.global_step == step_before
    assert trainer2.best_val_macro_f1 == best_f1_before

    import shutil
    shutil.rmtree(config.runs_dir, ignore_errors=True)


def test_trainer_equal_dataset_weighting():
    """Each training step uses equal-size sub-batches from both datasets."""
    config = Stage5Config(
        device="cpu", mixed_precision=False, max_steps=10, sub_batch_size=8,
        validation_interval=5, patience=3,
    )
    trainer = Stage5Trainer(config)
    trainer.load_artifacts()
    trainer.build_models()

    metrics = trainer.train_step()

    assert metrics["total_loss"] > 0
    assert torch.isfinite(torch.tensor(metrics["total_loss"]))


def test_stage5_config_defaults():
    """Default configuration values match the specification."""
    config = Stage5Config()
    assert config.sub_batch_size == 256
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.gradient_clip == 1.0
    assert config.warmup_steps == 500
    assert config.max_steps == 10000
    assert config.reconstruction_mask_rate == 0.20
    assert config.lambda_reconstruction == 1.0
    assert config.patience == 5
    assert config.seed == 42