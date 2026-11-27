import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from novel_ids.residual.stage6_training import (
    Stage6Trainer,
    Stage6TrainingConfig,
    screening_rejection_reasons,
)


class SyntheticFrozenStage5(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(901)
        self.register_buffer("shared_map", torch.randn(78, 128, generator=generator) / 9.0)
        self.register_buffer("dataset_map", torch.randn(78, 128, generator=generator) / 9.0)

    def forward(
        self, numeric_values, categorical_values, M_avail, M_miss,
        numeric_token_mask, categorical_token_mask, dataset, feature_descriptors,
    ):
        z_s = numeric_values @ self.shared_map
        z_d = numeric_values @ self.dataset_map
        return {"Z_s": z_s, "Z_d": z_d}


def _arrays(seed, n):
    rng = np.random.default_rng(seed)
    return {
        "numeric_values": rng.normal(size=(n, 78)).astype(np.float32),
        "categorical_values": np.zeros((n, 78), dtype=np.int64),
        "M_avail": np.ones((n, 78), dtype=np.float32),
        "M_miss": np.zeros((n, 78), dtype=np.float32),
        "numeric_token_mask": np.ones(78, dtype=bool),
        "categorical_token_mask": np.zeros(78, dtype=bool),
    }


def _datasets():
    return {
        "nsl_kdd": {"train": _arrays(1, 24), "validation": _arrays(2, 13)},
        "unsw_nb15": {"train": _arrays(3, 24), "validation": _arrays(4, 15)},
    }


def _config(seed=42):
    return Stage6TrainingConfig(
        residual_dim=64,
        lambda_red=0.1,
        variance_floor=1e-5,
        projection_mode="trainable",
        sub_batch_size=8,
        validation_batch_size=6,
        max_steps=4,
        validation_interval=2,
        early_stop_patience=2,
        seed=seed,
        device="cpu",
    )


def _state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _trainer(tmp_path, seed=42, name="run"):
    stage5 = SyntheticFrozenStage5()
    return Stage6Trainer(
        stage5=stage5,
        datasets=_datasets(),
        stage5_sha256="a" * 64,
        run_dir=tmp_path / name,
        config=_config(seed),
        device="cpu",
    )


def test_stage6_training_step_keeps_stage5_frozen_and_out_of_optimizer(tmp_path):
    trainer = _trainer(tmp_path)
    before = _state_hash(trainer.stage5)
    optimizer_ids = {
        id(p)
        for group in trainer.optimizer.param_groups
        for p in group["params"]
    }
    stage5_ids = {id(p) for p in trainer.stage5.parameters()}
    stage6_trainable_ids = {
        id(p) for p in trainer.branches.parameters() if p.requires_grad
    }
    assert optimizer_ids == stage6_trainable_ids
    assert optimizer_ids.isdisjoint(stage5_ids)

    metrics = trainer.train_step()
    after = _state_hash(trainer.stage5)
    assert before == after
    assert metrics["step"] == 1.0
    assert np.isfinite(metrics["combined_total"])
    assert all(p.grad is None for p in trainer.stage5.parameters())


def test_stage6_validation_is_label_free_and_reports_both_datasets(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.train_step()
    result = trainer.validate()
    assert set(result) == {"nsl_kdd", "unsw_nb15", "mean_validation_objective"}
    for name in ("nsl_kdd", "unsw_nb15"):
        assert result[name]["validation_rows"] > 0
        assert np.isfinite(result[name]["validation_objective"])
        assert np.isfinite(result[name]["reconstruction_skill"])
        assert result[name]["residual_std_min"] >= 0
        assert result[name]["effective_rank_participation"] >= 0


def test_stage6_checkpoint_records_stage5_provenance(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.train_step()
    path = trainer.save_checkpoint(best=True)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["stage5_checkpoint_sha256"] == "a" * 64
    assert checkpoint["stage"] == 6
    assert checkpoint["global_step"] == 1


def test_stage6_training_is_deterministic_for_fixed_seed(tmp_path):
    first = _trainer(tmp_path, seed=123, name="first")
    second = _trainer(tmp_path, seed=123, name="second")
    metrics_a = first.train_step()
    metrics_b = second.train_step()
    assert metrics_a["combined_total"] == metrics_b["combined_total"]
    for key, value in first.branches.state_dict().items():
        assert torch.equal(value, second.branches.state_dict()[key])


def test_fixed_projection_baseline_is_excluded_from_optimizer(tmp_path):
    stage5 = SyntheticFrozenStage5()
    config = _config()
    config = Stage6TrainingConfig(
        **{**config.__dict__, "projection_mode": "fixed_semi_orthogonal"}
    )
    trainer = Stage6Trainer(
        stage5=stage5,
        datasets=_datasets(),
        stage5_sha256="b" * 64,
        run_dir=tmp_path / "fixed",
        config=config,
        device="cpu",
    )
    optimizer_ids = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    for branch in trainer.branches.values():
        for parameter in branch.projection.parameters():
            assert not parameter.requires_grad
            assert id(parameter) not in optimizer_ids


def test_training_step_reports_separate_dataset_losses_and_equal_weighting(tmp_path):
    trainer = _trainer(tmp_path)
    metrics = trainer.train_step()
    expected = 0.5 * (
        metrics["nsl_kdd_total"] + metrics["unsw_nb15_total"]
    )
    assert np.isclose(metrics["combined_total"], expected, rtol=1e-6, atol=1e-7)
    assert "nsl_kdd_L_red" in metrics
    assert "unsw_nb15_L_red" in metrics


def test_screening_rejection_gates_collapse_skill_and_nonfinite():
    good = {
        "nsl_kdd": {
            "validation_objective": 0.2, "reconstruction_skill": 0.1,
            "residual_rms": 1.0, "residual_std_min": 0.2,
            "effective_rank_participation": 40.0, "near_zero_dimensions": 0,
        },
        "unsw_nb15": {
            "validation_objective": 0.3, "reconstruction_skill": 0.2,
            "residual_rms": 1.0, "residual_std_min": 0.2,
            "effective_rank_participation": 40.0, "near_zero_dimensions": 0,
        },
    }
    assert screening_rejection_reasons(good, 64) == []
    collapsed = {**good, "nsl_kdd": {
        **good["nsl_kdd"], "effective_rank_participation": 10.0
    }}
    assert any("collapse" in x for x in screening_rejection_reasons(collapsed, 64))
    floor_collapsed = {**good, "nsl_kdd": {
        **good["nsl_kdd"], "near_zero_dimensions": 17
    }}
    assert any("collapse" in x for x in screening_rejection_reasons(floor_collapsed, 64))
    floor_boundary = {**good, "nsl_kdd": {
        **good["nsl_kdd"], "near_zero_dimensions": 16
    }}
    assert not any("collapse" in x for x in screening_rejection_reasons(floor_boundary, 64))
    no_skill = {**good, "unsw_nb15": {
        **good["unsw_nb15"], "reconstruction_skill": 0.0
    }}
    assert any("not positive" in x for x in screening_rejection_reasons(no_skill, 64))
    nonfinite = {**good, "nsl_kdd": {
        **good["nsl_kdd"], "validation_objective": float("nan")
    }}
    assert any("non-finite" in x for x in screening_rejection_reasons(nonfinite, 64))


def test_stage6_rejects_test_or_label_fields_in_training_data_contract(tmp_path):
    data = _datasets()
    data["nsl_kdd"]["train"]["labels"] = np.zeros(24, dtype=np.int64)
    try:
        Stage6Trainer(
            stage5=SyntheticFrozenStage5(),
            datasets=data,
            stage5_sha256="c" * 64,
            run_dir=tmp_path / "reject_labels",
            config=_config(),
            device="cpu",
        )
    except ValueError as exc:
        assert "unexpected or missing input fields" in str(exc)
    else:
        raise AssertionError("Training data contract must reject label fields.")


def test_identity_projection_mode_requires_128d_and_is_accepted():
    config = Stage6TrainingConfig(
        residual_dim=128,
        lambda_red=0.1,
        variance_floor=1e-4,
        projection_mode="identity",
        sub_batch_size=8,
        validation_batch_size=6,
        max_steps=4,
        validation_interval=2,
        early_stop_patience=2,
        device="cpu",
    )
    assert config.projection_mode == "identity"
    with pytest.raises(ValueError, match="requires residual_dim=128"):
        Stage6TrainingConfig(
            residual_dim=64,
            lambda_red=0.1,
            variance_floor=1e-4,
            projection_mode="identity",
            device="cpu",
        )
