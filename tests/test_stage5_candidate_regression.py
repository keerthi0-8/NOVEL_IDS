from pathlib import Path
import tempfile
import numpy as np
import torch

from novel_ids.representation import stage5_training_candidate as stage5


def test_lambda_ramp_starts_zero_and_reaches_target():
    cfg = stage5.Stage5Config(
        device="cpu", mixed_precision=False, max_steps=100,
        lambda_reconstruction=0.7, lambda_ramp_steps=10,
        runs_dir=Path(tempfile.mkdtemp()),
    )
    trainer = stage5.Stage5Trainer(cfg)
    assert trainer._current_reconstruction_lambda() == 0.0
    trainer.global_step = 9
    assert abs(trainer._current_reconstruction_lambda() - 0.7) < 1e-9
    trainer.global_step = 20
    assert abs(trainer._current_reconstruction_lambda() - 0.7) < 1e-9


def test_reconstruction_mask_uses_selected_batch_availability_and_missingness():
    cfg = stage5.Stage5Config(
        device="cpu", mixed_precision=False, reconstruction_mask_rate=1.0,
        runs_dir=Path(tempfile.mkdtemp()),
    )
    trainer = stage5.Stage5Trainer(cfg)
    avail = torch.zeros(2, 78)
    miss = torch.zeros(2, 78)
    cats = torch.ones(2, 78, dtype=torch.long)
    avail[0, [5, 6, 7]] = 1
    avail[1, [6, 7]] = 1
    miss[0, 6] = 1
    miss[1, 7] = 1
    cats[0, 6] = 2
    cats[1, 6] = 0
    trainer.datasets["nsl_kdd"] = {
        "dataset_specific_positions": [5, 6, 7],
        "categorical_positions": [6],
    }
    hidden, visible = trainer._make_reconstruction_mask(
        "nsl_kdd", np.array([10, 11]), avail, miss, cats, validation=False
    )
    assert hidden[5].tolist() == [True, False]
    assert hidden[6].tolist() == [False, False]
    assert hidden[7].tolist() == [True, False]
    assert not any(mask.any() for mask in visible.values())


def test_validation_reconstruction_mask_is_fixed_across_calls():
    cfg = stage5.Stage5Config(
        device="cpu", mixed_precision=False, reconstruction_mask_rate=0.5,
        runs_dir=Path(tempfile.mkdtemp()),
    )
    trainer = stage5.Stage5Trainer(cfg)
    avail = np.ones((6, 78), dtype=np.float32)
    miss = np.zeros((6, 78), dtype=np.float32)
    cats = np.ones((6, 78), dtype=np.int64)
    trainer.datasets["nsl_kdd"] = {
        "dataset_specific_positions": [5, 6, 7],
        "categorical_positions": [6],
        "val_M_avail": avail,
        "val_M_miss": miss,
        "val_categorical": cats,
    }
    args = (
        "nsl_kdd", np.array([1, 2, 3]),
        torch.tensor(avail[[1, 2, 3]]),
        torch.tensor(miss[[1, 2, 3]]),
        torch.tensor(cats[[1, 2, 3]]),
    )
    first, _ = trainer._make_reconstruction_mask(*args, validation=True)
    second, _ = trainer._make_reconstruction_mask(*args, validation=True)
    for pos in [5, 6, 7]:
        assert torch.equal(first[pos], second[pos])


def test_early_stopping_waits_for_minimum_ten_evaluations():
    cfg = stage5.Stage5Config(
        device="cpu", mixed_precision=False, max_steps=20,
        validation_interval=1, min_validation_evaluations=10,
        max_consecutive_degrading_evaluations=3,
        runs_dir=Path(tempfile.mkdtemp()),
    )
    trainer = stage5.Stage5Trainer(cfg)
    sequence = [0.80] * 7 + [0.79, 0.78, 0.77]
    cursor = {"i": 0}
    def fake_train_step():
        trainer.global_step += 1
        return {"step": float(trainer.global_step), "classification_loss": 1.0,
                "reconstruction_loss": 0.1, "total_loss": 1.1}
    def fake_validate():
        value = sequence[cursor["i"]]
        cursor["i"] += 1
        return {
            "nsl_kdd_val_macro_f1": value,
            "unsw_nb15_val_macro_f1": value,
            "val_macro_f1_avg": value,
        }
    trainer.train_step = fake_train_step
    trainer.validate = fake_validate
    trainer.save_checkpoint = lambda **kwargs: None
    result = trainer.fit()
    assert result["validation_evaluations"] == 10
    assert result["global_step"] == 10
    assert result["consecutive_degrading_evaluations"] == 3


def test_two_phase_lambda_selection_uses_paired_three_seed_rule(monkeypatch, tmp_path):
    phase_b_scores = {
        0.0: {42: (0.60, 0.60), 43: (0.61, 0.60), 44: (0.59, 0.60)},
        0.1: {42: (0.82, 0.82), 43: (0.83, 0.81), 44: (0.81, 0.83)},
        0.5: {42: (0.90, 0.50), 43: (0.90, 0.50), 44: (0.90, 0.50)},
    }
    class FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg
        def fit(self):
            lam = float(self.cfg.lambda_reconstruction)
            if "/phaseA_" in self.cfg.experiment_name:
                pair = {
                    0.0: (0.60, 0.60),
                    0.1: (0.80, 0.80),
                    0.5: (0.90, 0.50),
                    1.0: (0.55, 0.55),
                }[lam]
            else:
                pair = phase_b_scores[lam][self.cfg.seed]
            return {
                "best_val_macro_f1": float(np.mean(pair)),
                "best_dataset_macro_f1": {
                    "nsl_kdd": pair[0], "unsw_nb15": pair[1]
                },
                "global_step": 100,
            }
    calls = []
    def fake_train_stage5(cfg, identity_vectors_path=None):
        calls.append((cfg.lambda_reconstruction, cfg.seed, cfg.experiment_name))
        return FakeTrainer(cfg)
    monkeypatch.setattr(stage5, "train_stage5", fake_train_stage5)
    cfg = stage5.Stage5Config(
        device="cpu", mixed_precision=False,
        lambda_candidates=(0.0, 0.1, 0.5, 1.0),
        runs_dir=tmp_path, experiment_name="selection_smoke",
    )
    report = stage5.run_stage5_lambda_selection(cfg)
    assert report["phase_b_candidates"] == [0.0, 0.1, 0.5]
    assert report["phase_b_seeds"] == [42, 43, 44]
    assert report["phase_b_reused_phase_a_seed"] is True
    assert len(calls) == 10  # Four Phase A runs plus six new Phase B runs.
    assert report["candidate_comparison"]["0.5"]["rejected"] is True
    assert report["selected_lambda"] == 0.1
    assert (tmp_path / "selection_smoke" / "lambda_selection.json").exists()


def test_dataset_specific_positions_and_vocab_match_approved_metadata(tmp_path):
    cfg = stage5.Stage5Config(
        device="cpu", mixed_precision=False, runs_dir=tmp_path
    )
    trainer = stage5.Stage5Trainer(cfg)
    trainer.load_artifacts()
    nsl = trainer.datasets["nsl_kdd"]
    unsw = trainer.datasets["unsw_nb15"]
    assert len(nsl["dataset_specific_positions"]) == 36
    assert len(unsw["dataset_specific_positions"]) == 37
    assert nsl["cat_position_to_vocab"] == {1: 3, 2: 70, 5: 11}
    assert unsw["cat_position_to_vocab"] == {1: 133, 2: 13, 41: 8}


def test_checkpoint_reload_preserves_validation_metrics_without_training(tmp_path):
    cfg = stage5.Stage5Config(
        device="cpu", mixed_precision=False, max_steps=10,
        sub_batch_size=4, runs_dir=tmp_path, experiment_name="roundtrip",
    )
    first = stage5.Stage5Trainer(cfg)
    first.load_artifacts()
    first.build_models()
    metrics_before = first.validate()
    first.best_val_macro_f1 = metrics_before["val_macro_f1_avg"]
    first.best_checkpoint_metrics = {
        "val_macro_f1_avg": metrics_before["val_macro_f1_avg"],
        "nsl_kdd_val_macro_f1": metrics_before["nsl_kdd_val_macro_f1"],
        "unsw_nb15_val_macro_f1": metrics_before["unsw_nb15_val_macro_f1"],
    }
    first.best_checkpoint_step = 0
    checkpoint = first.save_checkpoint(is_best=True, tag="roundtrip")

    second = stage5.Stage5Trainer(cfg)
    second.load_artifacts()
    second.build_models()
    second.load_checkpoint(checkpoint)
    metrics_after = second.validate()
    for key in (
        "val_macro_f1_avg",
        "nsl_kdd_val_macro_f1",
        "unsw_nb15_val_macro_f1",
        "nsl_kdd_val_recon_loss",
        "unsw_nb15_val_recon_loss",
    ):
        assert metrics_after[key] == metrics_before[key]


def test_source_batch_forward_backward_has_finite_gradients_without_optimizer_step(tmp_path):
    cfg = stage5.Stage5Config(
        device="cpu", mixed_precision=False, max_steps=100,
        lambda_ramp_steps=2, reconstruction_mask_rate=0.5,
        runs_dir=tmp_path, experiment_name="gradient_smoke",
    )
    trainer = stage5.Stage5Trainer(cfg)
    trainer.load_artifacts()
    trainer.build_models()
    trainer.global_step = 1  # End of the two-step lambda ramp; no optimizer step is called.
    trainer.optimizer.zero_grad(set_to_none=True)
    for dataset_name in cfg.dataset_names:
        indices = np.arange(8)
        result = trainer._forward_batch(dataset_name, indices, training=True)
        assert torch.isfinite(result["loss"])
        assert torch.isfinite(result["classification_loss"])
        assert torch.isfinite(result["reconstruction_loss"])
        (0.5 * result["loss"]).backward()
    assert trainer.global_step == 1
    for name, module in (
        ("shared encoder", trainer.dual_encoder),
        ("classification heads", torch.nn.ModuleList(list(trainer.classification_heads.values()))),
        ("reconstruction decoders", torch.nn.ModuleList(list(trainer.reconstruction_decoders.values()))),
    ):
        grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        assert grads, f"No gradients reached {name}"
        assert all(torch.isfinite(grad).all() for grad in grads), f"Non-finite gradient in {name}"
