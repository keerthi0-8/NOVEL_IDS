import torch

from novel_ids.residual.stage6 import (
    Stage6ResidualBranch,
    information_reconstruction_loss,
    projection_geometry_loss,
)
from novel_ids.residual.stage6_decorr import residual_decorrelation_loss
from novel_ids.residual.stage6_training import Stage6TrainingConfig


def test_independent_dimensions_have_near_zero_decorrelation():
    torch.manual_seed(110)
    r = torch.randn(6000, 6)
    loss, stats = residual_decorrelation_loss(r, variance_floor=1e-4)
    assert stats["active_dimensions"] == 6
    assert loss.item() < 0.002


def test_strongly_correlated_dimensions_have_large_decorrelation_loss():
    torch.manual_seed(111)
    x = torch.randn(3000, 1)
    r = torch.cat([x, x + 0.01 * torch.randn_like(x), -2.0 * x + 0.02 * torch.randn_like(x)], dim=1)
    loss, stats = residual_decorrelation_loss(r, variance_floor=1e-4)
    assert stats["mean_abs_offdiag_correlation"] > 0.95
    assert loss.item() > 0.90


def test_decorrelation_is_invariant_to_nonzero_coordinate_rescaling():
    torch.manual_seed(112)
    x = torch.randn(2048, 5)
    r = torch.cat([x[:, :1], x[:, :1] + 0.2 * x[:, 1:2], x[:, 2:]], dim=1)
    a, _ = residual_decorrelation_loss(r, variance_floor=1e-4)
    scaled = r * torch.tensor([0.2, 3.0, 0.5, 2.0, 1.7])
    b, _ = residual_decorrelation_loss(scaled, variance_floor=1e-4)
    assert torch.allclose(a, b, atol=2e-6, rtol=2e-5)


def test_zero_variance_dimensions_are_masked_and_loss_stays_finite():
    torch.manual_seed(113)
    x = torch.randn(512, 3)
    r = torch.cat([x, torch.ones(512, 1), torch.zeros(512, 1)], dim=1).requires_grad_()
    loss, stats = residual_decorrelation_loss(r, variance_floor=1e-4)
    assert stats["active_dimensions"] == 3
    assert stats["masked_dimensions"] == 2
    assert torch.isfinite(loss)
    loss.backward()
    assert r.grad is not None and torch.isfinite(r.grad).all()
    all_constant = torch.ones(16, 4, requires_grad=True)
    zero, empty_stats = residual_decorrelation_loss(all_constant, variance_floor=1e-4)
    assert zero.item() == 0.0 and torch.isfinite(zero)
    assert empty_stats["active_dimensions"] == 0
    zero.backward()
    assert all_constant.grad is not None and torch.isfinite(all_constant.grad).all()


def test_per_dataset_statistics_differ_from_pooled_shifted_example():
    torch.manual_seed(114)
    a = torch.randn(2048, 3)
    b = torch.randn(2048, 3) + torch.tensor([8.0, 8.0, 8.0])
    la, _ = residual_decorrelation_loss(a, variance_floor=1e-4)
    lb, _ = residual_decorrelation_loss(b, variance_floor=1e-4)
    pooled, _ = residual_decorrelation_loss(torch.cat([a, b], dim=0), variance_floor=1e-4)
    per_dataset_mean = 0.5 * (la + lb)
    assert per_dataset_mean.item() < pooled.item() * 0.10
    assert pooled.item() > 0.10


def test_combined_approved_paths_reach_encoder_decoder_projection_only():
    torch.manual_seed(115)
    branch = Stage6ResidualBranch(
        shared_dim=128, dataset_dim=128, predictor_hidden_dim=64,
        residual_hidden_dim=64, residual_dim=64, decoder_hidden_dim=64,
    )
    z_s = torch.randn(256, 128, requires_grad=True)
    z_d = torch.randn(256, 128, requires_grad=True)
    out = branch(z_s, z_d)
    l_decorr, _ = residual_decorrelation_loss(out["R"], variance_floor=1e-4)
    l_info = information_reconstruction_loss(out["E_d_hat"], out["E_d"])
    l_proj = projection_geometry_loss(out["R"], out["R_proj_objective"])
    (l_decorr + l_info + l_proj).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.decoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.projection.parameters())
    assert all(p.grad is None for p in branch.predictor.parameters())
    assert z_s.grad is None and z_d.grad is None


def test_decorr_config_switch_defaults_off():
    config = Stage6TrainingConfig(
        residual_dim=64, lambda_red=0.1, variance_floor=1e-4,
        projection_mode="trainable", device="cpu",
    )
    assert config.decorr_enabled is False
    assert config.lambda_decorr == 0.0
