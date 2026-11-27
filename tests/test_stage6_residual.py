import torch
from torch import nn

from novel_ids.residual.stage6 import (
    Stage6LossCoefficients,
    Stage6ResidualBranch,
    combine_stage6_losses,
    combine_stage6_losses_with_projection,
    freeze_stage5,
    information_reconstruction_loss,
    magnitude_stability_loss,
    prediction_loss,
    initialize_fixed_semi_orthogonal_projection,
    projection_geometry_loss,
    redundancy_warmup,
    stage5_representations,
    standardized_cross_covariance_loss,
)


torch.set_num_threads(2)


def _has_nonzero_grad(module):
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def test_stage6_branch_shapes_and_finite_values_for_both_residual_widths():
    torch.manual_seed(10)
    z_s = torch.randn(8, 128)
    z_d = torch.randn(8, 128)
    for residual_dim in (64, 128):
        branch = Stage6ResidualBranch(
            shared_dim=128,
            dataset_dim=128,
            predictor_hidden_dim=96,
            residual_hidden_dim=96,
            residual_dim=residual_dim,
            decoder_hidden_dim=96,
        )
        output = branch(z_s, z_d)
        assert output["Z_d_hat"].shape == (8, 128)
        assert output["E_d"].shape == (8, 128)
        assert output["R"].shape == (8, residual_dim)
        assert output["R_proj"].shape == (8, 128)
        assert output["E_d_hat"].shape == (8, 128)
        assert all(torch.isfinite(value).all() for value in output.values())


def test_prediction_loss_updates_predictor_only_and_detaches_stage5_inputs():
    torch.manual_seed(11)
    branch = Stage6ResidualBranch(128, 128, 96, 96, 64, 96)
    z_s = torch.randn(16, 128, requires_grad=True)
    z_d = torch.randn(16, 128, requires_grad=True)
    output = branch(z_s, z_d)
    loss = prediction_loss(output["Z_d_hat"], z_d)
    loss.backward()

    assert _has_nonzero_grad(branch.predictor)
    assert not _has_nonzero_grad(branch.encoder)
    assert not _has_nonzero_grad(branch.decoder)
    assert z_s.grad is None
    assert z_d.grad is None


def test_information_loss_updates_encoder_and_decoder_not_predictor():
    torch.manual_seed(12)
    branch = Stage6ResidualBranch(128, 128, 96, 96, 64, 96)
    z_s = torch.randn(16, 128, requires_grad=True)
    z_d = torch.randn(16, 128, requires_grad=True)
    output = branch(z_s, z_d)
    loss = information_reconstruction_loss(output["E_d_hat"], output["E_d"])
    loss.backward()

    assert _has_nonzero_grad(branch.encoder)
    assert _has_nonzero_grad(branch.decoder)
    assert not _has_nonzero_grad(branch.predictor)
    assert z_s.grad is None
    assert z_d.grad is None


def test_stage5_freeze_and_no_grad_forward():
    stage5 = nn.Sequential(nn.Linear(12, 8), nn.Dropout(0.9))
    freeze_stage5(stage5)
    assert not stage5.training
    assert all(not parameter.requires_grad for parameter in stage5.parameters())
    result = stage5_representations(stage5, torch.randn(4, 12, requires_grad=True))
    assert not result.requires_grad


def test_redundancy_loss_float32_masks_low_variance_and_stops_shared_gradient():
    torch.manual_seed(13)
    z_s = torch.randn(256, 128, requires_grad=True)
    residual = torch.randn(256, 64, requires_grad=True)
    with torch.no_grad():
        z_s[:, 0] = 1.0
        residual[:, 1] = 0.0
    loss, stats = standardized_cross_covariance_loss(
        z_s, residual, variance_floor=1e-5
    )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert stats["masked_shared_dimensions"] >= 1
    assert stats["masked_residual_dimensions"] >= 1
    loss.backward()
    assert z_s.grad is None
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()


def test_redundancy_loss_rejects_invalid_batch_and_floor():
    try:
        standardized_cross_covariance_loss(
            torch.randn(1, 4), torch.randn(1, 3), variance_floor=1e-5
        )
    except ValueError as exc:
        assert "batch size" in str(exc)
    else:
        raise AssertionError("Expected batch-size validation.")

    try:
        standardized_cross_covariance_loss(
            torch.randn(4, 4), torch.randn(4, 3), variance_floor=0
        )
    except ValueError as exc:
        assert "variance_floor" in str(exc)
    else:
        raise AssertionError("Expected variance-floor validation.")


def test_redundancy_warmup_is_configurable_and_linear():
    assert redundancy_warmup(0, 100, 0.4) == 0.0
    assert redundancy_warmup(25, 100, 0.4) == 0.1
    assert redundancy_warmup(100, 100, 0.4) == 0.4
    assert redundancy_warmup(150, 100, 0.4) == 0.4
    assert redundancy_warmup(5, 0, 0.4) == 0.4


def test_stage6_objective_uses_explicit_coefficients():
    coeffs = Stage6LossCoefficients(
        prediction=1.0, information=0.5, magnitude=0.2, redundancy=0.1
    )
    losses = [torch.tensor(value) for value in (2.0, 4.0, 5.0, 10.0)]
    total = combine_stage6_losses(
        l_pred=losses[0],
        l_info=losses[1],
        l_mag=losses[2],
        l_red=losses[3],
        coefficients=coeffs,
    )
    assert torch.isclose(total, torch.tensor(6.0))



def test_magnitude_stability_loss_matches_approved_formula_and_updates_encoder():
    torch.manual_seed(14)
    branch = Stage6ResidualBranch(128, 128, 128, 128, 64, 128)
    z_s = torch.randn(32, 128)
    z_d = torch.randn(32, 128)
    output = branch(z_s, z_d)
    loss, stats = magnitude_stability_loss(
        output["R"], min_dimension_std=0.1, epsilon=1e-6
    )
    r = output["R"].float()
    expected = (
        (torch.sqrt(r.square().mean() + 1e-6) - 1.0).square()
        + torch.relu(0.1 - torch.sqrt(r.var(dim=0, correction=0) + 1e-6)).square().mean()
    )
    assert torch.allclose(loss, expected, atol=1e-7, rtol=1e-6)
    assert torch.isfinite(loss)
    assert stats["rms"] >= 0.0
    loss.backward()
    assert _has_nonzero_grad(branch.encoder)
    assert not _has_nonzero_grad(branch.predictor)
    assert not _has_nonzero_grad(branch.decoder)


def test_magnitude_stability_penalizes_collapsed_residual_without_nan():
    collapsed = torch.zeros(16, 64, requires_grad=True)
    loss, stats = magnitude_stability_loss(collapsed)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert abs(stats["dimension_std_min"] - 0.001) < 1e-8
    assert abs(stats["rms"] - 0.001) < 1e-8
    loss.backward()
    assert collapsed.grad is not None
    assert torch.isfinite(collapsed.grad).all()


def test_magnitude_stability_rejects_invalid_inputs():
    for value in (
        torch.randn(8),
        torch.randn(1, 64),
    ):
        try:
            magnitude_stability_loss(value)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid residual input to be rejected.")


def test_projection_geometry_loss_updates_projection_only():
    torch.manual_seed(21)
    residual = torch.randn(32, 64, requires_grad=True)
    projection = nn.Linear(64, 128)
    projected = projection(residual.detach())
    loss = projection_geometry_loss(residual, projected)
    assert torch.isfinite(loss)
    loss.backward()
    assert residual.grad is None
    assert _has_nonzero_grad(projection)


def test_fixed_semi_orthogonal_projection_preserves_gram_geometry():
    torch.manual_seed(22)
    projection = initialize_fixed_semi_orthogonal_projection(nn.Linear(64, 128))
    residual = torch.randn(24, 64)
    projected = projection(residual)
    loss = projection_geometry_loss(residual, projected)
    assert loss.item() < 1e-10
    assert all(not p.requires_grad for p in projection.parameters())
    assert all(p.grad is None for p in projection.parameters())


def test_identity_projection_geometry_loss_is_zero_for_128_dimensions():
    residual = torch.randn(20, 128)
    projected = nn.Identity()(residual)
    loss = projection_geometry_loss(residual, projected)
    assert loss.item() < 1e-10


def test_projection_geometry_loss_rejects_invalid_shapes_and_batch():
    cases = [
        (torch.randn(8, 64), torch.randn(8, 128, 1), "rank-2"),
        (torch.randn(8, 64), torch.randn(7, 128), "batch sizes"),
        (torch.randn(1, 64), torch.randn(1, 128), "batch size"),
    ]
    for r, p, expected in cases:
        try:
            projection_geometry_loss(r, p)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("Expected invalid projection geometry input.")


def test_branch_projection_objective_path_updates_projection_not_encoder():
    torch.manual_seed(23)
    branch = Stage6ResidualBranch(128, 128, 96, 96, 64, 96)
    output = branch(torch.randn(24, 128), torch.randn(24, 128))
    loss = projection_geometry_loss(output["R"], output["R_proj_objective"])
    loss.backward()
    assert _has_nonzero_grad(branch.projection)
    assert not _has_nonzero_grad(branch.encoder)
    assert not _has_nonzero_grad(branch.predictor)
    assert not _has_nonzero_grad(branch.decoder)


def test_projection_objective_is_added_with_approved_weight_without_changing_base_losses():
    coeffs = Stage6LossCoefficients(
        prediction=1.0, information=1.0, magnitude=1.0, redundancy=1.0
    )
    values = [torch.tensor(v) for v in (2.0, 3.0, 4.0, 5.0)]
    base = combine_stage6_losses(
        l_pred=values[0], l_info=values[1], l_mag=values[2], l_red=values[3],
        coefficients=coeffs,
    )
    total = combine_stage6_losses_with_projection(
        l_pred=values[0], l_info=values[1], l_mag=values[2], l_red=values[3],
        l_proj=torch.tensor(6.0), coefficients=coeffs, projection_coefficient=1.0,
    )
    assert torch.isclose(base, torch.tensor(14.0))
    assert torch.isclose(total, base + 6.0)


def test_projection_objective_rejects_negative_weight():
    coeffs = Stage6LossCoefficients(1.0, 1.0, 1.0, 1.0)
    try:
        combine_stage6_losses_with_projection(
            l_pred=torch.tensor(1.0), l_info=torch.tensor(1.0),
            l_mag=torch.tensor(1.0), l_red=torch.tensor(1.0),
            l_proj=torch.tensor(1.0), coefficients=coeffs,
            projection_coefficient=-0.1,
        )
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative projection weight to be rejected.")
