import torch

from novel_ids.residual.stage6 import standardized_cross_covariance_loss


torch.set_num_threads(2)


def _unclamped_reference(z_s, residual):
    """Original standardized covariance formula for strictly non-degenerate data."""
    zs = z_s.detach().float()
    r = residual.float()
    zs_centered = zs - zs.mean(dim=0, keepdim=True)
    r_centered = r - r.mean(dim=0, keepdim=True)
    zs_std = zs_centered.square().mean(dim=0).sqrt()
    r_std = r_centered.square().mean(dim=0).sqrt()
    zs_tilde = zs_centered / zs_std.unsqueeze(0)
    r_tilde = r_centered / r_std.unsqueeze(0)
    cov = zs_tilde.T @ r_tilde / float(zs.shape[0] - 1)
    return cov.square().sum() / float(zs.shape[1] * r.shape[1])


def test_variance_clamp_preserves_original_std_mask_threshold_and_counts():
    floor = 1e-3
    # Construct centered dimensions with stds 0, 0.5*floor, floor, and 2*floor.
    base = torch.tensor([-1.0, 1.0, -1.0, 1.0]).unsqueeze(1)
    z_s = torch.cat(
        [torch.ones(4, 1), base * (0.5 * floor),
         base * floor, base * (2.0 * floor)],
        dim=1,
    )
    residual = torch.cat(
        [torch.zeros(4, 1), base * (0.5 * floor),
         base * floor, base * (2.0 * floor)],
        dim=1,
    )
    _, stats = standardized_cross_covariance_loss(
        z_s, residual, variance_floor=floor
    )
    # Original condition was std >= floor: below-floor masked; equality retained.
    assert stats["masked_shared_dimensions"] == 2
    assert stats["masked_residual_dimensions"] == 2


def test_variance_clamp_matches_unclamped_gradients_on_non_degenerate_dimensions():
    torch.manual_seed(801)
    z_s = torch.randn(256, 9)
    residual_actual = torch.randn(256, 7, requires_grad=True)
    residual_reference = residual_actual.detach().clone().requires_grad_(True)

    actual, stats = standardized_cross_covariance_loss(
        z_s, residual_actual, variance_floor=1e-5
    )
    reference = _unclamped_reference(z_s, residual_reference)
    actual.backward()
    reference.backward()

    assert stats == {
        "masked_shared_dimensions": 0,
        "masked_residual_dimensions": 0,
    }
    assert torch.allclose(actual, reference, rtol=1e-6, atol=1e-8)
    assert torch.allclose(
        residual_actual.grad, residual_reference.grad, rtol=1e-5, atol=1e-8
    )


def test_zero_variance_dimension_is_masked_and_has_finite_gradients():
    torch.manual_seed(802)
    z_s = torch.randn(256, 6)
    residual = torch.randn(256, 5, requires_grad=True)
    with torch.no_grad():
        residual[:, 2] = 0.0
    loss, stats = standardized_cross_covariance_loss(
        z_s, residual, variance_floor=1e-5
    )
    loss.backward()

    assert stats["masked_residual_dimensions"] == 1
    assert stats["masked_shared_dimensions"] == 0
    assert torch.isfinite(loss)
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()
    assert torch.count_nonzero(residual.grad[:, 2]) == 0
