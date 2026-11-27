import torch

from novel_ids.fusion.stage7 import AdaptiveResidualFusion


torch.set_num_threads(2)


def test_stage7_gate_shape_bounds_and_fusion_equation():
    torch.manual_seed(70)
    module = AdaptiveResidualFusion()
    z_s = torch.randn(12, 128)
    r_proj = torch.randn(12, 128)
    output = module(z_s, r_proj)

    gate = output["G"]
    z_f = output["Z_f"]
    assert gate.shape == (12, 128)
    assert z_f.shape == (12, 128)
    assert torch.isfinite(gate).all()
    assert torch.isfinite(z_f).all()
    assert torch.all(gate > 0)
    assert torch.all(gate < 1)
    assert torch.allclose(z_f, z_s + gate * r_proj)


def test_stage7_gate_bias_uses_approved_negative_initialization():
    module = AdaptiveResidualFusion()
    assert torch.allclose(module.gate.bias, torch.full((128,), -2.0))


def test_stage7_regularizer_is_disabled_by_default():
    module = AdaptiveResidualFusion()
    gate = torch.full((3, 128), 0.2)
    assert module.regularizer is None
    assert module.regularization_loss(gate) is None


def test_stage7_optional_regularizer_is_caller_supplied():
    module = AdaptiveResidualFusion(regularizer=lambda gate: gate.mean())
    result = module.regularization_loss(torch.full((3, 128), 0.25))
    assert result is not None
    assert result.ndim == 0
    assert torch.isclose(result, torch.tensor(0.25))


def test_stage7_rejects_incompatible_shapes():
    module = AdaptiveResidualFusion()
    try:
        module(torch.randn(4, 128), torch.randn(4, 64))
    except ValueError as exc:
        assert "matching" in str(exc)
    else:
        raise AssertionError("Expected incompatible-shape validation.")

    try:
        AdaptiveResidualFusion(shared_dim=64)
    except ValueError as exc:
        assert "dimensions must be 128" in str(exc)
    else:
        raise AssertionError("Expected dimension validation.")
