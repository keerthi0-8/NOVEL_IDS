import torch
import pytest

from novel_ids.alignment.stage8 import (
    CORAL_LAMBDA_CANDIDATES,
    NSL_KDD,
    UNSW_NB15,
    UNSEEN,
    CompatibilityEstimator,
    Stage8SelectiveAlignment,
    normalized_weighted_coral,
    weighted_covariance,
)
from novel_ids.fusion.stage9 import UnifiedRepresentation


torch.set_num_threads(2)


def _fit_estimator():
    torch.manual_seed(810)
    estimator = CompatibilityEstimator()
    estimator.fit_source_training_references(
        torch.randn(300, 128),
        torch.randn(320, 128) + 0.7,
    )
    return estimator


def _metadata(batch, features=6):
    available = torch.ones(batch, features, dtype=torch.bool)
    levels = torch.ones(batch, features, dtype=torch.long)
    confidence = torch.full((batch, features), 0.8)
    group_only = torch.zeros(batch, features, dtype=torch.bool)
    return available, levels, confidence, group_only


def test_compatibility_range_shape_stop_gradient_and_unseen_distance():
    estimator = _fit_estimator()
    z = torch.randn(5, 128, requires_grad=True)
    ids = torch.tensor([NSL_KDD, UNSW_NB15, UNSEEN, NSL_KDD, UNSEEN])
    meta = _metadata(5)
    out = estimator(z, ids, *meta)
    assert out["C"].shape == (5, 1)
    assert torch.all((out["C"] >= 0.05) & (out["C"] <= 0.95))
    assert out["C"].requires_grad is False
    assert out["distance"].shape == (5, 1)
    assert torch.allclose(
        out["distance"][2],
        torch.minimum(
            ((z[2].detach() - estimator.reference_means[0]).square()
             / estimator.reference_variances[0]).mean().reshape(1),
            ((z[2].detach() - estimator.reference_means[1]).square()
             / estimator.reference_variances[1]).mean().reshape(1),
        ),
    )
    assert not estimator.reference_means.requires_grad
    assert not estimator.reference_variances.requires_grad


def test_confidence_aggregation_respects_levels_availability_and_match_count():
    available = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    levels = torch.tensor([[1, 2, 3, 1], [1, 2, 3, 1]])
    confidence = torch.tensor([[0.9, 0.8, 1.0, 1.0], [0.2, 1.0, 1.0, 1.0]])
    group = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=torch.bool)
    score = CompatibilityEstimator.confidence_score(available, levels, confidence, group)
    assert torch.allclose(score[:, 0], torch.tensor([0.43333334, 0.2]), atol=1e-6)
    assert score[1] < score[0]


def test_compatibility_references_fit_tau_and_are_not_adapted_by_forward():
    estimator = CompatibilityEstimator()
    nsl = torch.randn(280, 128)
    unsw = torch.randn(290, 128) + 0.4
    estimator.fit_source_training_references(nsl, unsw)
    saved_means = estimator.reference_means.clone()
    saved_tau = estimator.tau.clone()
    assert estimator.tau.item() > 0
    out = estimator(
        torch.randn(4, 128), torch.tensor([NSL_KDD, UNSW_NB15, UNSEEN, UNSEEN]),
        *_metadata(4),
    )
    assert torch.equal(estimator.reference_means, saved_means)
    assert torch.equal(estimator.tau, saved_tau)
    assert torch.isfinite(out["C"]).all()


def test_weighted_covariance_constant_weights_matches_population_covariance():
    torch.manual_seed(811)
    x = torch.randn(300, 128)
    weights = torch.full((300, 1), 0.37)
    weighted = weighted_covariance(x, weights)
    centered = x - x.mean(0, keepdim=True)
    ordinary = centered.T @ centered / x.shape[0]
    assert torch.allclose(weighted, ordinary, rtol=1e-5, atol=1e-6)


def test_coral_identical_covariance_zero_scale_free_and_distribution_sensitive():
    torch.manual_seed(812)
    x = torch.randn(300, 128)
    weights = torch.full((300, 1), 0.6)
    zero = normalized_weighted_coral(x, x.clone(), weights, weights)
    scaled = normalized_weighted_coral(3.0 * x, 3.0 * x.clone(), weights, weights)
    shifted_cov = x * torch.linspace(0.2, 2.0, 128)
    different = normalized_weighted_coral(x, shifted_cov, weights, weights)
    assert zero.item() < 1e-7
    assert scaled.item() < 1e-7
    assert different.item() > 0.05


def test_coral_enforces_minimum_subbatch_and_lambda_candidates():
    assert CORAL_LAMBDA_CANDIDATES == (0.0, 0.05, 0.2, 1.0)
    x = torch.randn(255, 128)
    with pytest.raises(ValueError, match="at least 256"):
        normalized_weighted_coral(x, x, torch.ones(255, 1), torch.ones(255, 1))


def test_stage8_modes_and_coral_loss_are_finite():
    torch.manual_seed(813)
    nsl, unsw = torch.randn(256, 128), torch.randn(256, 128) + 0.5
    all_train = torch.cat((nsl, unsw))
    compat = Stage8SelectiveAlignment(coral_lambda=0.2, mode="compatibility", warmup_fraction=0.5)
    compat.compatibility.fit_source_training_references(nsl, unsw)
    ids = torch.tensor([NSL_KDD] * 256 + [UNSW_NB15] * 256)
    meta = _metadata(512)
    out = compat(all_train, all_train.clone(), ids, *meta)
    losses = compat.coral_loss(out, ids)
    assert torch.isfinite(losses["L_coral"])
    assert torch.isfinite(losses["L_align"])
    assert torch.allclose(losses["L_align"], losses["L_coral"] * 0.1)

    no_align = Stage8SelectiveAlignment(coral_lambda=1.0, mode="no_alignment")
    no_align.compatibility.fit_source_training_references(nsl, unsw)
    no_out = no_align(all_train, all_train, ids, *meta)
    no_loss = no_align.coral_loss(no_out, ids)
    assert torch.count_nonzero(no_out["C"]) == 0
    assert no_loss["L_align"].item() == 0.0

    full = Stage8SelectiveAlignment(coral_lambda=0.05, mode="full_alignment")
    full.compatibility.fit_source_training_references(nsl, unsw)
    full_out = full(all_train, all_train, ids, *meta)
    assert torch.all(full_out["C"] == 1.0)


def test_common_feature_only_compatibility_uses_level1_confidence_only():
    estimator = _fit_estimator()
    available = torch.ones(2, 4, dtype=torch.bool)
    levels = torch.tensor([[1, 2, 2, 3], [1, 2, 2, 3]])
    confidence = torch.full((2, 4), 0.8)
    group = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=torch.bool)
    ids = torch.tensor([NSL_KDD, UNSW_NB15])
    z = torch.randn(2, 128)
    normal = estimator(z, ids, available, levels, confidence, group)
    common = estimator(z, ids, available, levels, confidence, group, common_feature_only=True)
    # Restricting the denominator to available Level 1 correspondences
    # raises this mean when Level 2/3 features had diluted the ordinary score.
    assert torch.all(common["s_conf"] > normal["s_conf"])


def test_per_dimension_compatibility_switch_is_present_but_blocked_by_default():
    module = Stage8SelectiveAlignment()
    assert module.per_dimension_compatibility is False
    with pytest.raises(NotImplementedError, match="pending a separately approved formula"):
        Stage8SelectiveAlignment(per_dimension_compatibility=True)


def test_stage8_to_stage9_gradients_reach_both_projection_layers_not_compatibility():
    torch.manual_seed(815)
    nsl, unsw = torch.randn(256, 128), torch.randn(256, 128) + 0.8
    stage8 = Stage8SelectiveAlignment(coral_lambda=0.2)
    stage8.compatibility.fit_source_training_references(nsl, unsw)
    z_f = torch.cat((nsl, unsw)).clone().requires_grad_(True)
    z_d = torch.randn(512, 128, requires_grad=True)
    ids = torch.tensor([NSL_KDD] * 256 + [UNSW_NB15] * 256)
    available, levels, confidence, group = _metadata(512)
    confidence.requires_grad_(True)
    stage8_out = stage8(z_f, z_d, ids, available, levels, confidence, group)
    z_s, r, r_proj, gate = (torch.randn(512, 128) for _ in range(4))
    fused = UnifiedRepresentation()(
        z_s=z_s, r=r, r_proj=r_proj, gate=torch.sigmoid(gate),
        a=stage8_out["A"], u=stage8_out["U"], c=stage8_out["C"], z_f=z_f,
    )
    fused["Z_u"].square().mean().backward()
    assert stage8.W_a.weight.grad is not None
    assert stage8.W_u.weight.grad is not None
    assert torch.isfinite(stage8.W_a.weight.grad).all()
    assert torch.isfinite(stage8.W_u.weight.grad).all()
    assert torch.count_nonzero(stage8.W_a.weight.grad) > 0
    assert torch.count_nonzero(stage8.W_u.weight.grad) > 0
    assert confidence.grad is None


def test_stage9_equation_debug_outputs_sample_correspondence_and_gradients():
    torch.manual_seed(814)
    batch = 7
    z_s = torch.randn(batch, 128, requires_grad=True)
    r = torch.randn(batch, 128, requires_grad=True)
    r_proj = torch.randn(batch, 128, requires_grad=True)
    gate = torch.sigmoid(torch.randn(batch, 128, requires_grad=True))
    a = torch.randn(batch, 128, requires_grad=True)
    u = torch.randn(batch, 128, requires_grad=True)
    c = torch.rand(batch, 1, requires_grad=True)
    out = UnifiedRepresentation()(
        z_s=z_s, r=r, r_proj=r_proj, gate=gate, a=a, u=u, c=c,
        z_f=torch.randn(batch, 128),
    )
    expected = z_s + gate * r_proj + c.detach() * a + (1 - c.detach()) * u
    assert out["Z_u"].shape == (batch, 128)
    assert torch.allclose(out["Z_u"], expected)
    assert torch.equal(out["R"], r)
    assert torch.equal(out["A"], a)
    assert torch.equal(out["U"], u)
    assert torch.equal(out["C"], c.detach())
    out["Z_u"].square().mean().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    assert c.grad is None
