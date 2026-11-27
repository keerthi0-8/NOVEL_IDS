import torch

from novel_ids.transformer.stage10 import Stage10Transformer


def _inputs(batch=3, features=78, seed=101):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randn(batch, features, 128, generator=generator)
    z_u = torch.randn(batch, 128, generator=generator)
    mask = torch.rand(batch, features, generator=generator) > 0.25
    return tokens, z_u, mask


def test_output_shapes_for_source_and_extension_token_counts():
    torch.manual_seed(1)
    model = Stage10Transformer().eval()
    for feature_count in (78, 92):
        tokens, z_u, mask = _inputs(batch=2, features=feature_count, seed=feature_count)
        with torch.no_grad():
            output = model(tokens, z_u, mask)
        assert output["Z_t"].shape == (2, 128)
        assert output["token_outputs"].shape == (2, feature_count, 128)


def test_unavailable_token_content_cannot_change_cls_output():
    torch.manual_seed(2)
    model = Stage10Transformer().eval()
    tokens, z_u, mask = _inputs(batch=2, features=92, seed=2)
    unavailable = ~mask
    changed = tokens.clone()
    changed[unavailable] = torch.randn_like(changed[unavailable]) * 1000.0
    with torch.no_grad():
        original = model(tokens, z_u, mask)["Z_t"]
        altered = model(changed, z_u, mask)["Z_t"]
    difference = (original - altered).abs().max().item()
    assert difference < 1e-6


def test_available_token_permutation_is_equivariant_and_cls_invariant():
    torch.manual_seed(3)
    model = Stage10Transformer().eval()
    tokens, z_u, mask = _inputs(batch=2, features=12, seed=3)
    mask = torch.ones_like(mask)  # Isolate the permutation property for available tokens.
    permutation = torch.randperm(tokens.shape[1])
    with torch.no_grad():
        original = model(tokens, z_u, mask)
        permuted = model(tokens[:, permutation], z_u, mask[:, permutation])
    assert torch.allclose(original["Z_t"], permuted["Z_t"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        permuted["token_outputs"],
        original["token_outputs"][:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )


def test_all_unavailable_features_produce_finite_cls():
    torch.manual_seed(4)
    model = Stage10Transformer().eval()
    tokens, z_u, _ = _inputs(batch=2, features=78, seed=4)
    mask = torch.zeros(2, 78, dtype=torch.bool)
    with torch.no_grad():
        output = model(tokens, z_u, mask)
    assert torch.isfinite(output["Z_t"]).all()
    assert torch.count_nonzero(output["token_outputs"]) == 0


def test_mixed_batch_availability_is_independent_per_row():
    torch.manual_seed(5)
    model = Stage10Transformer().eval()
    tokens, z_u, mask = _inputs(batch=3, features=92, seed=5)
    mask[0] = False
    mask[1, ::2] = False
    mask[2, 1::3] = False
    with torch.no_grad():
        batched = model(tokens, z_u, mask)["Z_t"]
        alone = model(tokens[1:2], z_u[1:2], mask[1:2])["Z_t"]
    assert torch.allclose(batched[1:2], alone, atol=1e-6, rtol=1e-6)


def test_forward_backward_finite_and_gradients_reach_all_stage10_parts():
    torch.manual_seed(6)
    model = Stage10Transformer().train()
    tokens, z_u, mask = _inputs(batch=4, features=78, seed=6)
    output = model(tokens, z_u, mask)
    loss = output["Z_t"].square().mean() + output["token_outputs"].square().mean()
    loss.backward()

    assert torch.isfinite(output["Z_t"]).all()
    assert torch.isfinite(output["token_outputs"]).all()
    assert model.token_conditioning.weight.grad is not None
    assert model.token_conditioning.weight.grad.abs().sum().item() > 0
    assert model.cls_projection.weight.grad is not None
    assert model.cls_projection.weight.grad.abs().sum().item() > 0
    for parameter in model.encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0


def test_dropout_is_active_only_in_train_mode():
    torch.manual_seed(7)
    model = Stage10Transformer()
    tokens, z_u, mask = _inputs(batch=2, features=20, seed=7)

    model.eval()
    with torch.no_grad():
        eval_a = model(tokens, z_u, mask)["Z_t"]
        eval_b = model(tokens, z_u, mask)["Z_t"]
    assert torch.equal(eval_a, eval_b)

    model.train()
    with torch.no_grad():
        train_a = model(tokens, z_u, mask)["Z_t"]
        train_b = model(tokens, z_u, mask)["Z_t"]
    assert not torch.equal(train_a, train_b)


def test_no_nan_or_inf_in_outputs_or_gradients():
    torch.manual_seed(8)
    model = Stage10Transformer().train()
    tokens, z_u, mask = _inputs(batch=3, features=92, seed=8)
    mask[0] = False
    output = model(tokens, z_u, mask)
    assert all(torch.isfinite(value).all() for value in output.values())
    (output["Z_t"].square().mean() + output["token_outputs"].square().mean()).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_invalid_mask_is_rejected():
    model = Stage10Transformer().eval()
    tokens, z_u, mask = _inputs(batch=1, features=5, seed=9)
    mask = mask.to(torch.float32)
    mask[0, 0] = 0.5
    try:
        model(tokens, z_u, mask)
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("Non-binary M_avail was not rejected.")
