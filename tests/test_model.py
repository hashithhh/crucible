"""Phase 2.5 — shape, gradient and invariant tests for the transformer.

"A forward pass that runs is not a correct model." Each test pins one property
that a plausible-but-wrong implementation would break: silent non-causality, a
RoPE that encodes absolute rather than relative position, a KV cache that
drifts from the uncached result, a residual path that carries no gradient.

A tiny config keeps the suite fast; one test checks the real S25 size against
ADR-0006.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crucible.model import (  # noqa: E402
    MLP,
    Block,
    CausalSelfAttention,
    ModelConfig,
    RMSNorm,
    Transformer,
    apply_rope,
)

TINY = ModelConfig(
    vocab_size=50, d_model=64, n_layers=2, n_heads=4, context=32, d_ff=256
)
SEED = 0


@pytest.fixture
def model():
    torch.manual_seed(SEED)
    return Transformer(TINY).eval()


def ids(batch: int, steps: int, vocab: int = TINY.vocab_size, seed: int = SEED):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, steps), generator=g)


# --- config ---------------------------------------------------------------


def test_config_rejects_indivisible_head_split():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=8)


def test_s25_parameter_count_matches_adr_0006():
    """ADR-0006 sized S25 at 26,223,616 parameters; the model must agree."""
    m = Transformer(ModelConfig())
    assert m.n_params() == 26_223_616
    assert m.n_params(non_embedding=True) == 26_223_616 - 2049 * 512


# --- RMSNorm --------------------------------------------------------------


def test_rmsnorm_is_scale_invariant_and_shape_preserving():
    norm = RMSNorm(TINY.d_model, TINY.norm_eps)
    x = torch.randn(3, 5, TINY.d_model)
    out = norm(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    torch.testing.assert_close(norm(x * 7.5), out, rtol=1e-4, atol=1e-4)


def test_rmsnorm_survives_zeros_and_normalises_last_dim_only():
    norm = RMSNorm(TINY.d_model, TINY.norm_eps)
    assert torch.isfinite(norm(torch.zeros(2, 3, TINY.d_model))).all()
    x = torch.randn(4, 6, TINY.d_model)
    rows = norm(x)
    # Each position normalises alone: one row recomputed in isolation agrees.
    torch.testing.assert_close(norm(x[2:3, 4:5]), rows[2:3, 4:5])


def test_rmsnorm_has_one_parameter_vector():
    params = list(RMSNorm(TINY.d_model, TINY.norm_eps).parameters())
    assert len(params) == 1 and params[0].shape == (TINY.d_model,)


# --- RoPE -----------------------------------------------------------------


def test_rope_preserves_shape_and_norm():
    x = torch.randn(2, TINY.n_heads, 7, TINY.head_dim)
    pos = torch.arange(7).expand(2, 7)
    out = apply_rope(x, pos, theta=TINY.rope_theta)
    assert out.shape == x.shape and out.dtype == x.dtype
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_rope_at_position_zero_is_identity():
    x = torch.randn(1, TINY.n_heads, 1, TINY.head_dim)
    out = apply_rope(x, torch.zeros(1, 1, dtype=torch.long), theta=TINY.rope_theta)
    torch.testing.assert_close(out, x, rtol=1e-6, atol=1e-6)


def test_rope_dot_product_depends_only_on_offset():
    """The property the mechanism exists for: relative, not absolute, position."""
    q = torch.randn(1, 1, 1, TINY.head_dim)
    k = torch.randn(1, 1, 1, TINY.head_dim)

    def dot(m: int, n: int) -> torch.Tensor:
        qm = apply_rope(q, torch.tensor([[m]]), theta=TINY.rope_theta)
        kn = apply_rope(k, torch.tensor([[n]]), theta=TINY.rope_theta)
        return (qm * kn).sum()

    for (m, n), (m2, n2) in [((5, 3), (9, 7)), ((0, 4), (6, 10)), ((2, 2), (11, 11))]:
        torch.testing.assert_close(dot(m, n), dot(m2, n2), rtol=1e-4, atol=1e-4)


# --- components -----------------------------------------------------------


def test_mlp_is_position_wise():
    mlp = MLP(TINY).eval()
    x = torch.randn(2, 6, TINY.d_model)
    out = mlp(x)
    assert out.shape == x.shape
    perm = torch.randperm(6)
    torch.testing.assert_close(mlp(x[:, perm]), out[:, perm], rtol=1e-5, atol=1e-5)


def test_attention_is_causal():
    attn = CausalSelfAttention(TINY).eval()
    x = torch.randn(2, 8, TINY.d_model)
    pos = torch.arange(8).expand(2, 8)
    out = attn(x, pos)
    assert out.shape == x.shape

    tampered = x.clone()
    tampered[:, 5:] = torch.randn_like(tampered[:, 5:])  # change the future only
    torch.testing.assert_close(
        attn(tampered, pos)[:, :5], out[:, :5], rtol=1e-5, atol=1e-5
    )


def test_block_with_zeroed_outputs_is_identity():
    """Pre-norm residual: zero both sub-layer outputs and x passes through."""
    block = Block(TINY).eval()
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, CausalSelfAttention | MLP):
                linears = [
                    m for m in module.modules() if isinstance(m, torch.nn.Linear)
                ]
                linears[-1].weight.zero_()
    x = torch.randn(2, 5, TINY.d_model)
    pos = torch.arange(5).expand(2, 5)
    torch.testing.assert_close(block(x, pos), x, rtol=1e-6, atol=1e-6)


# --- whole model ----------------------------------------------------------


def test_forward_shape_and_dtype(model):
    logits = model(ids(3, 7))
    assert logits.shape == (3, 7, TINY.vocab_size)
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()


def test_single_step_forward_works(model):
    assert model(ids(1, 1)).shape == (1, 1, TINY.vocab_size)


def test_model_is_causal(model):
    seq = ids(2, 10)
    base = model(seq)
    tampered = seq.clone()
    tampered[:, 6:] = (tampered[:, 6:] + 1) % TINY.vocab_size
    torch.testing.assert_close(
        model(tampered)[:, :6], base[:, :6], rtol=1e-5, atol=1e-5
    )


def test_forward_is_deterministic(model):
    seq = ids(2, 9)
    torch.testing.assert_close(model(seq), model(seq), rtol=0, atol=0)


def test_context_limit_and_id_range_are_enforced(model):
    with pytest.raises(ValueError):
        model(ids(1, TINY.context + 1))
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        model(torch.tensor([[TINY.vocab_size]]))


def test_every_parameter_receives_gradient(model):
    logits = model(ids(2, 6))
    logits.square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    dead = [n for n, p in model.named_parameters() if not p.grad.any()]
    assert not dead, f"zero gradient at: {dead}"
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradient at: {bad}"


def test_gradients_ignore_later_tokens():
    """A loss at step t must produce the same gradients whatever follows t.

    Catches leakage that a forward-only causality check can miss, e.g. a mask
    applied to the output instead of to the attention scores.
    """
    torch.manual_seed(SEED)
    model = Transformer(TINY).eval()
    seq = ids(1, 8)
    cut = 3

    def grads_for(suffix_seed: int) -> list[torch.Tensor]:
        tampered = seq.clone()
        tampered[:, cut + 1 :] = ids(1, 8 - cut - 1, seed=suffix_seed)
        model.zero_grad(set_to_none=True)
        model(tampered)[0, cut].square().sum().backward()
        return [p.grad.clone() for p in model.parameters()]

    for a, b in zip(grads_for(1), grads_for(2), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


# --- KV cache -------------------------------------------------------------


def test_cache_reports_length_and_rejects_overflow(model):
    cache = model.new_cache(batch_size=1)
    assert cache.length == 0
    model(ids(1, 4), cache)
    assert cache.length == 4
    cache.reset()
    assert cache.length == 0
    with pytest.raises(ValueError):
        model(ids(1, TINY.context + 1), cache)


def test_kv_cache_matches_full_forward(model):
    """A prompt then one step at a time equals one full forward."""
    seq = ids(2, 12)
    full = model(seq)

    cache = model.new_cache(batch_size=2)
    stepwise = [model(seq[:, :5], cache)]
    for t in range(5, seq.shape[1]):
        stepwise.append(model(seq[:, t : t + 1], cache))
    torch.testing.assert_close(torch.cat(stepwise, dim=1), full, rtol=1e-4, atol=1e-4)


def test_cache_reset_gives_a_fresh_sequence(model):
    seq = ids(1, 6)
    cache = model.new_cache(batch_size=1)
    first = model(seq, cache)
    cache.reset()
    torch.testing.assert_close(model(seq, cache), first, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_runs_on_gpu_and_matches_cpu(model):
    seq = ids(1, 6)
    cpu = model(seq)
    gpu = model.cuda()(seq.cuda()).cpu()
    torch.testing.assert_close(gpu, cpu, rtol=1e-3, atol=1e-3)
