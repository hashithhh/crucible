"""Phase 2.4 tests — greedy, temperature, top-k, top-p, and cached generation.

Sampling bugs are quiet: a filter that keeps k+1 entries, a temperature applied
after filtering, a generate() whose cache drifts from the uncached path. Each
test here fails loudly on one of those.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from crucible.model import ModelConfig, Transformer  # noqa: E402
from crucible.sampling import (  # noqa: E402
    apply_temperature,
    generate,
    greedy,
    sample_next,
    top_k_filter,
    top_p_filter,
)

TINY = ModelConfig(
    vocab_size=50, d_model=64, n_layers=2, n_heads=4, context=32, d_ff=256
)
SEED = 0


def logits(rows: int = 3, vocab: int = 10) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(rows, vocab, generator=g)


def kept(row: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(row)


# --- greedy ---------------------------------------------------------------


def test_greedy_picks_the_argmax():
    x = logits()
    out = greedy(x)
    assert out.shape == (x.shape[0],) and out.dtype == torch.int64
    torch.testing.assert_close(out, x.argmax(dim=-1))


def test_greedy_breaks_ties_toward_the_lowest_id():
    x = torch.zeros(1, 5)
    x[0, 1] = x[0, 3] = 2.0
    assert greedy(x).item() == 1


def test_greedy_is_invariant_to_monotone_rescaling():
    x = logits()
    torch.testing.assert_close(greedy(x), greedy(x * 3.0 + 1.0))


# --- temperature ----------------------------------------------------------


def test_temperature_one_is_a_no_op():
    x = logits()
    torch.testing.assert_close(apply_temperature(x, 1.0), x)


def test_low_temperature_concentrates_mass_on_the_argmax():
    x = logits()
    hot = apply_temperature(x, 2.0).softmax(-1)
    cold = apply_temperature(x, 0.1).softmax(-1)
    assert (cold.max(-1).values > hot.max(-1).values).all()
    torch.testing.assert_close(cold.argmax(-1), x.argmax(-1))


def test_temperature_keeps_rejected_entries_rejected():
    x = logits()
    x[:, 0] = -float("inf")
    assert torch.isneginf(apply_temperature(x, 0.5)[:, 0]).all()


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_temperature_raises(bad):
    with pytest.raises(ValueError):
        apply_temperature(logits(), bad)


# --- top-k ----------------------------------------------------------------


def test_top_k_keeps_exactly_k_entries_with_original_values():
    x = logits(rows=4, vocab=20)
    out = top_k_filter(x, 5)
    assert out.shape == x.shape
    for row_in, row_out in zip(x, out, strict=True):
        keep = kept(row_out)
        assert keep.sum().item() == 5
        torch.testing.assert_close(row_out[keep], row_in[keep])
        expected = set(row_in.topk(5).indices.tolist())
        assert expected == set(keep.nonzero().flatten().tolist())


def test_top_k_at_or_above_vocab_is_a_no_op():
    x = logits(vocab=10)
    torch.testing.assert_close(top_k_filter(x, 10), x)
    torch.testing.assert_close(top_k_filter(x, 99), x)


@pytest.mark.parametrize("bad", [0, -3])
def test_bad_k_raises(bad):
    with pytest.raises(ValueError):
        top_k_filter(logits(), bad)


# --- top-p ----------------------------------------------------------------


def test_top_p_keeps_the_smallest_set_reaching_p():
    x = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.06, 0.04]]))
    # 0.5 + 0.25 = 0.75 < 0.9; + 0.15 = 0.90 >= 0.9, so exactly three survive.
    keep = kept(top_p_filter(x, 0.9)[0])
    assert keep.tolist() == [True, True, True, False, False]


def test_top_p_always_keeps_the_argmax():
    x = torch.log(torch.tensor([[0.9, 0.05, 0.03, 0.02]]))
    keep = kept(top_p_filter(x, 0.5)[0])
    assert keep.tolist() == [True, False, False, False]


def test_top_p_one_keeps_every_finite_entry():
    x = logits()
    out = top_p_filter(x, 1.0)
    assert kept(out).all()
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_bad_p_raises(bad):
    with pytest.raises(ValueError):
        top_p_filter(logits(), bad)


# --- sample_next ----------------------------------------------------------


def test_sample_next_returns_only_kept_ids():
    x = logits(rows=2, vocab=30)
    filtered = top_k_filter(x, 3)
    allowed = [
        set(filtered[r].isfinite().nonzero().flatten().tolist()) for r in range(2)
    ]
    g = torch.Generator().manual_seed(SEED)
    for _ in range(40):
        out = sample_next(x, top_k=3, generator=g)
        for r in range(2):
            assert out[r].item() in allowed[r]


def test_sample_next_is_reproducible_with_a_generator():
    x = logits()
    a = sample_next(x, temperature=0.8, generator=torch.Generator().manual_seed(7))
    b = sample_next(x, temperature=0.8, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)


def test_sample_next_tends_to_greedy_as_temperature_falls():
    x = logits(rows=1, vocab=20)
    g = torch.Generator().manual_seed(SEED)
    draws = [sample_next(x, temperature=0.01, generator=g).item() for _ in range(20)]
    assert set(draws) == {greedy(x).item()}


def test_sample_next_validates_like_the_filters():
    x = logits()
    with pytest.raises(ValueError):
        sample_next(x, temperature=0.0)
    with pytest.raises(ValueError):
        sample_next(x, top_k=0)
    with pytest.raises(ValueError):
        sample_next(x, top_p=2.0)


# --- generate -------------------------------------------------------------


@pytest.fixture
def model():
    torch.manual_seed(SEED)
    return Transformer(TINY).eval()


def prompt(batch: int = 2, steps: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED)
    return torch.randint(0, TINY.vocab_size, (batch, steps), generator=g)


def test_generate_extends_the_prompt_in_place(model):
    p = prompt()
    gen = torch.Generator().manual_seed(1)
    out = generate(model, p, max_new_tokens=6, temperature=1.0, generator=gen)
    assert out.shape == (p.shape[0], p.shape[1] + 6)
    torch.testing.assert_close(out[:, : p.shape[1]], p)
    assert out.min() >= 0 and out.max() < TINY.vocab_size


def test_generate_is_deterministic_at_low_temperature(model):
    p = prompt(batch=1)
    a = generate(
        model, p, 5, temperature=0.01, generator=torch.Generator().manual_seed(3)
    )
    b = generate(
        model, p, 5, temperature=0.01, generator=torch.Generator().manual_seed(3)
    )
    torch.testing.assert_close(a, b)


def test_generate_matches_uncached(model):
    """The cached decode must equal step-by-step full forwards, greedily."""
    seq = prompt(batch=1)
    expected = seq.clone()
    for _ in range(5):
        nxt = greedy(model(expected)[:, -1])
        expected = torch.cat([expected, nxt[:, None]], dim=1)

    gen = torch.Generator().manual_seed(0)
    got = generate(model, seq, 5, temperature=0.01, generator=gen)
    torch.testing.assert_close(got, expected)


def test_generate_refuses_to_exceed_the_context(model):
    with pytest.raises(ValueError):
        generate(model, prompt(batch=1, steps=TINY.context - 2), max_new_tokens=10)


def test_generate_stops_at_eos(model):
    p = prompt(batch=1)
    eos = greedy(model(p)[:, -1]).item()  # whatever it would emit first
    gen = torch.Generator().manual_seed(0)
    out = generate(model, p, 8, temperature=0.01, eos_id=eos, generator=gen)
    assert out.shape[1] == p.shape[1] + 1 and out[0, -1].item() == eos


def test_generate_leaves_no_gradients(model):
    gen = torch.Generator().manual_seed(0)
    out = generate(model, prompt(batch=1), 3, temperature=1.0, generator=gen)
    assert not out.requires_grad
    assert all(p.grad is None for p in model.parameters())
