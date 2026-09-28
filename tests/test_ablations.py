"""Phase 4: the two pieces of new model code, tested where they could be silently wrong.

A broken window or a broken router does not crash. It trains, produces a loss,
and turns an ablation into a comparison of something other than what ADR-0009
says. So these check the properties the ablation relies on, and nothing else.
"""

from __future__ import annotations

import pytest
import torch

from crucible.model import CausalSelfAttention, ModelConfig, MoE, Transformer

TINY = dict(vocab_size=64, d_model=32, n_layers=2, n_heads=2, context=32, d_ff=64)
WINDOW = 4
STEPS = 16


def positions(batch: int, steps: int) -> torch.Tensor:
    return torch.arange(steps).unsqueeze(0).expand(batch, steps)


@pytest.fixture
def windowed():
    torch.manual_seed(0)
    return CausalSelfAttention(ModelConfig(**TINY), window=WINDOW).eval()


def test_window_hides_everything_older_than_the_window(windowed):
    x = torch.randn(1, STEPS, TINY["d_model"])
    t = STEPS - 1
    out = windowed(x, positions(1, STEPS))
    poked = x.clone()
    poked[:, t - WINDOW] += 10.0  # just outside the window seen from step t
    after = windowed(poked, positions(1, STEPS))
    assert torch.allclose(after[:, t], out[:, t], atol=1e-5)


def test_window_still_sees_inside_the_window(windowed):
    x = torch.randn(1, STEPS, TINY["d_model"])
    t = STEPS - 1
    out = windowed(x, positions(1, STEPS))
    poked = x.clone()
    poked[:, t - WINDOW + 1] += 10.0  # the oldest step still inside
    after = windowed(poked, positions(1, STEPS))
    assert not torch.allclose(after[:, t], out[:, t])


def test_windowed_attention_is_still_causal(windowed):
    x = torch.randn(1, STEPS, TINY["d_model"])
    out = windowed(x, positions(1, STEPS))
    poked = x.clone()
    poked[:, -1] += 10.0
    after = windowed(poked, positions(1, STEPS))
    assert torch.allclose(after[:, :-1], out[:, :-1], atol=1e-5)


def test_a_window_wider_than_the_sequence_is_full_attention():
    torch.manual_seed(0)
    full = CausalSelfAttention(ModelConfig(**TINY)).eval()
    wide = CausalSelfAttention(ModelConfig(**TINY), window=STEPS).eval()
    wide.load_state_dict(full.state_dict())
    x = torch.randn(2, STEPS, TINY["d_model"])
    p = positions(2, STEPS)
    assert torch.allclose(full(x, p), wide(x, p), atol=1e-6)


def test_only_odd_layers_are_windowed():
    """ADR-0009: alternate full and local layers, starting full."""
    model = Transformer(ModelConfig(**{**TINY, "n_layers": 4}, window=WINDOW))
    assert [b.attn.window for b in model.blocks] == [0, WINDOW, 0, WINDOW]


def test_windowed_kv_cache_matches_the_full_forward():
    """The mask uses absolute positions, so cached decoding must agree."""
    torch.manual_seed(0)
    model = Transformer(ModelConfig(**TINY, window=WINDOW)).eval()
    ids = torch.randint(0, TINY["vocab_size"], (1, STEPS))
    with torch.no_grad():
        full = model(ids)
        cache = model.new_cache(1)
        steps = [model(ids[:, i : i + 1], cache) for i in range(STEPS)]
    assert torch.allclose(full, torch.cat(steps, 1), atol=1e-4)


def test_moe_is_position_wise():
    """No capacity limit: a token's output cannot depend on its batch-mates."""
    torch.manual_seed(0)
    moe = MoE(ModelConfig(**TINY, n_experts=4)).eval()
    x = torch.randn(3, STEPS, TINY["d_model"])
    together = moe(x)
    alone = torch.cat([moe(x[i : i + 1]) for i in range(3)])
    assert torch.allclose(together, alone, atol=1e-6)


def test_the_router_receives_a_gradient():
    """Without the gate scaling, top-1's hard choice would leave it untrained."""
    torch.manual_seed(0)
    moe = MoE(ModelConfig(**TINY, n_experts=4))
    moe(torch.randn(2, STEPS, TINY["d_model"])).sum().backward()
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.abs().sum() > 0


def test_balancing_loss_exists_for_moe_and_is_zero_for_dense():
    torch.manual_seed(0)
    ids = torch.randint(0, TINY["vocab_size"], (2, STEPS))
    moe = Transformer(ModelConfig(**TINY, n_experts=4))
    moe(ids)
    aux = moe.aux_loss()
    assert torch.is_tensor(aux) and torch.isfinite(aux) and aux > 0
    dense = Transformer(ModelConfig(**TINY))
    dense(ids)
    assert dense.aux_loss() == 0.0


def test_s25_moe_has_the_parameter_count_adr_0009_recorded():
    """76,571,648: three extra experts per layer plus a 4-way router."""
    assert Transformer(ModelConfig(n_experts=4)).n_params() == 76_571_648


@pytest.mark.parametrize("bad", [{"n_experts": 0}, {"window": -1}])
def test_invalid_ablation_settings_are_refused(bad):
    with pytest.raises(ValueError):
        ModelConfig(**bad)
