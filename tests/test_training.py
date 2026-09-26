"""Phase 3.2: the LR schedule and parameter grouping.

A wrong schedule does not crash. It trains, produces a plausible loss curve,
and costs three hours before anyone notices the cosine never landed — so the
properties ADR-0008 actually relies on are asserted here rather than eyeballed
on a plot.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from crucible.training import TrainConfig, describe, lr_at, param_groups


@pytest.fixture
def cfg():
    """A short run with the real shape: warmup, then a cosine that lands."""
    return TrainConfig(steps=100, warmup_steps=10, peak_lr=1e-3, min_lr=1e-4)


def test_warmup_starts_above_zero(cfg):
    """A zero-LR first step is a forward and backward pass thrown away."""
    assert lr_at(0, cfg) > 0


def test_warmup_is_linear(cfg):
    lrs = [lr_at(s, cfg) for s in range(cfg.warmup_steps)]
    gaps = [b - a for a, b in zip(lrs, lrs[1:], strict=False)]
    assert all(g == pytest.approx(gaps[0]) for g in gaps)


def test_warmup_ends_exactly_at_peak(cfg):
    assert lr_at(cfg.warmup_steps - 1, cfg) == pytest.approx(cfg.peak_lr)


def test_cosine_lands_exactly_on_min_lr_at_the_horizon(cfg):
    """Chinchilla's finding: a cosine that does not land is worse."""
    assert lr_at(cfg.steps, cfg) == pytest.approx(cfg.min_lr)


def test_schedule_holds_at_min_lr_past_the_horizon(cfg):
    """An accidental over-run degrades gently; it does not turn back upward."""
    assert lr_at(cfg.steps + 1, cfg) == pytest.approx(cfg.min_lr)
    assert lr_at(cfg.steps * 10, cfg) == pytest.approx(cfg.min_lr)


def test_schedule_never_leaves_the_declared_band(cfg):
    for s in range(cfg.steps * 2):
        assert cfg.min_lr <= lr_at(s, cfg) <= cfg.peak_lr + 1e-12


def test_schedule_is_non_increasing_after_warmup(cfg):
    lrs = [lr_at(s, cfg) for s in range(cfg.warmup_steps - 1, cfg.steps + 1)]
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:], strict=False))


def test_schedule_is_increasing_during_warmup(cfg):
    lrs = [lr_at(s, cfg) for s in range(cfg.warmup_steps)]
    assert all(b > a for a, b in zip(lrs, lrs[1:], strict=False))


def test_cosine_midpoint_is_halfway_down(cfg):
    """cos(pi/2) = 0, so the halfway step sits at the midpoint of the band."""
    middle = cfg.warmup_steps + (cfg.steps - cfg.warmup_steps) // 2
    assert lr_at(middle, cfg) == pytest.approx((cfg.peak_lr + cfg.min_lr) / 2, rel=1e-2)


def test_negative_step_is_rejected(cfg):
    with pytest.raises(ValueError):
        lr_at(-1, cfg)


def test_config_rejects_a_batch_that_is_not_whole_micro_batches():
    with pytest.raises(ValueError, match="micro-batch"):
        TrainConfig(batch_size=100, micro_batch=16)


def test_config_rejects_warmup_longer_than_the_run():
    with pytest.raises(ValueError, match="warmup"):
        TrainConfig(steps=100, warmup_steps=100)


def test_config_rejects_a_floor_above_the_peak():
    with pytest.raises(ValueError, match="min_lr"):
        TrainConfig(peak_lr=1e-4, min_lr=1e-3)


def test_default_config_is_one_epoch_of_the_measured_corpus():
    """ADR-0008: steps x batch_tokens should cover the train split once."""
    from crucible.training import TRAIN_TOKENS

    cfg = TrainConfig()
    assert cfg.batch_tokens == 65_536
    assert cfg.accumulation == 8
    # The run must not overrun the split, nor fall a whole batch short of it.
    assert 0 <= TRAIN_TOKENS - cfg.total_tokens < cfg.batch_tokens


def test_describe_names_the_numbers_a_run_log_needs():
    text = describe(TrainConfig())
    for fragment in ("6,561", "65,536", "accum", "0.0006"):
        assert fragment in text


class Tiny(nn.Module):
    """A matrix, a gain vector and a frozen matrix: one of each case."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)  # rank 2 -> decayed
        self.gain = nn.Parameter(torch.ones(4))  # rank 1 -> not decayed
        self.frozen = nn.Parameter(torch.ones(4, 4), requires_grad=False)


def test_matrices_are_decayed_and_vectors_are_not():
    decay, no_decay = param_groups(Tiny(), 0.1)
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_frozen_parameters_are_excluded_from_both_groups():
    model = Tiny()
    grouped = [p for g in param_groups(model, 0.1) for p in g["params"]]
    assert all(p.requires_grad for p in grouped)
    assert all(id(p) != id(model.frozen) for p in grouped)


def test_every_trainable_parameter_is_grouped_exactly_once():
    model = Tiny()
    grouped = [p for g in param_groups(model, 0.1) for p in g["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(grouped) == len(trainable)
    assert {id(p) for p in grouped} == {id(p) for p in trainable}


def test_tied_weights_are_counted_once():
    """S25 ties the embedding to the output head (ADR-0006)."""
    from crucible.model import ModelConfig, Transformer

    model = Transformer(ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2))
    grouped = [p for g in param_groups(model, 0.1) for p in g["params"]]
    assert len({id(p) for p in grouped}) == len(grouped)


def test_groups_are_returned_even_when_one_is_empty():
    class OnlyVectors(nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = nn.Parameter(torch.ones(4))

    groups = param_groups(OnlyVectors(), 0.1)
    assert len(groups) == 2
    assert groups[0]["params"] == []
