"""Item 11: the MFU script's throughput must survive the run it measures.

The first version took the median of the log's `tokens_per_s` field, which is a
cumulative average since the process started. On 2026-09-27 the machine slept
for 285 minutes mid-run; every record after that read ~3,200 tokens/s while the
GPU was doing ~42,000, and the MFU figure would have come out several times too
low. These tests pin the replacement: rates from consecutive records, never
paired across a resume.
"""

from __future__ import annotations

import json

import pytest

from crucible.model import ModelConfig
from scripts.measure_mfu import flops_per_token, throughput_from_log

BATCH = 65_536  # tokens per optimizer step, as ADR-0008
LOG_EVERY = 10  # steps between logged records, as scripts/train.py
RATE = 40_000.0  # the true loop speed the synthetic log encodes


def segment(start_step: int, n: int, *, stall_after: int | None = None) -> list[dict]:
    """One process lifetime: a run record, then step records at RATE.

    `seconds` restarts at 0, as it does in scripts/train.py after a resume.
    `tokens_per_s` is filled in the way train.py fills it -- cumulative since
    the process started -- so the stall poisons it exactly as it did for real.
    """
    records: list[dict] = [{"kind": "run", "start_step": start_step}]
    seconds = 0.0
    interval = LOG_EVERY * BATCH / RATE
    for i in range(n):
        step = start_step + i * LOG_EVERY
        if i:
            seconds += interval
        if stall_after is not None and i == stall_after:
            seconds += 285 * 60  # the laptop slept
        done = step - start_step + 1
        records.append(
            {
                "kind": "step",
                "step": step,
                "tokens": (step + 1) * BATCH,
                "seconds": round(seconds, 2),
                "tokens_per_s": round(done * BATCH / max(seconds, 1e-9)),
            }
        )
    return records


def write(tmp_path, records):
    path = tmp_path / "train_log.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_a_long_stall_does_not_drag_the_rate_down(tmp_path):
    records = segment(0, 60, stall_after=15)
    path = write(tmp_path, records)

    rate, _ = throughput_from_log(path)
    assert rate == pytest.approx(RATE, rel=0.01)

    # The control: the cumulative field this replaced really is poisoned.
    cumulative = sorted(r["tokens_per_s"] for r in records if r["kind"] == "step")
    assert cumulative[len(cumulative) // 2] < RATE / 3


def test_intervals_are_not_formed_across_a_resume(tmp_path):
    """`seconds` restarts at 0 on resume; pairing across it would be nonsense."""
    first = segment(0, 20)
    # The resumed process starts from an earlier checkpoint, so its first
    # steps repeat ones the killed process already logged, as happened for
    # real at steps 1,000-1,220.
    second = segment(100, 20)
    path = write(tmp_path, first + second)

    rate, detail = throughput_from_log(path)
    assert rate == pytest.approx(RATE, rel=0.01)
    # 19 intervals inside each segment and none across the boundary.
    assert detail.startswith("38 ")


def test_too_little_log_returns_none_rather_than_a_guess(tmp_path):
    path = write(tmp_path, segment(0, 2))
    assert throughput_from_log(path) is None


def test_a_missing_log_returns_none(tmp_path):
    assert throughput_from_log(tmp_path / "absent.jsonl") is None


def test_flops_per_token_is_6n_plus_the_attention_term():
    cfg = ModelConfig()
    params = 26_223_616
    attention = 12 * cfg.n_layers * cfg.context * cfg.d_model
    assert flops_per_token(cfg, params) == 6 * params + attention
    # S25: ~157.3M from the weights plus ~25.2M from attention.
    assert flops_per_token(cfg, params) == pytest.approx(182.5e6, rel=1e-3)
