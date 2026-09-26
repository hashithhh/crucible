"""Phase 3.4: resume must be exact, not approximate.

The failure this guards against does not look like a failure. A run that
resumes with correct weights but a fresh data sampler carries on happily,
prints a believable loss curve, and is quietly a different run: it re-sees
batches it already trained on and never sees the ones it skipped. Only
comparing against an uninterrupted run catches it.

Everything here is tiny and runs on CPU in fp32, where the arithmetic is
reproducible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from crucible.model import ModelConfig
from crucible.shards import DTYPE
from crucible.training import TrainConfig
from scripts.train import pick_precision, train

VOCAB = 64
STEPS = 8
BREAK_AT = 4


@pytest.fixture
def tokens(tmp_path):
    """A small two-split corpus of random ids, written as real shards."""
    rng = np.random.default_rng(0)
    sizes = {"train": 4000, "val": 500}
    shards: dict = {}
    for split, size in sizes.items():
        ids = rng.integers(0, VOCAB, size=size, dtype=np.int64).astype(DTYPE)
        name = f"{split}_00000.bin"
        (tmp_path / name).write_bytes(ids.tobytes())
        shards[split] = [name]
    shards["tokens_per_shard"] = 2000
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "vocab_size": VOCAB,
                "eot_id": VOCAB - 1,
                "dtype": "uint16",
                "tokens": sizes,
                "shards": shards,
                "date": "2026-09-26",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def small():
    """A model and a run small enough to train twice inside a test."""
    model_cfg = ModelConfig(
        vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, context=16, d_ff=64
    )
    train_cfg = TrainConfig(
        context=16,
        batch_size=4,
        micro_batch=2,
        steps=STEPS,
        warmup_steps=2,
        eval_every=0,
        checkpoint_every=BREAK_AT,
    )
    return train_cfg, model_cfg


def run(tokens, cfg, model_cfg, tmp, *, resume=None, stop_at=None):
    return train(
        cfg,
        model_cfg,
        tokens_dir=tokens,
        ckpt_dir=tmp / "ckpt",
        log_path=tmp / "log.jsonl",
        device=torch.device("cpu"),
        resume=resume,
        stop_at=stop_at,
        log_every=1,
    )


def test_resuming_reproduces_the_uninterrupted_run(tokens, small, tmp_path):
    cfg, model_cfg = small
    straight = run(tokens, cfg, model_cfg, tmp_path / "a")

    run(tokens, cfg, model_cfg, tmp_path / "b", stop_at=BREAK_AT)
    resumed = run(
        tokens,
        cfg,
        model_cfg,
        tmp_path / "b",
        resume=tmp_path / "b" / "ckpt" / f"step_{BREAK_AT:06d}.pt",
    )

    assert [r["step"] for r in resumed] == list(range(BREAK_AT, STEPS))
    for after, before in zip(resumed, straight[BREAK_AT:], strict=True):
        assert after["step"] == before["step"]
        assert after["loss"] == pytest.approx(before["loss"], rel=1e-6, abs=1e-6)


def test_a_resume_that_forgets_the_sampler_diverges(tokens, small, tmp_path):
    """The control: without the sampler state the losses do NOT match.

    Without this, the test above could pass for the wrong reason -- a run
    short enough, or a model dull enough, that any two continuations agree.
    """
    cfg, model_cfg = small
    straight = run(tokens, cfg, model_cfg, tmp_path / "a")

    run(tokens, cfg, model_cfg, tmp_path / "b", stop_at=BREAK_AT)
    path = tmp_path / "b" / "ckpt" / f"step_{BREAK_AT:06d}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    # Rewind only the sampler; weights and optimizer state stay correct.
    blob["sampler"] = np.random.default_rng(cfg.seed).bit_generator.state
    torch.save(blob, path)

    resumed = run(tokens, cfg, model_cfg, tmp_path / "b", resume=path)
    losses = [r["loss"] for r in resumed]
    expected = [r["loss"] for r in straight[BREAK_AT:]]
    assert losses != pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_checkpoint_carries_what_a_resume_needs(tokens, small, tmp_path):
    cfg, model_cfg = small
    run(tokens, cfg, model_cfg, tmp_path / "a", stop_at=BREAK_AT)
    blob = torch.load(
        tmp_path / "a" / "ckpt" / f"step_{BREAK_AT:06d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    for key in ("step", "model", "optimizer", "scaler", "sampler", "torch_rng"):
        assert key in blob, f"checkpoint is missing {key}"
    assert blob["step"] == BREAK_AT
    # The configs travel too, so a checkpoint can be read without the script
    # that wrote it -- and the horizon it records is the RUN's, not the point
    # it paused at, which is what lets the schedule continue unchanged.
    assert blob["train_config"]["steps"] == STEPS
    assert blob["model_config"]["vocab_size"] == VOCAB


def test_the_log_records_what_phase_3_5_asks_for(tokens, small, tmp_path):
    cfg, model_cfg = small
    run(tokens, cfg, model_cfg, tmp_path / "a")
    text = (tmp_path / "a" / "log.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.strip().splitlines()]
    kinds = {line["kind"] for line in lines}
    assert {"run", "step", "done"} <= kinds
    step = next(line for line in lines if line["kind"] == "step")
    for field in ("loss", "lr", "grad_norm", "tokens", "tokens_per_s", "time"):
        assert field in step, f"step record is missing {field}"


def test_logged_gradient_norms_are_finite(tokens, small, tmp_path):
    """The logged norm is pre-clip, so it may exceed grad_clip.

    What must never appear is a non-finite norm: that is the signal ADR-0008
    ties to halving peak_lr, and it has to be trustworthy.
    """
    cfg, model_cfg = small
    records = run(tokens, cfg, model_cfg, tmp_path / "a")
    assert all(np.isfinite(r["grad_norm"]) for r in records)


def test_cpu_runs_in_fp32_without_a_scaler():
    """ADR-0008, Precision: the scaler is an fp16 remedy and nothing else."""
    dtype, needs_scaler = pick_precision(torch.device("cpu"))
    assert dtype is torch.float32
    assert needs_scaler is False
