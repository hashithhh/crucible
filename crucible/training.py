"""Phase 3.2: the training configuration, LR schedule and parameter groups.

Everything here is deliberately free of loops and side effects, so the parts
of training that are easiest to get quietly wrong can be tested directly. The
loop that uses them is `scripts/train.py`.

Every constant comes from ADR-0008. None of them are chosen here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

TRAIN_TOKENS = 429_998_810  # data/tokens/meta.json, measured 2026-09-26


@dataclass(frozen=True)
class TrainConfig:
    """One run's constants. Values and reasoning: ADR-0008."""

    # Data shape
    context: int = 512  # ADR-0006
    batch_size: int = 128  # 65,536 tokens / context
    micro_batch: int = 16  # measured fastest on the local 4050
    steps: int = 6_561  # one epoch of the measured corpus

    # Optimizer
    peak_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Run control
    seed: int = 1337
    eval_every: int = 250
    checkpoint_every: int = 500

    def __post_init__(self) -> None:
        if self.batch_size % self.micro_batch:
            raise ValueError(
                f"batch_size {self.batch_size} is not a whole number of "
                f"micro-batches of {self.micro_batch}"
            )
        if self.warmup_steps >= self.steps:
            raise ValueError(
                f"warmup_steps {self.warmup_steps} must be fewer than the "
                f"{self.steps} steps it warms up into"
            )
        if self.min_lr > self.peak_lr:
            raise ValueError(f"min_lr {self.min_lr} exceeds peak_lr {self.peak_lr}")

    @property
    def accumulation(self) -> int:
        """Micro-batches per optimizer step."""
        return self.batch_size // self.micro_batch

    @property
    def batch_tokens(self) -> int:
        return self.batch_size * self.context

    @property
    def total_tokens(self) -> int:
        return self.batch_tokens * self.steps

    def shrunk(self, **changes) -> TrainConfig:
        """A copy with fields replaced. For smoke runs and tests."""
        return replace(self, **changes)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate for optimizer step `step`, counting from 0.

    CONTRACT (ADR-0008, Schedule)
    - Linear warmup over `warmup_steps`, reaching exactly `peak_lr` on the
      last warmup step. The first step is not zero: a zero-LR step is a
      wasted forward and backward pass.
    - Cosine decay from `peak_lr` to exactly `min_lr` at step `steps`. Landing
      on `min_lr` at the horizon is the point of fixing `steps` in advance;
      Chinchilla's result is that a cosine which does not land is worse.
    - Flat at `min_lr` beyond `steps`. A run extended by accident degrades
      gently rather than following the cosine back upward.
    - Monotonically non-increasing after warmup, and never outside
      [min_lr, peak_lr].
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if step < cfg.warmup_steps:
        # +1 so step 0 gets a real, if small, learning rate.
        return cfg.peak_lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.peak_lr - cfg.min_lr)


def param_groups(model, weight_decay: float) -> list[dict]:
    """Split parameters into decayed and undecayed groups (ADR-0008).

    CONTRACT
    - Tensors of rank 2 or higher are decayed; rank 0 and 1 are not. Weight
      decay on a normalisation gain or a bias shrinks a shift rather than a
      direction, which is not what the regulariser is for.
    - Parameters with `requires_grad=False` are excluded entirely, so a frozen
      tensor cannot be decayed by accident.
    - Every trainable parameter appears exactly once. Tied weights are one
      tensor and are counted once, which is why this walks
      `named_parameters()` rather than modules.
    - Both groups are always returned, even when one is empty, so downstream
      code sees a stable shape.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def describe(cfg: TrainConfig) -> str:
    """One-line summary for the run log."""
    return (
        f"{cfg.steps:,} steps x {cfg.batch_tokens:,} tokens "
        f"= {cfg.total_tokens / 1e6:.1f}M "
        f"({cfg.micro_batch} x {cfg.accumulation} accum), "
        f"lr {cfg.peak_lr:g} -> {cfg.min_lr:g} after {cfg.warmup_steps} warmup"
    )
