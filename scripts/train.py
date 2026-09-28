"""Phase 3.2-3.5: train S25 on the token shards.

    python scripts/train.py                        # the full run, ADR-0008
    python scripts/train.py --smoke                # 20 tiny steps, for CI
    python scripts/train.py --resume checkpoints/step_000500.pt

AdamW with a cosine schedule and gradient clipping (3.2), mixed precision
with a loss scaler where the hardware needs one (3.3), checkpoint and exact
resume (3.4), and a JSONL log of every number worth plotting (3.5).

Every constant is ADR-0008's. Nothing is chosen here.

WHAT "EXACT RESUME" MEANS. Model weights and optimizer state are the obvious
half. The half usually forgotten is the *data order*: a run that resumes with
a fresh sampler trains on different batches from that point on and is a
different run, however healthy its loss curve looks. So the sampler's RNG
state goes into the checkpoint beside the weights, and
`tests/test_train_resume.py` asserts that resuming reproduces the losses the
uninterrupted run produced.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crucible.model import ModelConfig, Transformer  # noqa: E402
from crucible.shards import ShardSet, TokenBatches, eval_windows  # noqa: E402
from crucible.training import TrainConfig, describe, lr_at, param_groups  # noqa: E402

DEFAULT_TOKENS = ROOT / "data" / "tokens"
DEFAULT_CKPT = ROOT / "checkpoints"
DEFAULT_LOG = ROOT / "results" / "train_log.jsonl"


PRECISIONS = ("auto", "bf16", "fp16", "fp32")


def pick_precision(
    device: torch.device, force: str | None = None
) -> tuple[torch.dtype, bool]:
    """Choose the autocast dtype, and whether a loss scaler is needed.

    ADR-0008, Precision: bf16 where the hardware has it, fp16 plus a scaler
    where it does not (a T4 is Turing and does not). The scaler exists because
    fp16's 5 exponent bits let small gradients underflow to zero; bf16 has
    fp32's exponent range and so needs none.

    `force` overrides the choice. It exists so the fp16 + scaler path -- the
    one a T4 takes -- can be exercised on a card that would otherwise pick
    bf16. Whatever is forced, fp16 always comes with a scaler: fp16 without
    one is the configuration ADR-0008 says loses gradients.
    """
    if force not in (None, *PRECISIONS):
        raise ValueError(f"precision must be one of {PRECISIONS}, got {force!r}")
    if force in (None, "auto"):
        if device.type != "cuda":
            return torch.float32, False
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        return torch.float16, True
    if force == "fp32":
        return torch.float32, False
    if device.type != "cuda":
        raise ValueError(f"{force} autocast needs a CUDA device, got {device.type}")
    if force == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 forced on a GPU without bf16 support")
        return torch.bfloat16, False
    return torch.float16, True


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JsonlLog:
    """Append-only run log. Phase 3.5 asks for a file, not just stdout."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, **record) -> None:
        record.setdefault("time", now())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


@torch.no_grad()
def evaluate(
    model: Transformer,
    shards: ShardSet,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int | None = None,
) -> float:
    """Mean cross-entropy over the held-out split, in nats per token.

    Walks the split in fixed order (`eval_windows`) so the number moves only
    when the model does. Weighted by token count, so a short final batch could
    not skew the mean even if one appeared.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x_np, y_np in eval_windows(shards, cfg.micro_batch, cfg.context, max_batches):
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(x)
        # Reduce the loss in fp32 whatever the forward ran in (ADR-0008).
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1)
        )
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    if was_training:
        model.train()
    return total_loss / max(total_tokens, 1)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batches: TokenBatches,
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    best_val: float | None,
) -> None:
    """Everything needed to continue this run, and nothing else.

    The sampler state is the entry that makes resume exact rather than merely
    plausible; see this module's docstring.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "sampler": batches.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "train_config": vars(cfg),
            "model_config": vars(model_cfg),
            "best_val": best_val,
            "saved": now(),
        },
        tmp,
    )
    # Rename last: a crash mid-write leaves the previous checkpoint intact
    # rather than a truncated file that loads as garbage.
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batches: TokenBatches,
) -> tuple[int, float | None]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    scaler.load_state_dict(blob["scaler"])
    batches.set_state(blob["sampler"])
    torch.set_rng_state(blob["torch_rng"].cpu())
    if blob.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in blob["cuda_rng"]])
    return blob["step"], blob.get("best_val")


def train(
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    *,
    tokens_dir: Path,
    ckpt_dir: Path,
    log_path: Path,
    device: torch.device,
    resume: Path | None = None,
    stop_at: int | None = None,
    precision: str | None = None,
    eval_batches: int | None = None,
    log_every: int = 10,
) -> list[dict]:
    """Run the loop. Returns the per-step records, which tests assert on.

    `stop_at` pauses the run early **without moving the horizon**: the cosine
    still lands at `cfg.steps`, so a session that ends after 2,000 of 6,561
    steps and resumes tomorrow follows the same schedule as one that never
    stopped. Shortening `cfg.steps` instead would silently compress the
    schedule and make the two runs incomparable, which is exactly the bug
    this parameter exists to avoid.
    """
    torch.manual_seed(cfg.seed)
    dtype, needs_scaler = pick_precision(device, precision)

    train_shards = ShardSet.open(tokens_dir, "train")
    val_shards = ShardSet.open(tokens_dir, "val")
    batches = TokenBatches(train_shards, cfg.micro_batch, cfg.context, cfg.seed)

    model = Transformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        param_groups(model, cfg.weight_decay),
        lr=cfg.peak_lr,
        betas=cfg.betas,
        eps=cfg.eps,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=needs_scaler)
    log = JsonlLog(log_path)

    start_step, best_val = 0, None
    if resume is not None:
        start_step, best_val = load_checkpoint(
            resume, model=model, optimizer=optimizer, scaler=scaler, batches=batches
        )

    log.write(
        kind="run",
        resumed_from=str(resume) if resume else None,
        start_step=start_step,
        params=model.n_params(),
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        loss_scaler=needs_scaler,
        config=describe(cfg),
    )
    print(f"{describe(cfg)}\n{model.n_params():,} params on {device}, {dtype}")
    if needs_scaler:
        print("fp16: loss scaler on (ADR-0008, Precision)")

    last = cfg.steps if stop_at is None else min(cfg.steps, stop_at)
    model.train()
    records: list[dict] = []
    started = time.monotonic()
    for step in range(start_step, last):
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_aux = 0.0
        for _ in range(cfg.accumulation):
            x_np, y_np = batches.next()
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            with torch.autocast(
                device.type, dtype=dtype, enabled=dtype != torch.float32
            ):
                logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.size(-1)), y.reshape(-1)
            )
            # Divide before backward so the accumulated gradient is the mean
            # over the whole batch, not the sum over its micro-batches.
            # The MoE balancing loss (ADR-0009) joins the backward pass only;
            # the logged loss stays pure cross-entropy, so it is comparable
            # with dense runs. aux_loss() is 0.0 for a dense model.
            aux = model.aux_loss()
            scaler.scale((loss + aux) / cfg.accumulation).backward()
            step_loss += loss.item() / cfg.accumulation
            step_aux += float(aux) / cfg.accumulation

        # Unscale before clipping: clipping a scaled gradient would clip
        # against a threshold that moves with the scaler (ADR-0008).
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        record = {
            "kind": "step",
            "step": step,
            "lr": lr,
            "loss": step_loss,
            "grad_norm": float(grad_norm),
            **({"aux_loss": step_aux} if model_cfg.n_experts > 1 else {}),
            "tokens": (step + 1) * cfg.batch_tokens,
            "seconds": round(time.monotonic() - started, 2),
        }
        if device.type == "cuda":
            record["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        if needs_scaler:
            # A drop between records means the scaler skipped a step on an
            # overflow and halved -- expected under fp16 (ADR-0008).
            record["loss_scale"] = scaler.get_scale()
        records.append(record)

        if step % log_every == 0 or step == last - 1:
            elapsed = time.monotonic() - started
            done = step - start_step + 1
            record["tokens_per_s"] = round(done * cfg.batch_tokens / max(elapsed, 1e-9))
            log.write(**record)
            print(
                f"  step {step:>6}  loss {step_loss:6.3f}  lr {lr:.2e}  "
                f"|g| {float(grad_norm):6.3f}  {record['tokens_per_s']:,} tok/s",
                flush=True,
            )
        if not math.isfinite(step_loss):
            log.write(kind="abort", step=step, reason="loss is not finite")
            raise RuntimeError(f"loss went non-finite at step {step}")

        if cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            val = evaluate(model, val_shards, cfg, device, dtype, eval_batches)
            best_val = val if best_val is None else min(best_val, val)
            log.write(kind="eval", step=step, val_loss=val, best_val=best_val)
            print(f"  step {step:>6}  val {val:.4f}  (best {best_val:.4f})")

        if cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step + 1:06d}.pt",
                step=step + 1,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                batches=batches,
                cfg=cfg,
                model_cfg=model_cfg,
                best_val=best_val,
            )

    final = evaluate(model, val_shards, cfg, device, dtype, eval_batches)
    finished = last >= cfg.steps
    save_checkpoint(
        ckpt_dir / ("final.pt" if finished else f"step_{last:06d}.pt"),
        step=last,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        batches=batches,
        cfg=cfg,
        model_cfg=model_cfg,
        best_val=final if best_val is None else min(best_val, final),
    )
    log.write(
        kind="done" if finished else "paused",
        step=last,
        val_loss=final,
        seconds=round(time.monotonic() - started, 1),
    )
    word = "final" if finished else "paused at"
    print(f"\n{word} val loss {final:.4f} after {last:,} of {cfg.steps:,} steps")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--steps", type=int, help="override ADR-0008's step count")
    parser.add_argument("--micro-batch", type=int, help="override for this machine")
    parser.add_argument("--eval-batches", type=int, help="cap the held-out pass")
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default="auto",
        help="override ADR-0008's hardware-based choice; fp16 always gets a scaler",
    )
    parser.add_argument(
        "--stop-at",
        type=int,
        help="pause after this step without moving the schedule's horizon",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="20 tiny steps: proves the loop runs, measures nothing",
    )
    args = parser.parse_args(argv)

    cfg = TrainConfig()
    if args.smoke:
        cfg = cfg.shrunk(
            steps=20,
            warmup_steps=5,
            batch_size=8,
            micro_batch=4,
            eval_every=10,
            checkpoint_every=0,
        )
    if args.steps is not None:
        cfg = cfg.shrunk(steps=args.steps)
    if args.micro_batch is not None:
        cfg = cfg.shrunk(micro_batch=args.micro_batch)

    train(
        cfg,
        ModelConfig(),
        tokens_dir=args.tokens,
        ckpt_dir=args.checkpoints,
        log_path=args.log,
        device=torch.device(args.device),
        resume=args.resume,
        stop_at=args.stop_at,
        precision=args.precision,
        eval_batches=4 if args.smoke else args.eval_batches,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
