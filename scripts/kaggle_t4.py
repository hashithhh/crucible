"""Phase 3, item 11: the T4 measurements, in one script for a Kaggle notebook.

In a Kaggle notebook with the "GPU T4" accelerator:

    !git clone https://github.com/hashithhh/crucible.git
    %cd crucible
    !python scripts/kaggle_t4.py

then download `results/t4_check.json` and commit it. Needs no dataset and no
tokenizer: every token here is synthetic, because nothing measured depends on
what the tokens say.

WHAT IT CHECKS, all of which ADR-0006 or ADR-0008 asserted without measuring:

1. The T4 has no bf16, so `pick_precision` picks fp16 + a loss scaler on its
   own. Until now that choice had only been forced on a card that has bf16.
2. A short S25 training run on that automatic path stays finite with the
   scaler live.
3. ADR-0008's projected Kaggle micro-batch of 64 fits in 16 GB.
4. Throughput, and MFU two ways: against a matmul ceiling measured on this T4,
   and against the 65 TFLOP/s datasheet peak that `model_budget.py` uses. The
   second is the one to feed back into it:

       python scripts/model_budget.py --t4-mfu <mfu_vs_datasheet>

It refuses to write `results/t4_check.json` on anything but a T4 unless given
--dry-run, so a laptop run can never be mistaken for the Kaggle number.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crucible.model import ModelConfig, Transformer  # noqa: E402
from crucible.shards import DTYPE  # noqa: E402
from crucible.training import TRAIN_TOKENS, TrainConfig  # noqa: E402
from scripts.measure_mfu import (  # noqa: E402
    benchmark,
    flops_per_token,
    matmul_ceiling,
)
from scripts.train import pick_precision, train  # noqa: E402

T4_DATASHEET_FLOPS = 65e12  # scripts/model_budget.py, T4_PEAK_FLOPS
SMOKE_STEPS = 30
SYNTHETIC_TOKENS = {"train": 2_000_000, "val": 200_000}
MICRO_BATCHES = (16, 64)  # ADR-0008: local measured, T4 projected


def synthetic_tokens(directory: Path, vocab: int) -> None:
    """Random-id shards in the real on-disk format, for the smoke run."""
    rng = np.random.default_rng(0)
    shards: dict = {"tokens_per_shard": max(SYNTHETIC_TOKENS.values())}
    for split, size in SYNTHETIC_TOKENS.items():
        ids = rng.integers(0, vocab, size=size, dtype=np.int64).astype(DTYPE)
        name = f"{split}_00000.bin"
        (directory / name).write_bytes(ids.tobytes())
        shards[split] = [name]
    meta = {
        "vocab_size": vocab,
        "eot_id": vocab - 1,
        "dtype": "uint16",
        "tokens": dict(SYNTHETIC_TOKENS),
        "shards": shards,
        "date": time.strftime("%Y-%m-%d"),
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def smoke(device: torch.device, work: Path) -> dict:
    """A short S25 run on the automatically chosen precision."""
    model_cfg = ModelConfig()
    synthetic_tokens(work, model_cfg.vocab_size)
    cfg = TrainConfig().shrunk(
        steps=SMOKE_STEPS,
        warmup_steps=5,
        batch_size=32,
        micro_batch=16,
        eval_every=0,
        checkpoint_every=0,
    )
    records = train(
        cfg,
        model_cfg,
        tokens_dir=work,
        ckpt_dir=work / "ckpt",
        log_path=work / "log.jsonl",
        device=device,
        eval_batches=2,
        log_every=1,
    )
    scales = [r["loss_scale"] for r in records if "loss_scale" in r]
    drops = sum(1 for a, b in zip(scales, scales[1:], strict=False) if b < a)
    return {
        "steps": len(records),
        "losses_finite": all(math.isfinite(r["loss"]) for r in records),
        "first_loss": round(records[0]["loss"], 4),
        "last_loss": round(records[-1]["loss"], 4),
        "scaler_live": bool(scales),
        "loss_scale_last": scales[-1] if scales else None,
        "loss_scale_drops": drops,
    }


def throughput_by_micro_batch(
    device: torch.device, dtype: torch.dtype
) -> tuple[dict, dict, float, int]:
    """Peak memory and tokens/s at each micro-batch; the FASTEST one wins.

    Not the largest. A dry run on a 6 GB card showed why: micro-batch 64 did
    not fail, it spilled past device memory into system RAM through the
    Windows driver's fallback and ran 5.6x slower. "It ran" is not "it fits",
    so a run whose peak exceeds the card's memory is marked as spilled and
    excluded, and among the rest the fastest is reported.
    """
    capacity = torch.cuda.get_device_properties(device).total_memory
    memory: dict = {}
    rates: dict = {}
    for micro in MICRO_BATCHES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = Transformer(ModelConfig()).to(device)
        cfg = TrainConfig().shrunk(micro_batch=micro)
        try:
            rate, _ = benchmark(model, cfg, device, dtype)
        except torch.OutOfMemoryError:
            memory[str(micro)] = "out of memory"
            continue
        finally:
            del model
        peak = torch.cuda.max_memory_allocated(device)
        if peak > capacity:
            memory[str(micro)] = (
                f"spilled: {peak / 1e9:.2f} GB on a {capacity / 1e9:.2f} GB card"
            )
            continue
        memory[str(micro)] = round(peak / 1e9, 2)
        rates[str(micro)] = round(rate)
    if not rates:
        raise RuntimeError("no micro-batch fitted in device memory; nothing to report")
    best = max(rates, key=lambda k: rates[k])
    return memory, rates, float(rates[best]), int(best)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "t4_check.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="allow a non-T4 GPU; the result is labelled and is not the T4 number",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("no GPU: in Kaggle, set Accelerator to 'GPU T4'", file=sys.stderr)
        return 1
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    is_t4 = "T4" in name
    if not is_t4 and not args.dry_run:
        print(
            f"refusing: this is a {name}, not a T4. The point of this script is "
            "the T4's own numbers. Use --dry-run (with --out elsewhere) to test.",
            file=sys.stderr,
        )
        return 2

    dtype, needs_scaler = pick_precision(device)
    scaler_word = "on" if needs_scaler else "off"
    print(f"{name}: auto precision {dtype}, loss scaler {scaler_word}")

    with tempfile.TemporaryDirectory() as tmp:
        smoke_result = smoke(device, Path(tmp))
    print(f"smoke: {smoke_result}")

    memory, rates, tokens_per_s, micro = throughput_by_micro_batch(device, dtype)
    params = Transformer(ModelConfig()).n_params()
    per_token = flops_per_token(ModelConfig(), params)
    achieved = tokens_per_s * per_token
    ceiling = matmul_ceiling(device, dtype)

    record = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": name,
        "is_t4": is_t4,
        "dry_run": args.dry_run,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "auto_precision": str(dtype).replace("torch.", ""),
        "loss_scaler": needs_scaler,
        "smoke": smoke_result,
        "peak_memory_gb_by_micro_batch": memory,
        "tokens_per_s_by_micro_batch": rates,
        "micro_batch_used": micro,
        "tokens_per_s": round(tokens_per_s),
        "flops_per_token": per_token,
        "achieved_tflops": round(achieved / 1e12, 3),
        "ceiling_tflops_measured": round(ceiling / 1e12, 3),
        "mfu_vs_measured": round(achieved / ceiling, 4),
        "mfu_vs_datasheet": round(achieved / T4_DATASHEET_FLOPS, 4),
        "s25_epoch_hours": round(TRAIN_TOKENS / tokens_per_s / 3600, 2),
        "adr_0006_assumed_band_vs_datasheet": [0.25, 0.40],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(record, indent=2))
    print(f"\nwrote {args.out}")
    print(f"next: python scripts/model_budget.py --t4-mfu {record['mfu_vs_datasheet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
