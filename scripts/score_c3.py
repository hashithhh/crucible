"""C3: score the trained model's held-out loss against the registered bar.

    python scripts/score_c3.py                    # checkpoints/final.pt
    python scripts/score_c3.py --checkpoint checkpoints/step_003000.pt --dry-run

THE THRESHOLDS ARE READ FROM THE LEDGER, NOT STORED HERE. C3 registered them
on 2026-09-26 before the run started, and the whole value of that registration
is that the bar cannot move once a number exists. A copy in this file would be
a second bar, free to drift toward whatever the model happened to achieve. If
the ledger's three bounds cannot be parsed in their registered form, this
refuses to score.

It also refuses to score a checkpoint that stopped short of the horizon unless
asked for a dry run: C3 is registered against the final checkpoint at
ADR-0008's 6,561 steps, not against the best number seen along the way.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crucible.model import ModelConfig, Transformer  # noqa: E402
from crucible.shards import ShardSet  # noqa: E402
from crucible.training import TrainConfig  # noqa: E402
from scripts.train import evaluate, pick_precision  # noqa: E402

LEDGER = ROOT / "results" / "LEDGER.md"
DEFAULT_CKPT = ROOT / "checkpoints" / "final.pt"
DEFAULT_TOKENS = ROOT / "data" / "tokens"

BYTES_PER_TOKEN = 3.7333  # results/tokenizer_train.json, measured held out
LN2 = 0.6931471805599453

# The exact lines C3 registered. Matching the prose is deliberate: the scorer
# breaks loudly if the entry is reworded, rather than scoring against a stale
# copy of a bar that somebody edited.
PATTERNS = {
    "hard_fail_at": r"\*\*Hard fail: ≥ ([\d.]+)\*\*",
    "pass_at": r"\*\*Pass: ≤ ([\d.]+)\*\*",
    "stretch_at": r"\*\*Stretch, recorded but NOT gating: ≤ ([\d.]+)\*\*",
}
BASELINES = {"uniform": 7.6251, "unigram": 6.0734, "bigram": 3.6190}


def parse_thresholds(ledger: Path = LEDGER) -> dict[str, float]:
    """C3's three registered bounds, read out of the ledger itself."""
    if not ledger.is_file():
        raise FileNotFoundError(f"{ledger} not found; C3's thresholds live there")
    text = ledger.read_text(encoding="utf-8")
    start = text.find("# C3 ")
    if start < 0:
        raise ValueError(f"{ledger.name} has no C3 entry")
    end = text.find("# C4 ", start)
    section = text[start : end if end > 0 else len(text)]

    found: dict[str, float] = {}
    for name, pattern in PATTERNS.items():
        match = re.search(pattern, section)
        if match is None:
            raise ValueError(
                f"C3 no longer states its {name.replace('_', ' ')} in the "
                "registered form. Refusing to score against an unreadable bar."
            )
        found[name] = float(match.group(1))

    if not found["stretch_at"] < found["pass_at"] < found["hard_fail_at"]:
        raise ValueError(f"C3's thresholds are not ordered: {found}")
    return found


def verdict(loss: float, bounds: dict[str, float]) -> str:
    if loss >= bounds["hard_fail_at"]:
        return "hard fail"
    if loss <= bounds["pass_at"]:
        return "pass"
    return "fail"


def bits_per_byte(nats: float) -> float:
    return nats / LN2 / BYTES_PER_TOKEN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="score a mid-run checkpoint; the result is not a C3 result",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    bounds = parse_thresholds()
    if not args.checkpoint.is_file():
        print(f"no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    horizon = blob["train_config"]["steps"]
    if blob["step"] < horizon and not args.dry_run:
        print(
            f"refusing to score: {args.checkpoint.name} is at step "
            f"{blob['step']:,} of {horizon:,}. C3 is registered against the "
            "final checkpoint at the horizon, not the best number along the "
            "way. Use --dry-run to look early.",
            file=sys.stderr,
        )
        return 2

    device = torch.device(args.device)
    model = Transformer(ModelConfig(**blob["model_config"]))
    model.load_state_dict(blob["model"])
    model = model.to(device).eval()
    dtype, _ = pick_precision(device)

    cfg = TrainConfig(**blob["train_config"])
    val = ShardSet.open(args.tokens, "val")
    print(f"evaluating {args.checkpoint.name} on {len(val):,} held-out tokens...")
    loss = evaluate(model, val, cfg, device, dtype)
    call = verdict(loss, bounds)
    met_stretch = loss <= bounds["stretch_at"]

    bpb = bits_per_byte(loss)
    print(f"\nheld-out loss   {loss:.4f} nats/token  ({bpb:.3f} bits/byte)")
    for name, value in BASELINES.items():
        print(f"  vs {name:<8}   {value:.4f}  ({loss / value - 1:+.1%})")
    print(f"\npass at         {bounds['pass_at']}")
    print(f"hard fail at    {bounds['hard_fail_at']}  (the bigram baseline)")
    stretch_note = "met" if met_stretch else "not met"
    print(f"stretch         {bounds['stretch_at']}  {stretch_note}")
    tail = "  (dry run, not a C3 result)" if args.dry_run else ""
    print(f"\nC3: {call.upper()}{tail}")

    record = {
        "date": time.strftime("%Y-%m-%d"),
        "checkpoint": args.checkpoint.name,
        "step": blob["step"],
        "horizon": horizon,
        "dry_run": args.dry_run,
        "val_tokens": len(val),
        "val_loss": round(loss, 4),
        "bits_per_byte": round(bpb, 4),
        "baselines": BASELINES,
        **bounds,
        "met_stretch": met_stretch,
        "verdict": call,
        "thresholds_from": "results/LEDGER.md, C3, registered 2026-09-26",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "c3_result.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out / 'c3_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
