"""Phase 4: run ADR-0009's six ablation runs and summarise them.

    python scripts/ablate.py              # run whatever has not finished yet
    python scripts/ablate.py --only moe4  # one run
    python scripts/ablate.py --summary    # just rebuild the results table

Each run is 820 steps (1/8 epoch) on ADR-0008's config with one thing
changed. A finished run leaves `results/phase4/<run>.json` and is skipped next
time; an unfinished one resumes from its latest checkpoint, because this
machine's memory pressure stopped three runs in Phase 3.

The summary applies ADR-0009's rule, fixed before any run: the gap between
the two baseline seeds is the noise floor, and a variant closer to the
baseline than that is reported as "no difference".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crucible.model import ModelConfig  # noqa: E402
from crucible.training import TrainConfig  # noqa: E402
from scripts.measure_mfu import throughput_from_log  # noqa: E402
from scripts.train import train  # noqa: E402

STEPS = 820  # ADR-0009: 1/8 of ADR-0008's 6,561
CHECKPOINT_EVERY = 410  # halfway, so a stopped run loses at most ~10 minutes
OUT = ROOT / "results" / "phase4"
CKPT = ROOT / "checkpoints" / "phase4"
TOKENS = ROOT / "data" / "tokens"

BASE = TrainConfig().shrunk(
    steps=STEPS, eval_every=0, checkpoint_every=CHECKPOINT_EVERY
)

# ADR-0009, Variants. The two baselines come first: no verdict can be given
# until the noise floor exists.
RUNS: dict[str, tuple[TrainConfig, ModelConfig]] = {
    "baseline": (BASE, ModelConfig()),
    "baseline_seed1338": (BASE.shrunk(seed=1338), ModelConfig()),
    "lr_2e-4": (BASE.shrunk(peak_lr=2e-4, min_lr=2e-5), ModelConfig()),
    "no_warmup": (BASE.shrunk(warmup_steps=0), ModelConfig()),
    "moe4": (BASE, ModelConfig(n_experts=4)),
    "hybrid128": (BASE, ModelConfig(window=128)),
}


def latest_checkpoint(name: str) -> Path | None:
    found = sorted((CKPT / name).glob("step_*.pt"))
    return found[-1] if found else None


def final_record(log: Path) -> dict | None:
    done = None
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("kind") == "done":
                done = record
    return done


def run(name: str, device: torch.device) -> dict:
    cfg, model_cfg = RUNS[name]
    result_path = OUT / f"{name}.json"
    if result_path.is_file():
        print(f"{name}: already done, skipping")
        return json.loads(result_path.read_text(encoding="utf-8"))

    log = OUT / f"{name}.jsonl"
    resume = latest_checkpoint(name)
    how = f"resuming from {resume.name}" if resume else "fresh"
    print(f"\n=== {name} === {how}")
    started = time.monotonic()
    train(
        cfg,
        model_cfg,
        tokens_dir=TOKENS,
        ckpt_dir=CKPT / name,
        log_path=log,
        device=device,
        resume=resume,
    )
    done = final_record(log)
    if done is None:
        raise RuntimeError(f"{name}: training returned but logged no 'done' record")
    rate = throughput_from_log(log)
    result = {
        "run": name,
        "val_loss": round(done["val_loss"], 4),
        "tokens_per_s": round(rate[0]) if rate else None,
        "seconds_last_process": round(time.monotonic() - started, 1),
        "resumed": resume is not None,
        "train_config": vars(cfg),
        "model_config": vars(model_cfg),
        "date": time.strftime("%Y-%m-%d"),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: held-out {result['val_loss']}")
    return result


def summarise() -> dict | None:
    results = {}
    for name in RUNS:
        path = OUT / f"{name}.json"
        if path.is_file():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
    if "baseline" not in results or "baseline_seed1338" not in results:
        print("summary needs both baselines; not yet available")
        return None

    base = results["baseline"]["val_loss"]
    noise = abs(base - results["baseline_seed1338"]["val_loss"])
    variants = {}
    for name, r in results.items():
        if name.startswith("baseline"):
            continue
        diff = r["val_loss"] - base
        if abs(diff) <= noise:
            verdict = "no difference"
        else:
            verdict = "better" if diff < 0 else "worse"
        variants[name] = {
            "val_loss": r["val_loss"],
            "diff_vs_baseline": round(diff, 4),
            "verdict": verdict,
            "tokens_per_s": r["tokens_per_s"],
        }
    summary = {
        "protocol": "ADR-0009: 820 steps each, held-out loss at step 820",
        "baseline": base,
        "baseline_seed1338": results["baseline_seed1338"]["val_loss"],
        "noise_floor": round(noise, 4),
        "variants": variants,
        "missing": [n for n in RUNS if n not in results],
        "date": time.strftime("%Y-%m-%d"),
    }
    path = ROOT / "results" / "phase4_ablations.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    seed2 = summary["baseline_seed1338"]
    print(f"\nbaseline {base}   seed 1338 {seed2}   noise floor {noise:.4f}")
    for name, v in variants.items():
        diff, verdict = v["diff_vs_baseline"], v["verdict"]
        print(f"  {name:<12} {v['val_loss']:.4f}  {diff:+.4f}  {verdict}")
    print(f"wrote {path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=list(RUNS))
    parser.add_argument("--summary", action="store_true", help="only rebuild the table")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.summary:
        device = torch.device(args.device)
        for name in [args.only] if args.only else list(RUNS):
            run(name, device)
    summarise()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
