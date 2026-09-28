"""Phase 5: train the scaling ladder at ~20 tokens/param and fit a power law.

    python scripts/scale.py              # train whatever has not finished; resumable
    python scripts/scale.py --summary    # just refit and rewrite the results

Three new runs (S3, S7, S13); S25 is the Phase 3 run, read from
`results/c3_result.json`. Design and caveats: ADR-0010.

The fit is L = E + K * N^(-gamma) over the four points. ADR-0010's prediction
from Chinchilla is gamma in [0.28, 0.34] along the 20:1 frontier. Four points
and three parameters leave one degree of freedom, so gamma is reported with
that caveat attached.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crucible.model import ModelConfig, Transformer  # noqa: E402
from crucible.training import TrainConfig  # noqa: E402
from scripts.ablate import final_record  # noqa: E402
from scripts.measure_mfu import throughput_from_log  # noqa: E402
from scripts.train import train  # noqa: E402

TOKENS_PER_PARAM = 20  # Chinchilla's compute-optimal ratio, ADR-0010
CHINCHILLA_BAND = (0.28, 0.34)  # ADR-0010: beta and alpha, approach 3
FIT_GRID = 4000  # candidate values of E tried by the fit
OUT = ROOT / "results" / "phase5"
CKPT = ROOT / "checkpoints" / "phase5"
TOKENS = ROOT / "data" / "tokens"

# ADR-0006 §3 shapes: head_dim 64, MLP 4 x d_model.
SIZES = {
    "S3": ModelConfig(d_model=256, n_layers=4, n_heads=4, d_ff=1024),
    "S7": ModelConfig(d_model=320, n_layers=6, n_heads=5, d_ff=1280),
    "S13": ModelConfig(d_model=384, n_layers=8, n_heads=6, d_ff=1536),
}
S25_PARAMS = 26_223_616


def params_of(model_cfg: ModelConfig) -> int:
    return Transformer(model_cfg).n_params()


def plan(model_cfg: ModelConfig) -> TrainConfig:
    base = TrainConfig()
    steps = round(TOKENS_PER_PARAM * params_of(model_cfg) / base.batch_tokens)
    return base.shrunk(steps=steps, eval_every=0, checkpoint_every=steps // 2)


def fit_power_law(n: list[float], loss: list[float]) -> dict:
    """Fit L = E + K * N^(-gamma): a grid over E, log-linear for K and gamma.

    For each candidate irreducible loss E below the smallest observed loss,
    log(L - E) is linear in log N; the E whose fit has the least squared error
    in loss space wins. Numpy only, and deterministic.
    """
    n_arr = np.asarray(n, dtype=float)
    l_arr = np.asarray(loss, dtype=float)
    best: dict | None = None
    for e in np.linspace(0.0, l_arr.min() - 1e-4, FIT_GRID):
        slope, intercept = np.polyfit(np.log(n_arr), np.log(l_arr - e), 1)
        pred = e + np.exp(intercept) * n_arr**slope
        sse = float(((pred - l_arr) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {
                "E": float(e),
                "K": float(np.exp(intercept)),
                "gamma": -float(slope),
                "sse": sse,
            }
    if best is None:
        raise ValueError("no fit found")
    return best


def sensitivity(points: list[dict], fit: dict) -> dict:
    """How far the exponent moves with the floor, since one degree of freedom
    lets E and gamma trade off. Also the fit without S25, the one point not
    trained at exactly 20 tokens/param (ADR-0010)."""
    n = np.asarray([p["params"] for p in points], dtype=float)
    loss = np.asarray([p["val_loss"] for p in points], dtype=float)
    fixed = []
    for e in (0.0, 0.5, 0.8, 1.0):
        slope, intercept = np.polyfit(np.log(n), np.log(loss - e), 1)
        pred = e + np.exp(intercept) * n**slope
        rms = float(np.sqrt(((pred - loss) ** 2).mean()))
        fixed.append({"E_fixed": e, "gamma": round(-slope, 3), "rms": round(rms, 4)})
    best_rms = float(np.sqrt(fit["sse"] / len(points)))
    at_20 = [i for i, p in enumerate(points) if p["size"] != "S25"]
    sub = fit_power_law(n[at_20].tolist(), loss[at_20].tolist())
    return {
        "fits_with_E_fixed": fixed,
        "best_fit_rms": round(best_rms, 4),
        "without_s25": {"E": round(sub["E"], 3), "gamma": round(sub["gamma"], 3)},
    }


def run(name: str, device: torch.device) -> dict:
    result_path = OUT / f"{name}.json"
    if result_path.is_file():
        print(f"{name}: already done, skipping")
        return json.loads(result_path.read_text(encoding="utf-8"))

    model_cfg = SIZES[name]
    cfg = plan(model_cfg)
    log = OUT / f"{name}.jsonl"
    found = sorted((CKPT / name).glob("step_*.pt"))
    resume = found[-1] if found else None
    how = f"resuming from {resume.name}" if resume else "fresh"
    print(f"\n=== {name} === {cfg.steps:,} steps, {how}")
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
        "size": name,
        "params": params_of(model_cfg),
        "steps": cfg.steps,
        "tokens": cfg.total_tokens,
        "val_loss": round(done["val_loss"], 4),
        "tokens_per_s": round(rate[0]) if rate else None,
        "seconds_last_process": round(time.monotonic() - started, 1),
        "resumed": resume is not None,
        "date": time.strftime("%Y-%m-%d"),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: held-out {result['val_loss']}")
    return result


def summarise() -> dict | None:
    points = []
    for name in SIZES:
        path = OUT / f"{name}.json"
        if not path.is_file():
            print(f"summary waits for {name}")
            return None
        r = json.loads(path.read_text(encoding="utf-8"))
        points.append({k: r[k] for k in ("size", "params", "tokens", "val_loss")})
    c3_path = ROOT / "results" / "c3_result.json"
    c3 = json.loads(c3_path.read_text(encoding="utf-8"))
    points.append(
        {
            "size": "S25",
            "params": S25_PARAMS,
            "tokens": c3["step"] * TrainConfig().batch_tokens,
            "val_loss": c3["val_loss"],
        }
    )

    fit = fit_power_law([p["params"] for p in points], [p["val_loss"] for p in points])
    losses = [p["val_loss"] for p in points]
    gains = [round(a - b, 4) for a, b in zip(losses, losses[1:], strict=False)]
    low, high = CHINCHILLA_BAND
    summary = {
        "protocol": "ADR-0010: ~20 tokens/param per size; S25 from Phase 3 (16.5)",
        "points": [
            {**p, "tokens_per_param": round(p["tokens"] / p["params"], 1)}
            for p in points
        ],
        "fit": {k: round(v, 4) for k, v in fit.items()},
        "chinchilla_band": list(CHINCHILLA_BAND),
        "in_band": low <= fit["gamma"] <= high,
        "loss_falls_with_size": all(g > 0 for g in gains),
        "gains_shrink": all(b < a for a, b in zip(gains, gains[1:], strict=False)),
        "gains_between_sizes": gains,
        "caveat": "4 points, 3 fitted parameters: one degree of freedom.",
        "sensitivity": sensitivity(points, fit),
        "date": time.strftime("%Y-%m-%d"),
    }
    summary["verdict"] = (
        "Chinchilla effective-exponent prediction "
        + ("confirmed" if summary["in_band"] else "NOT confirmed")
        + f" (gamma {fit['gamma']:.2f} vs [{low}, {high}])"
    )
    path = ROOT / "results" / "phase5_scaling.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for p in summary["points"]:
        size, ratio, loss = p["size"], p["tokens_per_param"], p["val_loss"]
        print(f"  {size:<4} {p['params']:>11,}  {ratio:>5}/param  {loss:.4f}")
    f = summary["fit"]
    band = summary["in_band"]
    print(f"fit: L = {f['E']} + {f['K']} * N^-{f['gamma']}   in band: {band}")
    print(f"wrote {path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="store_true", help="only refit")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.summary:
        device = torch.device(args.device)
        for name in SIZES:
            run(name, device)
    summarise()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
