"""Phase 3, item 11: measure model FLOPs utilisation for S25.

    python scripts/measure_mfu.py                 # uses results/train_log.jsonl
    python scripts/measure_mfu.py --benchmark     # times the model directly

ADR-0006 assumed a 25-40% MFU band when it budgeted T4 hours, and recorded
measuring it as the first thing Phase 3 owes. This measures it.

WHY THE DENOMINATOR IS MEASURED, NOT QUOTED. MFU needs a peak FLOP/s figure,
and vendor numbers are published both with and without sparsity, so the same
run can be reported at wildly different utilisations depending which is picked.
This measures the ceiling instead: large dense matmuls in the same dtype the
run uses, on this device. That is a number this machine actually reached, which
makes the resulting MFU a statement about the training loop rather than about a
spec sheet.

WHAT IT CANNOT DO. There is no Turing GPU here, so the T4 half of item 11 is
not measured and is not guessed at. What this gives is the local figure, and an
explicit record that the Kaggle number is still outstanding.
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

from crucible.model import ModelConfig, Transformer  # noqa: E402
from crucible.training import TrainConfig  # noqa: E402

DEFAULT_LOG = ROOT / "results" / "train_log.jsonl"
MATMUL_N = 8192  # square matmuls big enough to be tensor-core bound
MATMUL_REPS = 30


def flops_per_token(model_cfg: ModelConfig, params: int) -> int:
    """Forward+backward FLOPs for one token.

    The PaLM/nanoGPT convention: 6 x parameters for the weight matmuls (2 for
    the forward, 4 for the backward), plus the attention term that scales with
    sequence length, 12 x layers x context x d_model, which the parameter count
    cannot capture because it has no weights.

    The tied embedding/output tensor is counted once, which is correct: the
    embedding side is a lookup and costs no matmul, the output side is a matmul.
    """
    attention = 12 * model_cfg.n_layers * model_cfg.context * model_cfg.d_model
    return 6 * params + attention


def matmul_ceiling(device: torch.device, dtype: torch.dtype) -> float:
    """The largest dense matmul throughput this device reaches, in FLOP/s."""
    a = torch.randn(MATMUL_N, MATMUL_N, device=device, dtype=dtype)
    b = torch.randn(MATMUL_N, MATMUL_N, device=device, dtype=dtype)
    for _ in range(3):  # warm up: the first calls include kernel selection
        a @ b
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(MATMUL_REPS):
        a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # 2 FLOPs per multiply-accumulate, N^3 of them.
    return MATMUL_REPS * 2 * MATMUL_N**3 / elapsed


def throughput_from_log(path: Path) -> tuple[float, str] | None:
    """Median tokens/s over the logged steps of the real run.

    The median rather than the mean: the first logged step carries CUDA context
    setup and the steps beside an eval sit next to a held-out pass. Neither
    should set the figure the whole run is judged by.
    """
    if not path.is_file():
        return None
    rates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "step" and "tokens_per_s" in record:
            rates.append(record["tokens_per_s"])
    if len(rates) < 3:
        return None
    rates.sort()
    return float(rates[len(rates) // 2]), f"{len(rates)} logged steps"


def benchmark(
    model: Transformer, cfg: TrainConfig, device: torch.device, dtype: torch.dtype
) -> tuple[float, str]:
    """Time the real training step shape, forward and backward."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(
        0, model.config.vocab_size, (cfg.micro_batch, cfg.context), device=device
    )
    steps = 12
    started = time.perf_counter()
    for i in range(steps + 3):
        if i == 3:  # discard warmup
            torch.cuda.synchronize()
            started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=dtype):
            logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), ids.reshape(-1)
        )
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tokens = steps * cfg.micro_batch * cfg.context
    return tokens / elapsed, f"{steps} timed steps at micro-batch {cfg.micro_batch}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="time the model directly instead of reading the run log",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "mfu.json")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type != "cuda":
        print("MFU on CPU is not meaningful; this needs a GPU", file=sys.stderr)
        return 1

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cfg, model_cfg = TrainConfig(), ModelConfig()
    model = Transformer(model_cfg).to(device)
    params = model.n_params()
    per_token = flops_per_token(model_cfg, params)

    measured = None if args.benchmark else throughput_from_log(args.log)
    if measured is None:
        tokens_per_s, detail = benchmark(model, cfg, device, dtype)
        source = f"live benchmark, {detail}"
    else:
        tokens_per_s, detail = measured
        source = f"{args.log.name}, median of {detail}"

    ceiling = matmul_ceiling(device, dtype)
    achieved = tokens_per_s * per_token
    mfu = achieved / ceiling

    name = torch.cuda.get_device_name(device)
    print(f"device        {name}")
    print(f"dtype         {dtype}")
    print(f"params        {params:,}")
    print(f"FLOPs/token   {per_token / 1e6:.1f}M  (6N + attention term)")
    print(f"throughput    {tokens_per_s:,.0f} tokens/s   [{source}]")
    print(f"achieved      {achieved / 1e12:.2f} TFLOP/s")
    print(f"matmul ceil   {ceiling / 1e12:.2f} TFLOP/s  (measured, dense, this device)")
    print(f"MFU           {mfu:.1%}")
    band_low, band_high = 0.25, 0.40
    if mfu < band_low:
        print(
            f"\nADR-0006 assumed {band_low:.0%}-{band_high:.0%} when it budgeted "
            f"T4 hours. This is {mfu:.1%}, BELOW that band, so its hour "
            f"estimates are optimistic by roughly {band_low / mfu:.1f}x."
        )

    record = {
        "date": time.strftime("%Y-%m-%d"),
        "gpu": name,
        "dtype": str(dtype).replace("torch.", ""),
        "params": params,
        "flops_per_token": per_token,
        "tokens_per_s": round(tokens_per_s),
        "achieved_tflops": round(achieved / 1e12, 3),
        "matmul_ceiling_tflops": round(ceiling / 1e12, 3),
        "mfu": round(mfu, 4),
        "source": source,
        "adr_0006_assumed_band": [band_low, band_high],
        "t4_measured": False,
        "note": (
            "T4 MFU is NOT measured here: no Turing GPU on this machine. "
            "Item 11 of the Phase 3 definition of done stays open for the "
            "Kaggle half."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
