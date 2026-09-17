"""Vocabulary-size sweep for ADR-0001.

    python scripts/sweep_vocab.py            # writes results/vocab_sweep.{csv,json,png}

For each vocabulary size: train on TRAIN_BYTES of TinyStories-valid, measure
bytes/token on a disjoint held-out slice, time training, and compute the
embedding cost. Every constant below is justified in docs/adr/0001-vocab-size.md.

Corpus handling (ADR-0001, "Method"):
- The file is split into stories on `<|endoftext|>` lines. The separator is
  removed: it will be a special token in training, one id regardless of the
  vocabulary, so letting BPE learn its text would distort both the merges and
  bytes/token. The first story is dropped because the file starts mid-story.
- Training stories are taken from the start of the file, held-out stories from
  the end. They never overlap.
- A size the training text cannot supply (ADR-0003: merges exhausted) is
  recorded as unreachable, and the corpus's measured ceiling is added as its
  own row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import statistics
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crucible.data import VALID, load_corpus  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

VOCAB_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768]
TRAIN_BYTES = 5_000_000
HELDOUT_BYTES = 500_000
D_MODEL = 768
PARAM_BUDGET = 100_000_000
SEPARATOR = "<|endoftext|>"
TIMING_REPEATS = 3  # median of 3; single runs varied ~3% in ADR-0005 benchmarks
EXHAUSTION_PROBE_VOCAB = 1_000_000  # far above any reachable size; forces ADR-0003

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Reference palette (dataviz skill, references/palette.md), light mode.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
SERIES = "#2a78d6"
CAPTION_WRAP_CHARS = 190  # fits an 11-inch figure at 7.5 pt

# The size decided in docs/adr/0001-vocab-size.md. The figure rings this, not
# the chord knee: ADR-0001 found the knee unstable and decided on token budget.
ADR_0001_VOCAB = 2048


# --- corpus --------------------------------------------------------------------


def split_stories(text: str) -> list[str]:
    stories, lines = [], []
    for line in text.split("\n"):
        if line == SEPARATOR:
            stories.append("\n".join(lines))
            lines = []
        else:
            lines.append(line)
    if lines:
        stories.append("\n".join(lines))
    return [s for s in stories[1:] if s.strip()]  # [0] starts mid-story


def take_bytes(stories: list[str], budget: int) -> int:
    """How many stories fit in `budget` bytes when joined with newlines."""
    used = 0
    for k, s in enumerate(stories):
        cost = len(s.encode("utf-8")) + (1 if k else 0)
        if used + cost > budget:
            return k
        used += cost
    return len(stories)


def build_splits() -> tuple[str, str, dict]:
    stories = split_stories(load_corpus())
    n_train = take_bytes(stories, TRAIN_BYTES)
    n_held = take_bytes(stories[::-1], HELDOUT_BYTES)
    held_start = len(stories) - n_held
    if n_train > held_start:
        raise RuntimeError("train and held-out slices overlap; corpus too small")
    train = "\n".join(stories[:n_train])
    heldout = "\n".join(stories[held_start:])
    meta = {
        "corpus_file": VALID.name,
        "corpus_sha256": VALID.sha256,
        "stories_total": len(stories),
        "train_stories": n_train,
        "train_bytes": len(train.encode("utf-8")),
        "heldout_stories": n_held,
        "heldout_bytes": len(heldout.encode("utf-8")),
        "heldout_first_story_index": held_start,
    }
    return train, heldout, meta


# --- measurement ---------------------------------------------------------------


def merge_ceiling(train: str) -> int:
    """Merges the training text supports (ADR-0003), read from the exhaustion."""
    try:
        Tokenizer().train(train, EXHAUSTION_PROBE_VOCAB)
    except ValueError as exc:
        m = re.search(r"exhausted after (\d+) merges", str(exc))
        if m:
            return int(m.group(1))
        raise
    raise RuntimeError(f"corpus did not exhaust below {EXHAUSTION_PROBE_VOCAB}")


def measure(train: str, heldout: str, vocab: int, status: str) -> dict:
    row = {
        "vocab_size": vocab,
        "status": status,
        "embedding_params": vocab * D_MODEL,
        "embedding_share_of_budget": vocab * D_MODEL / PARAM_BUDGET,
        "train_seconds_median": "",
        "train_seconds_runs": "",
        "encode_heldout_seconds": "",
        "heldout_tokens": "",
        "bytes_per_token": "",
    }
    if status == "unreachable":
        return row

    times, encodings = [], []
    for _ in range(TIMING_REPEATS):
        tok = Tokenizer()
        t0 = time.perf_counter()
        tok.train(train, vocab)
        times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        encodings.append(tok.encode(heldout))
        enc_seconds = time.perf_counter() - t0
    if any(e != encodings[0] for e in encodings):
        raise RuntimeError(f"non-deterministic encoding at vocab {vocab}")

    n_bytes = len(heldout.encode("utf-8"))
    row.update(
        train_seconds_median=round(statistics.median(times), 3),
        train_seconds_runs=" ".join(f"{t:.3f}" for t in times),
        encode_heldout_seconds=round(enc_seconds, 3),
        heldout_tokens=len(encodings[0]),
        bytes_per_token=round(n_bytes / len(encodings[0]), 4),
    )
    return row


def chord_knee(xs: list[float], ys: list[float]) -> int:
    """Index of the point at maximum distance above the endpoint chord.

    Normalise both axes to [0, 1] using the first and last points, then take
    the point furthest above the straight line joining them. This is only the
    core idea of Kneedle (after Satopaa et al., 2011): no smoothing, no
    sensitivity parameter.

    Because the chord is defined by the first and last points, the endpoints
    have outsized leverage: adding or removing one can move the knee.
    ADR-0001 records how much.
    """
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    diffs = [
        (y - y0) / (y1 - y0) - (x - x0) / (x1 - x0) for x, y in zip(xs, ys, strict=True)
    ]
    return max(range(len(diffs)), key=diffs.__getitem__)


# --- hardware ------------------------------------------------------------------


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    if sys.platform == "darwin":
        import subprocess

        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown CPU"


def ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        names = ["total_phys", "avail_phys", "total_page", "avail_page"]
        names += ["total_virtual", "avail_virtual", "avail_ext_virtual"]

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)]
            _fields_ += [(n, ctypes.c_ulonglong) for n in names]

        s = MemStatus()
        s.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.total_phys / 2**30
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def hardware() -> dict:
    ram = ram_gb()
    return {
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(ram, 1) if ram else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "threads_used": 1,
    }


# --- figure --------------------------------------------------------------------


def _style_axis(ax, sizes: list[int], ceiling_vocab: int) -> None:
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(axis="x", labelrotation=45, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Vocabulary size (log scale)")
    ax.axvline(ceiling_vocab, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))


def _line(ax, xs, ys) -> None:
    ax.plot(
        xs,
        ys,
        color=SERIES,
        linewidth=2,
        marker="o",
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        zorder=3,
    )


def _ring(ax, x, y) -> None:
    ax.plot(
        [x],
        [y],
        marker="o",
        markersize=13,
        markerfacecolor="none",
        markeredgecolor=INK,
        markeredgewidth=1.5,
        zorder=4,
    )


def plot(rows, chosen_vocab, ceiling_vocab, meta, hw, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
        }
    )
    rows = sorted(rows, key=lambda r: r["vocab_size"])
    measured = [r for r in rows if r["bytes_per_token"] != ""]
    sizes = [r["vocab_size"] for r in rows]
    chosen = next(r for r in measured if r["vocab_size"] == chosen_vocab)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    for ax in (ax1, ax2):
        # Ticks at the requested grid only; the ceiling is labelled on its rule.
        _style_axis(ax, VOCAB_SIZES, ceiling_vocab)

    # Panel 1: compression, measured sizes only.
    xs = [r["vocab_size"] for r in measured]
    _line(ax1, xs, [r["bytes_per_token"] for r in measured])
    _ring(ax1, chosen_vocab, chosen["bytes_per_token"])
    ax1.annotate(
        f"ADR-0001: {chosen_vocab:,}\n{chosen['bytes_per_token']:.2f} bytes/token",
        (chosen_vocab, chosen["bytes_per_token"]),
        xytext=(12, -34),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    ax1.set_ylabel("Held-out bytes per token (higher = better)")
    ax1.set_title("Compression", loc="left", color=INK, fontweight="bold")
    lo, hi = ax1.get_ylim()
    label = f" corpus ceiling\n {ceiling_vocab:,}"
    ax1.text(ceiling_vocab, lo + 0.04 * (hi - lo), label, color=INK_2, fontsize=8)

    # Panel 2: embedding share. Pure arithmetic, so every size is shown.
    _line(ax2, sizes, [100 * r["embedding_share_of_budget"] for r in rows])
    chosen_share = 100 * chosen_vocab * D_MODEL / PARAM_BUDGET
    _ring(ax2, chosen_vocab, chosen_share)
    ax2.annotate(
        f"{chosen_vocab:,}: {chosen_share:.1f}%\n"
        f"({chosen_vocab * D_MODEL / 1e6:.1f}M params)",
        (chosen_vocab, chosen_share),
        xytext=(-14, 20),
        textcoords="offset points",
        ha="right",
        color=INK,
        fontsize=9,
    )
    ax2.set_ylabel(f"Input embedding, % of {PARAM_BUDGET / 1e6:.0f}M budget")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.set_title(
        f"Embedding cost at d_model={D_MODEL}",
        loc="left",
        color=INK,
        fontweight="bold",
    )

    unreachable = ", ".join(
        f"{r['vocab_size']:,}" for r in rows if r["status"] == "unreachable"
    )
    ram = f", {hw['ram_gb']} GB RAM" if hw["ram_gb"] else ""
    caption = (
        f"TinyStories-valid: trained on {meta['train_bytes'] / 1e6:.2f} MB "
        f"({meta['train_stories']:,} stories); bytes/token on a disjoint "
        f"{meta['heldout_bytes'] / 1e3:.0f} KB held-out slice "
        f"({meta['heldout_stories']:,} stories); <|endoftext|> removed; "
        f"cl100k pre-tokenization (ADR-0005). "
        f"Ringed: {chosen_vocab:,}, chosen in ADR-0001 (the sweep narrows the choice "
        f"to 2,048 or 4,096; the training-token budget decides). "
        f"{unreachable} exceed this corpus's merge ceiling (ADR-0003) and have no "
        f"compression point. Hardware: {hw['cpu']} ({hw['logical_cpus']} logical "
        f"CPUs; training is single-threaded){ram}; {hw['os']}; "
        f"Python {hw['python']}."
    )
    caption = textwrap.fill(caption, width=CAPTION_WRAP_CHARS)
    fig.text(0.012, 0.015, caption, color=INK_2, fontsize=7.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --- main ----------------------------------------------------------------------


def replot(out: Path) -> int:
    """Re-draw the figure from saved results; no training.

    Hardware and corpus metadata come from the JSON, i.e. the machine and
    sample the sweep actually ran on, not the machine doing the re-draw.
    """
    summary = json.loads((out / "vocab_sweep.json").read_text(encoding="utf-8"))
    rows = []
    with (out / "vocab_sweep.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bpt = r["bytes_per_token"]
            rows.append(
                {
                    "vocab_size": int(r["vocab_size"]),
                    "status": r["status"],
                    "bytes_per_token": float(bpt) if bpt else "",
                    "embedding_share_of_budget": float(r["embedding_share_of_budget"]),
                }
            )
    png = out / "vocab_sweep.png"
    ceiling = summary["ceiling_vocab_size"]
    plot(rows, ADR_0001_VOCAB, ceiling, summary["corpus"], summary["hardware"], png)
    print(f"re-drew {png} from saved results (no training)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--replot",
        action="store_true",
        help="re-draw the figure from the saved CSV and JSON in --out; no training",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        return replot(args.out)

    hw = hardware()
    train, heldout, meta = build_splits()
    print(
        f"train {meta['train_bytes']:,} B / {meta['train_stories']:,} stories; "
        f"held-out {meta['heldout_bytes']:,} B / {meta['heldout_stories']:,} stories"
    )

    Tokenizer().encode("warm up")  # build the pre-tokenization pattern, untimed

    ceiling = merge_ceiling(train)
    ceiling_vocab = 256 + ceiling
    print(f"merge ceiling: {ceiling:,} merges -> max vocab {ceiling_vocab:,}")

    rows = []
    for vocab in VOCAB_SIZES:
        status = "measured" if vocab <= ceiling_vocab else "unreachable"
        rows.append(measure(train, heldout, vocab, status))
    if ceiling_vocab not in VOCAB_SIZES:
        rows.append(measure(train, heldout, ceiling_vocab, "ceiling"))
    rows.sort(key=lambda r: r["vocab_size"])

    prev = None
    for r in rows:
        r["gain_vs_previous_pct"] = ""
        if r["bytes_per_token"] == "":
            continue
        if prev is not None:
            gain = r["bytes_per_token"] / prev["bytes_per_token"] - 1
            r["gain_vs_previous_pct"] = round(100 * gain, 2)
        prev = r

    for r in rows:
        print(
            f"  {r['vocab_size']:>6,}  {r['status']:<11}  "
            f"bytes/token={r['bytes_per_token']!s:<7}  "
            f"gain={r['gain_vs_previous_pct']!s:<6}  "
            f"train={r['train_seconds_median']!s}s"
        )

    # Knee over the requested sizes that could be measured. The ceiling row is
    # reported but excluded: it is a property of this corpus sample, not a grid
    # point, and including it would move the knee whenever TRAIN_BYTES changes.
    grid = [r for r in rows if r["status"] == "measured"]
    k = chord_knee(
        [math.log2(r["vocab_size"]) for r in grid],
        [r["bytes_per_token"] for r in grid],
    )
    knee_vocab = grid[k]["vocab_size"]

    fields = [
        "vocab_size",
        "status",
        "bytes_per_token",
        "gain_vs_previous_pct",
        "heldout_tokens",
        "train_seconds_median",
        "train_seconds_runs",
        "encode_heldout_seconds",
        "embedding_params",
        "embedding_share_of_budget",
    ]
    csv_path = args.out / "vocab_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows({key: r[key] for key in fields} for r in rows)

    summary = {
        "knee_vocab_size": knee_vocab,
        "knee_method": "max distance above endpoint chord, (log2 vocab, bytes/token), "
        "measured grid sizes",
        "merge_ceiling": ceiling,
        "ceiling_vocab_size": ceiling_vocab,
        "constants": {
            "vocab_sizes": VOCAB_SIZES,
            "train_bytes_budget": TRAIN_BYTES,
            "heldout_bytes_budget": HELDOUT_BYTES,
            "d_model": D_MODEL,
            "param_budget": PARAM_BUDGET,
            "timing_repeats": TIMING_REPEATS,
        },
        "corpus": meta,
        "hardware": hw,
        "date": time.strftime("%Y-%m-%d"),
    }
    json_path = args.out / "vocab_sweep.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    png = args.out / "vocab_sweep.png"
    plot(rows, ADR_0001_VOCAB, ceiling_vocab, meta, hw, png)
    print(f"knee: {knee_vocab:,}\nwrote {csv_path}\nwrote {json_path}\nwrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
