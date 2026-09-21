"""C1: our tokenizer at the ADR-0001 vocab vs tiktoken, on ADR-0001's held-out slice.

    python scripts/compare_tiktoken.py      # writes results/c1_tiktoken.json

Needs `pip install -e ".[compare]"` (tiktoken). tiktoken downloads its
encodings on first use.

NOT LIKE FOR LIKE. Ours is a 2,048-token vocabulary trained on 5 MB of
TinyStories. cl100k_base and o200k_base are pretrained ~100K and ~200K
vocabularies trained by OpenAI on far larger, broader corpora. For context the
script also reports ours at the largest vocabulary this training sample allows
(its ADR-0003 merge ceiling), which is still 7-14x smaller than tiktoken's.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sweep_vocab  # noqa: E402  (same split, hardware probe, constants)

from tokenizer import Tokenizer  # noqa: E402

ADR_0001_VOCAB = sweep_vocab.ADR_0001_VOCAB
ADR_0001_BYTES_PER_TOKEN = 3.7258  # results/vocab_sweep.csv; must reproduce
TIKTOKEN_ENCODINGS = ["cl100k_base", "o200k_base"]
THROUGHPUT_REPEATS = 5  # median; first call is a separate, untimed warm-up

# C1 as stated by the project owner on 2026-09-22 (the vault record was not
# found; see results/c1_tiktoken.md). "Worse" = extra tokens needed for the
# same text, relative to the reference encoding.
C1_TARGET_WITHIN = 0.10
C1_KILL_ABOVE = 0.30
C1_REFERENCE = "cl100k_base"

RESULTS_DIR = sweep_vocab.RESULTS_DIR


def time_encode(encode, text: str) -> tuple[list[int], float, list[float]]:
    ids = encode(text)  # warm-up (caches, lazy init); not timed
    runs = []
    for _ in range(THROUGHPUT_REPEATS):
        t0 = time.perf_counter()
        again = encode(text)
        runs.append(time.perf_counter() - t0)
        if again != ids:
            raise RuntimeError("non-deterministic encoding")
    return ids, statistics.median(runs), runs


def make_row(name, kind, vocab, timed, n_bytes) -> dict:
    ids, seconds, runs = timed
    return {
        "tokenizer": name,
        "kind": kind,
        "vocab_size": vocab,
        "tokens": len(ids),
        "bytes_per_token": round(n_bytes / len(ids), 4),
        "encode_seconds_median": round(seconds, 4),
        "encode_seconds_runs": [round(r, 4) for r in runs],
        "encode_mb_per_s": round(n_bytes / seconds / 1e6, 3),
        "encode_mtokens_per_s": round(len(ids) / seconds / 1e6, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    try:
        import tiktoken
    except ImportError:
        print('tiktoken not installed: pip install -e ".[compare]"', file=sys.stderr)
        return 1

    train, heldout, meta = sweep_vocab.build_splits()
    n_bytes = len(heldout.encode("utf-8"))
    rows = []

    ours = Tokenizer()
    ours.train(train, ADR_0001_VOCAB)
    timed = time_encode(ours.encode, heldout)
    rows.append(make_row("crucible", "ours, ADR-0001", ADR_0001_VOCAB, timed, n_bytes))
    if rows[0]["bytes_per_token"] != ADR_0001_BYTES_PER_TOKEN:
        raise RuntimeError(
            f"did not reproduce ADR-0001: {rows[0]['bytes_per_token']} "
            f"!= {ADR_0001_BYTES_PER_TOKEN}; the slice or tokenizer has changed"
        )

    ceiling_vocab = 256 + sweep_vocab.merge_ceiling(train)
    at_ceiling = Tokenizer()
    at_ceiling.train(train, ceiling_vocab)
    timed = time_encode(at_ceiling.encode, heldout)
    kind = "ours, sample ceiling (context only)"
    rows.append(make_row("crucible", kind, ceiling_vocab, timed, n_bytes))

    for name in TIKTOKEN_ENCODINGS:
        enc = tiktoken.get_encoding(name)
        # encode_ordinary: no special-token handling, matching our default
        # encode(); the slice contains no separators anyway.
        timed = time_encode(enc.encode_ordinary, heldout)
        kind = "tiktoken, pretrained"
        rows.append(make_row(name, kind, enc.n_vocab, timed, n_bytes))

    ref = next(r for r in rows if r["tokenizer"] == C1_REFERENCE)
    for r in rows:
        r["extra_tokens_vs_cl100k"] = round(r["tokens"] / ref["tokens"] - 1, 4)
        r["vocab_ratio_vs_ours"] = round(r["vocab_size"] / ADR_0001_VOCAB, 1)

    worse = rows[0]["extra_tokens_vs_cl100k"]
    if worse <= C1_TARGET_WITHIN:
        verdict = "meets target"
    elif worse > C1_KILL_ABOVE:
        verdict = "past kill threshold"
    else:
        verdict = "misses target, above kill threshold"

    summary = {
        "c1": {
            "reference": C1_REFERENCE,
            "metric": "extra tokens for the same text vs reference",
            "target_within": C1_TARGET_WITHIN,
            "kill_above": C1_KILL_ABOVE,
            "ours_extra_tokens": worse,
            "verdict_as_specified": verdict,
        },
        "rows": rows,
        "heldout": {k: meta[k] for k in ("heldout_bytes", "heldout_stories")},
        "corpus": {k: meta[k] for k in ("corpus_file", "corpus_sha256")},
        "tiktoken_version": tiktoken.__version__,
        "hardware": sweep_vocab.hardware(),
        "python": platform.python_version(),
        "date": time.strftime("%Y-%m-%d"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "c1_tiktoken.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for r in rows:
        print(
            f"{r['tokenizer']:<12} {r['kind']:<37} vocab={r['vocab_size']:>7,}  "
            f"bytes/token={r['bytes_per_token']:.4f}  "
            f"extra_vs_cl100k={r['extra_tokens_vs_cl100k']:+.1%}  "
            f"{r['encode_mb_per_s']:.3f} MB/s"
        )
    print(f"C1 ({C1_REFERENCE}): ours {worse:+.1%} tokens -> {verdict}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
