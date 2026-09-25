"""Phase 3, step 1 — train the production tokenizer on the full corpus.

    python scripts/train_tokenizer.py                  # 1.9 GB, vocab 2,048
    python scripts/train_tokenizer.py --sample-mb 50   # quick dry run

ADR-0001 chose vocab 2,048 from a 5 MB sample and said explicitly not to
assume its 3.7258 bytes/token carries over. This trains on all of
TinyStories-train and re-measures on the same held-out slice ADR-0001 used,
so the two numbers are comparable.

Memory: `train(text)` would hold the corpus and every chunk of it at once,
which does not fit. This streams the file and merges chunk counts instead
(`train_from_counts`). Slices are cut only where the preceding character is
non-whitespace, the one rule that keeps counting-in-slices equal to counting
the whole corpus -- see tests/test_streaming_train.py, which also pins why
the obvious rule (cut at line ends) is silently wrong.

The `<|endoftext|>` separator is removed before counting and registered as a
special token afterwards, exactly as the ADR-0001 sweep treated it: it is one
id at any vocabulary size, so letting BPE learn its text would distort both
the merges and bytes/token.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sweep_vocab  # noqa: E402  (same held-out slice as ADR-0001)

from crucible.data import TRAIN  # noqa: E402
from tokenizer import Tokenizer, pretokenize  # noqa: E402

VOCAB_SIZE = sweep_vocab.ADR_0001_VOCAB  # 2,048
SEPARATOR_LINE = "\n<|endoftext|>\n"
SEPARATOR = "<|endoftext|>"
SLICE_BYTES = 32 << 20  # 32 MiB of text per slice
ADR_0001_BYTES_PER_TOKEN = 3.7258  # the 5 MB-sample number, for comparison

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = sweep_vocab.RESULTS_DIR


def safe_cut(text: str) -> int:
    """Largest index that can end a slice without changing any chunk.

    A boundary is safe when the character before it is not whitespace: that
    cannot split a whitespace run, nor a chunk carrying a leading space.
    Returns 0 if no safe point exists, and the caller then reads more.
    """
    for i in range(len(text) - 1, 0, -1):
        if not text[i - 1].isspace():
            return i
    return 0


def count_chunks(path: Path, limit: int | None = None) -> tuple[Counter[str], dict]:
    """Stream `path`, returning chunk counts and corpus statistics."""
    counts: Counter[str] = Counter()
    stats = {"text_bytes": 0, "separators": 0, "slices": 0}
    pending = ""
    read = 0

    def absorb(text: str) -> None:
        stats["separators"] += text.count(SEPARATOR)
        # Match the ADR-0001 sweep: stories joined by a single newline.
        text = text.replace(SEPARATOR_LINE, "\n").replace(SEPARATOR, "")
        stats["text_bytes"] += len(text.encode("utf-8"))
        counts.update(pretokenize(text))
        stats["slices"] += 1

    with path.open("r", encoding="utf-8") as f:
        while True:
            block = f.read(SLICE_BYTES)
            if not block:
                break
            read += len(block.encode("utf-8"))
            pending += block
            cut = safe_cut(pending)
            if cut:
                absorb(pending[:cut])
                pending = pending[cut:]
            if limit is not None and read >= limit:
                break

    if pending:
        absorb(pending)

    stats["chunk_types"] = len(counts)
    stats["chunk_total"] = sum(counts.values())
    return counts, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    parser.add_argument(
        "--sample-mb",
        type=float,
        default=None,
        help="stop after roughly this many MB of the corpus (dry run)",
    )
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    if not TRAIN.path.is_file():
        print(
            f"{TRAIN.path} not found. Run: python scripts/get_data.py --full",
            file=sys.stderr,
        )
        return 1

    limit = None if args.sample_mb is None else int(args.sample_mb * 1e6)
    label = "full corpus" if limit is None else f"first ~{args.sample_mb:.0f} MB"
    print(f"counting chunks over {TRAIN.name} ({label})", flush=True)

    t0 = time.perf_counter()
    counts, stats = count_chunks(TRAIN.path, limit)
    count_seconds = time.perf_counter() - t0
    print(
        f"  {stats['text_bytes'] / 1e9:.3f} GB of text in {stats['slices']} slices, "
        f"{stats['separators']:,} separators, {stats['chunk_types']:,} chunk types, "
        f"{stats['chunk_total']:,} chunks in {count_seconds:.1f}s",
        flush=True,
    )

    tok = Tokenizer()
    t0 = time.perf_counter()
    tok.train_from_counts(counts, args.vocab)
    train_seconds = time.perf_counter() - t0
    tok.register_special_tokens({SEPARATOR: args.vocab})
    print(
        f"  trained vocab {args.vocab} in {train_seconds:.1f}s; "
        f"vocab_size {tok.vocab_size} with the separator registered",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    prefix = args.out / f"tinystories-{args.vocab}"
    tok.save(str(prefix))

    # Re-measure on ADR-0001's held-out slice, so the numbers are comparable.
    _, heldout, meta = sweep_vocab.build_splits()
    held_bytes = len(heldout.encode("utf-8"))
    t0 = time.perf_counter()
    ids = tok.encode(heldout)
    encode_seconds = time.perf_counter() - t0
    bytes_per_token = held_bytes / len(ids)
    if tok.decode(ids) != heldout:
        raise RuntimeError("round-trip failed on the held-out slice")

    summary = {
        "vocab_size_learned": args.vocab,
        "vocab_size_with_specials": tok.vocab_size,
        "model": str(prefix) + ".model",
        "corpus": {
            "file": TRAIN.name,
            "sha256": TRAIN.sha256,
            "sample_mb": args.sample_mb,
        },
        "corpus_stats": stats,
        "count_seconds": round(count_seconds, 1),
        "train_seconds": round(train_seconds, 1),
        "heldout": {
            "source": "ADR-0001 held-out slice (TinyStories-valid, far end)",
            "bytes": held_bytes,
            "stories": meta["heldout_stories"],
            "tokens": len(ids),
            "bytes_per_token": round(bytes_per_token, 4),
            "encode_seconds": round(encode_seconds, 2),
        },
        "adr_0001_bytes_per_token": ADR_0001_BYTES_PER_TOKEN,
        "change_vs_adr_0001": round(bytes_per_token / ADR_0001_BYTES_PER_TOKEN - 1, 4),
        "hardware": sweep_vocab.hardware(),
        "python": platform.python_version(),
        "date": time.strftime("%Y-%m-%d"),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "tokenizer_train.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"  held-out: {bytes_per_token:.4f} bytes/token "
        f"({summary['change_vs_adr_0001']:+.2%} vs ADR-0001's 5 MB sample)"
    )
    print(f"wrote {prefix}.model\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
