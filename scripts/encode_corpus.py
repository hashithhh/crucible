"""Phase 3.1: encode TinyStories into deduplicated uint16 token shards.

    python scripts/encode_corpus.py                    # full corpus, all cores
    python scripts/encode_corpus.py --input data/TinyStories-valid.txt \
        --out data/tokens-valid --limit 5000           # quick check

Stories are split on `<|endoftext|>`, deduplicated exactly, encoded in parallel,
and written as raw little-endian uint16 with no header, so a reader can memmap
them directly. Each story is followed by the separator id, which is what teaches
the model where a document ends.

WHY PARALLEL. C1 measured the encoder at 0.33 MB/s single-threaded: ~97 minutes
for 1.9 GB. Encoding is independent per story, so the work fans out across
cores; hashing and dedup stay in the parent, which is cheap and keeps the
duplicate set global.

TWO CONSTANTS ARE OWED TO ADR-0008. `TOKENS_PER_SHARD` and `VAL_FRACTION` are
defaults with reasons, not decisions: see their comments. CLAUDE.md requires an
ADR for each, and Phase 3 item 9 is where they land.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crucible.data import TRAIN  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

SEPARATOR = "<|endoftext|>"
DEFAULT_MODEL = ROOT / "models" / "tinystories-2048"
DEFAULT_OUT = ROOT / "data" / "tokens"
READ_BYTES = 32 << 20  # 32 MiB of text per read, as scripts/train_tokenizer.py

# 64M tokens = 128 MB per shard. Large enough that per-file overhead is
# irrelevant at 510M tokens (~8 shards), small enough to memmap one shard on a
# machine with 2-3 GB free. Owed to ADR-0008.
TOKENS_PER_SHARD = 64 << 20

# 0.5% of ~510M tokens is ~2.5M held-out tokens: enough that eval loss is
# stable between checkpoints, cheap enough that training loses nothing that
# matters. Owed to ADR-0008.
VAL_FRACTION = 0.005
VAL_BUCKETS = 100_000  # hash-bucket resolution for the split

_TOK: Tokenizer | None = None


def _init_worker(model_prefix: str) -> None:
    """Load one tokenizer per worker process, once."""
    global _TOK
    _TOK = Tokenizer()
    _TOK.load(model_prefix)


def _encode_batch(batch: list[str]) -> list[list[int]]:
    assert _TOK is not None, "worker not initialised"
    eot = _TOK.vocab_size - 1
    return [_TOK.encode(story) + [eot] for story in batch]


def story_digest(story: str) -> int:
    """Stable 8-byte digest. Used for both dedup and the train/val split."""
    raw = hashlib.blake2b(story.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(raw, "big")


def iter_stories(path: Path, limit: int | None = None):
    """Yield stories from a separator-delimited corpus, streaming.

    The tail of each read is carried forward so a story is never split across
    reads. Empty stories (consecutive separators, or the trailing newline at
    end of file) are skipped rather than encoded as documents of length 1.
    """
    carry = ""
    produced = 0
    with path.open("r", encoding="utf-8") as f:
        while True:
            text = f.read(READ_BYTES)
            if not text:
                break
            carry += text
            parts = carry.split(SEPARATOR)
            carry = parts.pop()  # incomplete tail
            for part in parts:
                story = part.strip("\n")
                if not story:
                    continue
                yield story
                produced += 1
                if limit is not None and produced >= limit:
                    return
    story = carry.strip("\n")
    if story and (limit is None or produced < limit):
        yield story


class ShardWriter:
    """Appends uint16 ids to rolling shard files."""

    def __init__(self, out_dir: Path, split: str, tokens_per_shard: int) -> None:
        self.out_dir = out_dir
        self.split = split
        self.tokens_per_shard = tokens_per_shard
        self.files: list[str] = []
        self.tokens = 0
        self._handle = None
        self._in_shard = 0

    def _roll(self) -> None:
        if self._handle is not None:
            self._handle.close()
        name = f"{self.split}_{len(self.files):05d}.bin"
        self.files.append(name)
        self._handle = (self.out_dir / name).open("wb")
        self._in_shard = 0

    def write(self, ids: list[int]) -> None:
        import numpy as np

        if self._handle is None:
            self._roll()
        arr = np.asarray(ids, dtype=np.uint16)
        view = memoryview(arr)
        start = 0
        while start < len(arr):
            room = self.tokens_per_shard - self._in_shard
            if room == 0:
                self._roll()
                room = self.tokens_per_shard
            take = min(room, len(arr) - start)
            self._handle.write(view[start : start + take])
            self._in_shard += take
            start += take
        self.tokens += len(arr)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def batched(iterable, size: int):
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=TRAIN.path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=0, help="0 = all cores")
    parser.add_argument("--batch", type=int, default=2000, help="stories per task")
    parser.add_argument("--limit", type=int, help="stop after N stories (testing)")
    parser.add_argument("--tokens-per-shard", type=int, default=TOKENS_PER_SHARD)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args(argv)

    try:
        import numpy  # noqa: F401
    except ImportError:
        print("numpy required: pip install numpy", file=sys.stderr)
        return 1
    if not args.input.is_file():
        print(f"no such corpus: {args.input}", file=sys.stderr)
        return 1
    if not Path(f"{args.model}.model").is_file():
        print(f"no tokenizer at {args.model}.model", file=sys.stderr)
        return 1

    workers = args.workers or mp.cpu_count()
    args.out.mkdir(parents=True, exist_ok=True)
    val_cut = int(args.val_fraction * VAL_BUCKETS)

    train_w = ShardWriter(args.out, "train", args.tokens_per_shard)
    val_w = ShardWriter(args.out, "val", args.tokens_per_shard)

    seen = 0
    duplicates = 0
    digests: set[int] = set()
    is_val: list[bool] = []

    def unique_stories():
        """Dedup in the parent: the duplicate set has to be global."""
        nonlocal seen, duplicates
        for story in iter_stories(args.input, args.limit):
            seen += 1
            d = story_digest(story)
            if not args.no_dedup:
                if d in digests:
                    duplicates += 1
                    continue
                digests.add(d)
            is_val.append(d % VAL_BUCKETS < val_cut)
            yield story

    print(
        f"encoding {args.input.name} with {workers} workers, "
        f"{args.batch} stories/task, val {args.val_fraction:.3%}"
    )
    started = time.monotonic()
    emitted = 0
    pool_args = {"initializer": _init_worker, "initargs": (str(args.model),)}
    with mp.Pool(workers, **pool_args) as pool:
        tasks = batched(unique_stories(), args.batch)
        for encoded in pool.imap(_encode_batch, tasks, chunksize=1):
            for ids in encoded:
                (val_w if is_val[emitted] else train_w).write(ids)
                emitted += 1
            elapsed = time.monotonic() - started
            total = train_w.tokens + val_w.tokens
            print(
                f"\r  {emitted:,} stories  {total / 1e6:.1f}M tokens  "
                f"{total / elapsed / 1e6:.2f}M tok/s  {elapsed / 60:.1f} min",
                end="",
                flush=True,
            )
    print()
    train_w.close()
    val_w.close()
    elapsed = time.monotonic() - started

    is_full_corpus = args.input.resolve() == TRAIN.path.resolve()
    meta = {
        "vocab_size": 2049,
        "eot_id": 2048,
        "dtype": "uint16",
        "corpus": {
            "file": args.input.name,
            "sha256": TRAIN.sha256 if is_full_corpus else None,
        },
        "stories": {"seen": seen, "unique": emitted, "duplicates": duplicates},
        "tokens": {
            "train": train_w.tokens,
            "val": val_w.tokens,
            "total": train_w.tokens + val_w.tokens,
            "adr_0006_estimate": 510_500_000,
        },
        "shards": {
            "train": train_w.files,
            "val": val_w.files,
            "tokens_per_shard": args.tokens_per_shard,
        },
        "val_fraction": args.val_fraction,
        "dedup": not args.no_dedup,
        "workers": workers,
        "seconds": round(elapsed, 1),
        "date": time.strftime("%Y-%m-%d"),
    }
    meta_text = json.dumps(meta, indent=2) + "\n"
    (args.out / "meta.json").write_text(meta_text, encoding="utf-8")

    total = meta["tokens"]["total"]
    dup_share = duplicates / max(seen, 1)
    print(f"\nstories: {seen:,} seen, {duplicates:,} duplicate ({dup_share:.1%})")
    print(f"tokens : {total:,}  train {train_w.tokens:,}  val {val_w.tokens:,}")
    if args.limit is None and is_full_corpus:
        est = meta["tokens"]["adr_0006_estimate"]
        print(f"         ADR-0006 estimated {est:,} ({total / est - 1:+.1%})")
    print(f"shards : {len(train_w.files)} train, {len(val_w.files)} val in {args.out}")
    print(f"took   : {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
