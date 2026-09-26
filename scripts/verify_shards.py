"""Phase 3.1: verify the token shards actually contain the corpus.

    python scripts/verify_shards.py              # default: 2,000 stories
    python scripts/verify_shards.py --limit 200  # quicker

`encode_corpus.py` reported what it believed it wrote. This reads the shards
back independently and checks that belief against the corpus on disk, because
a pipeline that silently drops or reorders tokens still prints a plausible
summary and still trains -- just on something other than the data claimed.

FOUR CHECKS, weakest to strongest:

1. Every id is a legal token. Catches endianness and dtype mistakes.
2. The separator appears exactly once per unique story, across both splits.
   A global count, so it catches a story lost anywhere in the corpus rather
   than only near the start.
3. The head of each split re-encodes to exactly the same ids. This is the
   real check: it re-derives dedup and the train/val assignment from the
   corpus and compares ids position by position, so an off-by-one shard
   boundary or a misrouted story fails here.
4. Those same ids decode back to the original story text.

Writes results/shard_verification.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crucible.data import TRAIN  # noqa: E402
from crucible.shards import ShardSet, read_meta  # noqa: E402
from scripts.encode_corpus import VAL_BUCKETS, iter_stories, story_digest  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

DEFAULT_TOKENS = ROOT / "data" / "tokens"
DEFAULT_MODEL = ROOT / "models" / "tinystories-2048"
SCAN_CHUNK = 1 << 24  # 16M tokens (32 MB) per pass when scanning a whole split
S25_PARAMS = 26_223_616  # ADR-0006, S25; here only to report the ratio


def count_id(shards: ShardSet, wanted: int) -> int:
    """Count occurrences of one id across a split, without loading it all."""
    seen = 0
    for offset in range(0, len(shards), SCAN_CHUNK):
        block = shards.read(offset, min(SCAN_CHUNK, len(shards) - offset))
        seen += int(np.count_nonzero(block == wanted))
    return seen


def max_id(shards: ShardSet) -> int:
    highest = 0
    for offset in range(0, len(shards), SCAN_CHUNK):
        block = shards.read(offset, min(SCAN_CHUNK, len(shards) - offset))
        highest = max(highest, int(block.max()))
    return highest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--input", type=Path, default=TRAIN.path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=2000, help="stories to re-encode")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)

    meta = read_meta(args.tokens)
    eot = meta["eot_id"]
    vocab = meta["vocab_size"]
    splits = {name: ShardSet.open(args.tokens, name) for name in ("train", "val")}
    failures: list[str] = []
    started = time.monotonic()

    print(f"tokens : {sum(len(s) for s in splits.values()):,}")
    for name, shards in splits.items():
        print(f"  {name:<5} {len(shards):>13,} in {len(shards.files)} shard(s)")

    print("\n1. id range")
    for name, shards in splits.items():
        highest = max_id(shards)
        ok = highest < vocab
        print(f"   {name:<5} max id {highest} < {vocab}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{name}: id {highest} is outside the vocabulary")

    print("\n2. separator count")
    counted = {name: count_id(s, eot) for name, s in splits.items()}
    total_eot = sum(counted.values())
    expected = meta["stories"]["unique"]
    ok = total_eot == expected
    print(f"   train {counted['train']:,}  val {counted['val']:,}")
    print(
        f"   total {total_eot:,} vs {expected:,} unique stories  "
        f"{'ok' if ok else 'FAIL'}"
    )
    if not ok:
        failures.append(f"{total_eot:,} separators for {expected:,} stories")

    print(f"\n3. re-encoding the first {args.limit:,} unique stories")
    tok = Tokenizer()
    tok.load(str(args.model))
    val_cut = int(meta["val_fraction"] * VAL_BUCKETS)

    rebuilt: dict[str, list[int]] = {"train": [], "val": []}
    originals: dict[str, list[str]] = {"train": [], "val": []}
    digests: set[int] = set()
    kept = 0
    for story in iter_stories(args.input, None):
        d = story_digest(story)
        if d in digests:
            continue
        digests.add(d)
        split = "val" if d % VAL_BUCKETS < val_cut else "train"
        rebuilt[split].extend(tok.encode(story) + [eot])
        originals[split].append(story)
        kept += 1
        if kept >= args.limit:
            break

    for name, ids in rebuilt.items():
        if not ids:
            print(f"   {name:<5} no stories in this sample, skipped")
            continue
        wanted = np.asarray(ids, dtype=splits[name].read(0, 1).dtype)
        on_disk = splits[name].read(0, len(ids))
        ok = np.array_equal(on_disk, wanted)
        print(
            f"   {name:<5} {len(ids):>9,} ids  {len(originals[name]):>6,} stories  "
            f"{'ok' if ok else 'FAIL'}"
        )
        if not ok:
            first = int(np.argmax(on_disk != wanted))
            failures.append(
                f"{name}: ids differ from position {first}: "
                f"shard {on_disk[first]} vs corpus {wanted[first]}"
            )

    print("\n4. decoding those ids back to text")
    for name, ids in rebuilt.items():
        if not ids:
            continue
        on_disk = splits[name].read(0, len(ids)).tolist()
        pieces: list[list[int]] = []
        current: list[int] = []
        for i in on_disk:
            if i == eot:
                pieces.append(current)
                current = []
            else:
                current.append(i)
        paired = enumerate(zip(pieces, originals[name], strict=True))
        bad = [k for k, (piece, text) in paired if tok.decode(piece) != text]
        print(
            f"   {name:<5} {len(pieces):>6,} stories  "
            f"{len(bad)} mismatched  {'ok' if not bad else 'FAIL'}"
        )
        if bad:
            failures.append(f"{name}: {len(bad)} stories did not decode back")

    total_tokens = sum(len(s) for s in splits.values())
    estimate = meta["tokens"]["adr_0006_estimate"]
    ratio = total_tokens / S25_PARAMS
    print("\nagainst ADR-0006")
    print(
        f"   tokens {total_tokens:,} vs {estimate:,} "
        f"({total_tokens / estimate - 1:+.1%})"
    )
    print(f"   {ratio:.2f} tokens per parameter for S25's {S25_PARAMS:,}")

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "corpus": meta["corpus"],
        "checked_stories": kept,
        "tokens": {name: len(s) for name, s in splits.items()},
        "total_tokens": total_tokens,
        "separators": counted,
        "unique_stories": expected,
        "adr_0006_estimate": estimate,
        "delta_vs_estimate": round(total_tokens / estimate - 1, 4),
        "tokens_per_param": round(ratio, 2),
        "s25_params": S25_PARAMS,
        "failures": failures,
        "seconds": round(time.monotonic() - started, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "shard_verification.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {path}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
