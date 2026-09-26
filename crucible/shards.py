"""Phase 3.1: read the token shards written by `scripts/encode_corpus.py`.

A split is many files but one token stream. Everything here exists to make
that true for callers: `ShardSet` hides the file boundaries, and a training
window that straddles two shards is stitched rather than skipped.

WHY MEMMAP. The train split is ~1 GB, larger than is comfortable to hold
resident next to a model and its optimiser state. `np.memmap` lets the OS page
in only the windows actually sampled, so opening a split costs no RAM and a
batch costs the few pages it touches.

WHY THE RNG STATE IS EXPOSED. Phase 3.4 requires resume to be exact, not
approximate. A run that resumes with a fresh sampler sees a different data
order and is a different run; `TokenBatches.state` is what makes the sampler
part of the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DTYPE = np.dtype("<u2")  # little-endian uint16, as written by encode_corpus.py
META_NAME = "meta.json"


class ShardSet:
    """One split's shards, addressed as a single contiguous token stream.

    CONTRACT
    - `len(shards)` is the total number of tokens in the split.
    - `read(start, length)` returns exactly `length` ids starting at absolute
      offset `start`, crossing shard boundaries transparently. A range that
      runs past the end raises IndexError rather than returning a short array,
      because a short batch is a silent training bug.
    - The returned array is a copy: it stays valid if the memmap is dropped and
      is safe to hand to `torch.from_numpy`.
    - Opening is cheap and lazy; no shard is read until a window touches it.
    """

    def __init__(self, files: list[Path]) -> None:
        if not files:
            raise ValueError("a split needs at least one shard file")
        self.files = files
        self._maps: list[np.memmap | None] = [None] * len(files)
        lengths = []
        for path in files:
            size = path.stat().st_size
            if size % DTYPE.itemsize:
                raise ValueError(
                    f"{path.name} is {size} bytes, not a whole number of uint16"
                )
            lengths.append(size // DTYPE.itemsize)
        self.lengths = lengths
        # starts[i] is the absolute offset of shard i, and starts[-1] is the
        # total. Keeping the total in the array is what makes the binary
        # search in read() a single clean call.
        self.starts = np.cumsum([0, *lengths])

    @classmethod
    def open(cls, tokens_dir: Path, split: str) -> ShardSet:
        """Open a split using `meta.json` as the manifest.

        The manifest is authoritative rather than a directory glob: a glob
        would silently pick up a stale shard left by an earlier, shorter run
        and train on tokens that are not in the recorded corpus.
        """
        meta = read_meta(tokens_dir)
        names = meta["shards"].get(split)
        if not names:
            raise KeyError(f"{META_NAME} lists no shards for split {split!r}")
        files = [tokens_dir / name for name in names]
        missing = [f.name for f in files if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"{META_NAME} lists missing shards: {missing}")
        shards = cls(files)
        declared = meta["tokens"].get(split)
        if declared is not None and declared != len(shards):
            raise ValueError(
                f"{split}: {META_NAME} declares {declared:,} tokens, "
                f"shards hold {len(shards):,}"
            )
        return shards

    def __len__(self) -> int:
        return int(self.starts[-1])

    def _map(self, index: int) -> np.memmap:
        m = self._maps[index]
        if m is None:
            m = np.memmap(self.files[index], dtype=DTYPE, mode="r")
            self._maps[index] = m
        return m

    def read(self, start: int, length: int) -> np.ndarray:
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if start < 0:
            raise IndexError(f"start must be >= 0, got {start}")
        if start + length > len(self):
            raise IndexError(
                f"read({start:,}, {length:,}) runs past the split's "
                f"{len(self):,} tokens"
            )
        out = np.empty(length, dtype=DTYPE)
        # -1 because searchsorted returns the insertion point to the right of
        # equal values: for start == starts[i] we want shard i, not i + 1.
        index = int(np.searchsorted(self.starts, start, side="right")) - 1
        filled = 0
        while filled < length:
            offset = start + filled - int(self.starts[index])
            take = min(length - filled, self.lengths[index] - offset)
            out[filled : filled + take] = self._map(index)[offset : offset + take]
            filled += take
            index += 1
        return out

    def close(self) -> None:
        """Drop the memmaps. Windows will not delete a file that is mapped."""
        self._maps = [None] * len(self.files)


def read_meta(tokens_dir: Path) -> dict:
    path = tokens_dir / META_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/encode_corpus.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class TokenBatches:
    """Training batches sampled uniformly at random, with a resumable RNG.

    CONTRACT
    - `next()` returns `(x, y)` int64 arrays of shape (batch_size, context),
      where `y` is `x` shifted one step: the next-token target at every
      position.
    - Windows are drawn uniformly over every valid start offset, with
      replacement. Document boundaries are not respected, which is the
      deliberate choice: the separator token is in the stream, so the model
      learns where a document ends rather than being told.
    - Two samplers built with the same seed yield the same sequence of batches.
    - `state` round-trips through `set_state`, so a resumed run continues the
      same data order (Phase 3.4).
    """

    def __init__(
        self,
        shards: ShardSet,
        batch_size: int,
        context: int,
        seed: int,
    ) -> None:
        if batch_size < 1 or context < 1:
            raise ValueError(
                f"batch_size and context must be >= 1, got {batch_size} and {context}"
            )
        # +1 because a window of `context` inputs needs one further token to
        # supply the target for its last position.
        if len(shards) < context + 1:
            raise ValueError(
                f"split holds {len(shards):,} tokens, too few for a "
                f"{context}-token window"
            )
        self.shards = shards
        self.batch_size = batch_size
        self.context = context
        self._high = len(shards) - context  # exclusive: start + context + 1 <= len
        self._rng = np.random.default_rng(seed)

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        starts = self._rng.integers(0, self._high, size=self.batch_size)
        rows = [self.shards.read(int(s), self.context + 1) for s in starts]
        window = np.stack(rows).astype(np.int64)
        return window[:, :-1], window[:, 1:]

    @property
    def state(self) -> dict:
        """The sampler's RNG state, for the checkpoint."""
        return self._rng.bit_generator.state

    def set_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state


def eval_windows(
    shards: ShardSet,
    batch_size: int,
    context: int,
    max_batches: int | None = None,
):
    """Yield deterministic, non-overlapping windows from offset 0.

    Held-out loss is compared across checkpoints, so it must not move for
    reasons that have nothing to do with the model. Random sampling would add
    that jitter; this walks the split in order instead. Windows advance by
    `context`, so every token outside the dropped tail is predicted exactly
    once. A trailing partial batch is dropped so every batch has one shape.
    """
    stride = context + 1
    offset = 0
    made = 0
    while offset + stride <= len(shards):
        rows = []
        while len(rows) < batch_size and offset + stride <= len(shards):
            rows.append(shards.read(offset, stride))
            offset += context
        if len(rows) < batch_size:
            return
        window = np.stack(rows).astype(np.int64)
        yield window[:, :-1], window[:, 1:]
        made += 1
        if max_batches is not None and made >= max_batches:
            return
