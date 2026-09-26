"""Phase 3.1: the shard reader's public behaviour.

Shards here are tiny and synthetic. The point of nearly every case is a
boundary: the reader's whole job is to make many files look like one stream,
and the only place that can go wrong is where one file ends.

Ids are consecutive integers so that a wrong offset produces a visibly wrong
value rather than plausible-looking tokens.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from crucible.shards import DTYPE, ShardSet, TokenBatches, eval_windows

SHARD = 10  # tokens per synthetic shard; small enough to cross by hand


def build(tmp_path, split="train", n_shards=3, shard_tokens=SHARD, meta=True):
    """Write `n_shards` shards holding 0, 1, 2, ... and return (dir, stream)."""
    stream = np.arange(n_shards * shard_tokens, dtype=DTYPE)
    names = []
    for i in range(n_shards):
        name = f"{split}_{i:05d}.bin"
        chunk = stream[i * shard_tokens : (i + 1) * shard_tokens]
        (tmp_path / name).write_bytes(chunk.tobytes())
        names.append(name)
    if meta:
        (tmp_path / "meta.json").write_text(
            json.dumps(
                {
                    "vocab_size": 2049,
                    "eot_id": 2048,
                    "dtype": "uint16",
                    "tokens": {split: int(stream.size)},
                    "shards": {split: names, "tokens_per_shard": shard_tokens},
                    "date": "2026-09-26",
                }
            ),
            encoding="utf-8",
        )
    return tmp_path, stream


def test_length_is_the_whole_stream(tmp_path):
    d, stream = build(tmp_path)
    assert len(ShardSet.open(d, "train")) == stream.size


@pytest.mark.parametrize(
    "start,length",
    [
        (0, 1),  # first token
        (0, 30),  # the entire stream in one read
        (9, 2),  # straddles the first boundary
        (10, 5),  # begins exactly on a boundary
        (5, 20),  # spans two boundaries
        (19, 11),  # straddles the second boundary and runs to the end
        (29, 1),  # last token
        (7, 0),  # empty read
    ],
)
def test_read_matches_the_concatenated_stream(tmp_path, start, length):
    d, stream = build(tmp_path)
    got = ShardSet.open(d, "train").read(start, length)
    assert np.array_equal(got, stream[start : start + length])


def test_read_past_the_end_raises_rather_than_returning_short(tmp_path):
    d, stream = build(tmp_path)
    shards = ShardSet.open(d, "train")
    with pytest.raises(IndexError):
        shards.read(stream.size - 2, 3)
    with pytest.raises(IndexError):
        shards.read(stream.size, 1)
    with pytest.raises(IndexError):
        shards.read(-1, 1)


def test_reads_are_copies_not_views(tmp_path):
    """A caller may keep a batch after the memmaps are dropped."""
    d, _ = build(tmp_path)
    shards = ShardSet.open(d, "train")
    got = shards.read(8, 4)
    shards.close()
    assert np.array_equal(got, [8, 9, 10, 11])


def test_open_refuses_a_manifest_that_disagrees_with_the_files(tmp_path):
    """A stale token count means the recorded corpus is not the one on disk."""
    d, _ = build(tmp_path)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    meta["tokens"]["train"] += 1
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="declares"):
        ShardSet.open(d, "train")


def test_open_refuses_a_missing_shard(tmp_path):
    d, _ = build(tmp_path)
    (d / "train_00001.bin").unlink()
    with pytest.raises(FileNotFoundError):
        ShardSet.open(d, "train")


def test_open_without_a_manifest_names_the_script_that_writes_one(tmp_path):
    d, _ = build(tmp_path, meta=False)
    with pytest.raises(FileNotFoundError, match="encode_corpus"):
        ShardSet.open(d, "train")


def test_open_rejects_an_unknown_split(tmp_path):
    d, _ = build(tmp_path)
    with pytest.raises(KeyError):
        ShardSet.open(d, "val")


def test_odd_sized_shard_is_not_a_uint16_stream(tmp_path):
    d, _ = build(tmp_path, n_shards=1)
    (d / "train_00000.bin").write_bytes(b"\x01\x02\x03")
    with pytest.raises(ValueError, match="uint16"):
        ShardSet(sorted(d.glob("train_*.bin")))


def test_batch_targets_are_inputs_shifted_one_step(tmp_path):
    d, _ = build(tmp_path)
    x, y = TokenBatches(ShardSet.open(d, "train"), 4, 3, seed=0).next()
    assert x.shape == y.shape == (4, 3)
    assert x.dtype == y.dtype == np.int64
    for xi, yi in zip(x, y, strict=True):
        # Ids are consecutive, so the shift is checkable arithmetically.
        assert np.array_equal(yi, xi + 1)


def test_same_seed_gives_the_same_batches(tmp_path):
    d, _ = build(tmp_path)
    a = TokenBatches(ShardSet.open(d, "train"), 2, 4, seed=7).next()
    b = TokenBatches(ShardSet.open(d, "train"), 2, 4, seed=7).next()
    assert np.array_equal(a[0], b[0])


def test_different_seeds_give_different_batches(tmp_path):
    d, _ = build(tmp_path, n_shards=20)
    a = TokenBatches(ShardSet.open(d, "train"), 8, 4, seed=1).next()
    b = TokenBatches(ShardSet.open(d, "train"), 8, 4, seed=2).next()
    assert not np.array_equal(a[0], b[0])


def test_sampler_state_round_trips_so_a_resume_continues_the_same_order(tmp_path):
    """Phase 3.4: resume must be exact, and that includes the data order."""
    d, _ = build(tmp_path, n_shards=20)
    original = TokenBatches(ShardSet.open(d, "train"), 4, 4, seed=3)
    original.next()
    saved = original.state
    expected = [original.next()[0] for _ in range(3)]

    resumed = TokenBatches(ShardSet.open(d, "train"), 4, 4, seed=999)
    resumed.set_state(saved)
    got = [resumed.next()[0] for _ in range(3)]

    assert all(np.array_equal(e, g) for e, g in zip(expected, got, strict=True))


def test_sampler_needs_a_window_to_fit(tmp_path):
    d, _ = build(tmp_path, n_shards=1, shard_tokens=4)
    with pytest.raises(ValueError, match="too few"):
        TokenBatches(ShardSet.open(d, "train"), 1, 4, seed=0)


def test_eval_windows_are_in_order_and_do_not_overlap(tmp_path):
    d, stream = build(tmp_path)
    shards = ShardSet.open(d, "train")
    batches = list(eval_windows(shards, batch_size=2, context=4))
    inputs = np.concatenate([x.reshape(-1) for x, _ in batches])
    # Consecutive from 0: each window continues exactly where the last ended.
    assert np.array_equal(inputs, stream[: inputs.size].astype(np.int64))


def test_eval_windows_are_deterministic(tmp_path):
    d, _ = build(tmp_path)
    shards = ShardSet.open(d, "train")
    first = [x for x, _ in eval_windows(shards, 2, 4)]
    second = [x for x, _ in eval_windows(shards, 2, 4)]
    assert len(first) == len(second)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_eval_windows_drop_the_short_tail_rather_than_reshaping(tmp_path):
    d, _ = build(tmp_path)
    shards = ShardSet.open(d, "train")
    for x, y in eval_windows(shards, batch_size=2, context=4):
        assert x.shape == y.shape == (2, 4)


def test_eval_windows_respects_max_batches(tmp_path):
    d, _ = build(tmp_path, n_shards=40)
    shards = ShardSet.open(d, "train")
    assert len(list(eval_windows(shards, 2, 4, max_batches=3))) == 3
