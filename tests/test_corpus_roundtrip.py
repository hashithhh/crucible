"""Corpus-wide round-trip: decode(encode(story)) == story for every story.

Slow (roughly a minute), so it only runs with `pytest --runslow`, and it
skips if data/ has not been fetched (`python scripts/get_data.py`).

Setup mirrors ADR-0001: vocab 2,048, trained on the first 5 MB of stories
with the `<|endoftext|>` separator removed. The check then covers ALL of
TinyStories-valid, not a sample: every story, including the leading fragment
the file starts with, and including the stories the tokenizer was trained on.
"""

import pytest

from crucible.data import VALID, load_corpus
from tokenizer import Tokenizer

pytestmark = pytest.mark.slow

ADR_0001_VOCAB = 2048  # docs/adr/0001-vocab-size.md
TRAIN_BYTES = 5_000_000  # same training budget as the ADR-0001 sweep
SEPARATOR = "\n<|endoftext|>\n"
MAX_REPORTED = 10  # failures listed in the assertion message; all are counted


def _stories() -> list[str]:
    text = load_corpus()
    stories = text.split(SEPARATOR)
    # Lossless split: rejoining must rebuild the file exactly, or some text
    # would escape the check.
    assert SEPARATOR.join(stories) == text
    return stories


def _training_text(stories: list[str]) -> str:
    """Stories from the start of the file, up to TRAIN_BYTES, newline-joined.

    Skips the leading fragment, as the ADR-0001 sweep does.
    """
    taken, used = [], 0
    for s in stories[1:]:
        cost = len(s.encode("utf-8")) + (1 if taken else 0)
        if used + cost > TRAIN_BYTES:
            break
        taken.append(s)
        used += cost
    return "\n".join(taken)


def _first_difference(expected: str, got: str) -> str:
    for k, (a, b) in enumerate(zip(expected, got, strict=False)):
        if a != b:
            context = expected[max(0, k - 20) : k + 20]
            return f"char {k}: expected {a!r}, got {b!r}; context {context!r}"
    k = min(len(expected), len(got))
    return (
        f"char {k}: lengths differ (expected {len(expected)}, got {len(got)}); "
        f"extra {(expected[k:] or got[k:])[:40]!r}"
    )


@pytest.mark.skipif(not VALID.path.is_file(), reason="run scripts/get_data.py")
def test_every_story_roundtrips():
    stories = _stories()
    tok = Tokenizer()
    tok.train(_training_text(stories), ADR_0001_VOCAB)
    assert tok.vocab_size == ADR_0001_VOCAB

    failures = []
    for i, story in enumerate(stories):
        decoded = tok.decode(tok.encode(story))
        if decoded != story:
            failures.append(f"story {i}: {_first_difference(story, decoded)}")

    assert not failures, (
        f"{len(failures)} of {len(stories)} stories failed to round-trip; "
        f"first {min(len(failures), MAX_REPORTED)}:\n"
        + "\n".join(failures[:MAX_REPORTED])
    )
