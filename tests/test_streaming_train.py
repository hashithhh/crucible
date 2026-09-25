"""Training from chunk counts, for corpora that do not fit in memory.

Phase 3 trains the tokenizer on 1.9 GB. `train(text)` holds the whole corpus
and every chunk of it at once, which does not fit. `train_from_counts` takes
the counts instead, so a script can stream the file and merge counters.

That is only sound if slicing the corpus and counting the slices gives the
same counts as counting it whole. These tests pin that, and pin that the two
training paths agree.
"""

from collections import Counter
from itertools import pairwise

from tests.conftest import COMPRESSION_CORPUS, VOCAB_SIZE
from tokenizer import Tokenizer, pretokenize


def test_pretokenize_is_lossless():
    for text in (COMPRESSION_CORPUS[:5000], "a b\n\nc  ", "", "x"):
        assert "".join(pretokenize(text)) == text


TEXT = "one two three\nfour five\n\nsix seven eight\nnine\n" * 40


def count_in_slices(text: str, boundaries: list[int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    edges = [0, *boundaries, len(text)]
    for start, stop in pairwise(edges):
        counts.update(pretokenize(text[start:stop]))
    return counts


def test_slicing_at_line_ends_is_not_additive():
    """The obvious rule is wrong, and quietly so.

    A blank line is one "\\n\\n" chunk. Cut between the two newlines and it
    becomes two "\\n" chunks, so a naive line-chunked stream miscounts
    whitespace. Recorded as a test because the failure is invisible: the
    tokenizer still trains, just on slightly wrong counts.
    """
    ends = [i + 1 for i, ch in enumerate(TEXT) if ch == "\n"]
    assert count_in_slices(TEXT, ends[::7]) != Counter(pretokenize(TEXT))


def test_slicing_after_a_non_whitespace_character_is_additive():
    """The rule the Phase 3 streaming loop uses.

    Cut only where the character before the boundary is not whitespace. That
    cannot split a whitespace run, and cannot split a chunk carrying a leading
    space, so the counts match the whole-corpus counts exactly.
    """
    safe = [
        i
        for i in range(1, len(TEXT))
        if not TEXT[i - 1].isspace() and TEXT[i].isspace()
    ]
    assert count_in_slices(TEXT, safe[::5]) == Counter(pretokenize(TEXT))


def test_train_from_counts_matches_train():
    text = COMPRESSION_CORPUS
    direct = Tokenizer()
    direct.train(text, VOCAB_SIZE)

    streamed = Tokenizer()
    streamed.train_from_counts(Counter(pretokenize(text)), VOCAB_SIZE)

    probe = text[:3000]
    assert streamed.encode(probe) == direct.encode(probe)
    assert streamed.vocab_size == direct.vocab_size


def test_train_from_counts_keeps_the_train_contract():
    counts = Counter(pretokenize(COMPRESSION_CORPUS))
    tok = Tokenizer()
    tok.train_from_counts(counts, 300)
    assert tok.vocab_size == 300

    for bad_call in (
        lambda: Tokenizer().train_from_counts(counts, 255),
        lambda: Tokenizer().train_from_counts(counts, 10_000),  # exhausts
    ):
        try:
            bad_call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_counts_may_be_a_plain_dict():
    """Scripts merge counters; nothing should require a Counter specifically."""
    counts = dict(Counter(pretokenize(COMPRESSION_CORPUS)))
    tok = Tokenizer()
    tok.train_from_counts(counts, 300)
    assert tok.vocab_size == 300
