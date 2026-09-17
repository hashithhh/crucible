"""Training contract: vocab size, argument validation, compression."""

import pytest

from tests.conftest import VOCAB_SIZE
from tokenizer import Tokenizer


def test_vocab_size_below_256_raises(ascii_corpus):
    t = Tokenizer()
    with pytest.raises(ValueError):
        t.train(ascii_corpus, 255)


def test_vocab_size_exactly_256_is_legal(ascii_corpus):
    """256 means no merges at all — a valid, degenerate tokenizer."""
    t = Tokenizer()
    t.train(ascii_corpus, 256)
    s = "the cat sat"
    assert t.decode(t.encode(s)) == s
    assert t.encode(s) == list(s.encode("utf-8"))


def test_vocab_size_is_respected_exactly(ascii_corpus):
    """Every id produced must fall inside the requested vocabulary."""
    for size in (256, 280, 320, 400):
        t = Tokenizer()
        t.train(ascii_corpus, size)
        ids = t.encode(ascii_corpus[:2000])
        assert max(ids) < size
        assert min(ids) >= 0


def test_compression_improves_with_vocab_size(compression_corpus):
    """bytes/token must strictly improve as the vocabulary grows.

    Sizes are chosen to stay well inside the corpus's supply of useful merges
    (336 distinct bigrams vs 104 merges at the largest size). If this fails,
    suspect your merge loop, not the corpus.
    """
    sample = compression_corpus[:6000]
    n_bytes = len(sample.encode("utf-8"))

    ratios = []
    for size in (256, 280, 312, 360):
        t = Tokenizer()
        t.train(compression_corpus, size)
        ratios.append(n_bytes / len(t.encode(sample)))

    for smaller, larger in zip(ratios, ratios[1:]):
        assert larger > smaller, f"compression did not improve: {ratios}"


def test_vocab_size_is_exact(compression_corpus):
    """The promise is `exactly vocab_size`, not `at most`.

    test_vocab_size_is_respected_exactly only bounds ids from above, so a
    tokenizer that quietly stops merging early would pass it. This one does
    not let that slide.
    """
    for size in (256, 300, 360):
        t = Tokenizer()
        t.train(compression_corpus, size)
        assert t.vocab_size == size


def test_merge_exhaustion_behaviour(ascii_corpus):
    """What does train() owe when the corpus runs out of pairs to merge?

    ASCII_CORPUS has 24 distinct bigrams and collapses to one token after
    ~90 merges. Asking for 600 is asking for merges that do not exist.

    Raise? Stop early and expose a smaller vocab_size? Pad with unused
    tokens? All three are defensible. Decide, write ADR-0003, then replace
    this skip with the assertion.
    """
    import pytest
    pytest.skip("pin this once ADR-0003 (merge exhaustion) is written")


def test_verbose_does_not_change_result(ascii_corpus):
    a, b = Tokenizer(), Tokenizer()
    a.train(ascii_corpus, VOCAB_SIZE, verbose=False)
    b.train(ascii_corpus, VOCAB_SIZE, verbose=True)
    assert a.encode(ascii_corpus[:500]) == b.encode(ascii_corpus[:500])
