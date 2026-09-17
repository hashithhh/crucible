"""Determinism, including under a deliberate frequency tie.

These assert behaviour through the public interface only — no assumptions
about how merges are stored.
"""

from tests.conftest import VOCAB_SIZE
from tokenizer import Tokenizer


def _encode_fingerprint(tok, probes):
    return [tok.encode(p) for p in probes]


PROBES = ["the cat sat on the mat", "a rat ate a hat", "ABCDABCD"]


def test_training_is_deterministic(compression_corpus):
    a, b = Tokenizer(), Tokenizer()
    a.train(compression_corpus, VOCAB_SIZE)
    b.train(compression_corpus, VOCAB_SIZE)
    assert _encode_fingerprint(a, PROBES) == _encode_fingerprint(b, PROBES)


def test_deterministic_under_frequency_tie(tie_corpus):
    """TIE_CORPUS contains two pairs with identical counts.

    Whatever rule breaks the tie, it must break it the SAME WAY every time.
    An implementation that depends on dict or set iteration order will
    eventually fail this.
    """
    runs = []
    for _ in range(5):
        t = Tokenizer()
        t.train(tie_corpus, 258)  # room for exactly two merges
        runs.append(_encode_fingerprint(t, ["ABCD", "ABAB", "CDCD"]))
    assert all(r == runs[0] for r in runs), f"tie-break is unstable: {runs}"


def test_tie_break_rule_is_pinned(tie_corpus):
    """Characterisation test for ADR-0002.

    The rule: highest count, then the lexicographically SMALLEST pair.
    TIE_CORPUS gives (65,66)="AB" and (67,68)="CD" 50 occurrences each, so
    "AB" must win id 256 and "CD" must take 257. If the tie-break ever
    changes, this test is what notices.
    """
    t = Tokenizer()
    t.train(tie_corpus, 258)
    assert t.encode("ABCD") == [256, 257]
    assert t.encode("ABAB") == [256, 256]
    assert t.encode("CDCD") == [257, 257]


def test_encoding_is_stable_across_calls(trained):
    s = "the cat sat on the mat"
    assert trained.encode(s) == trained.encode(s)
