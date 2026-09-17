"""Pre-tokenization (ADR-0005): merges are learned and applied within chunks.

Public interface only. Chunk boundaries are not exposed, so each test builds a
corpus whose chunking is unambiguous under the cl100k pattern and checks what
the tokenizer can and cannot learn from it. A tokenizer that merged across the
raw byte stream would learn more merges, or different ones, and fail.
"""

import unicodedata

import pytest

from tests.conftest import COMPRESSION_CORPUS, ROUNDTRIP_CASES
from tokenizer import Tokenizer


def _tokens(tok, text):
    return [tok.decode([i]) for i in tok.encode(text)]


def _max_merges(text):
    """Merges the corpus supports before ADR-0003 exhaustion."""
    try:
        Tokenizer().train(text, 10_000)
    except ValueError as exc:
        assert "exhausted" in str(exc)
        return int(str(exc).split("after ")[1].split()[0])
    raise AssertionError("corpus did not exhaust")


# --- merges never span a word boundary --------------------------------------


def test_merges_never_span_a_word_boundary():
    """ "ab ab ab ..." chunks as "ab", " ab", " ab", ... .

    Inside chunks the only merges are "ab" and " ab". Across the raw stream,
    "b " occurs 1000 times and would be merged, and far longer tokens would
    follow. So: exactly two merges exist, and no token ever contains "b ".
    """
    corpus = "ab " * 1000
    assert _max_merges(corpus) == 2

    t = Tokenizer()
    t.train(corpus, 258)
    assert _tokens(t, "ab ab") == ["ab", " ab"]
    for i in range(t.vocab_size):
        assert "b " not in t.decode([i])


def test_no_learned_token_crosses_a_character_class_boundary():
    """Every learned token is a legal chunk shape.

    Trained to near the corpus ceiling, so hundreds of merges were available
    to go wrong. Under cl100k a letter run may carry one leading non-letter,
    non-digit, non-newline character; digit runs are pure and at most 3 long.
    """
    t = Tokenizer()
    t.train(COMPRESSION_CORPUS, 600)

    def cat(ch):
        return unicodedata.category(ch)[0]

    for i in range(256, t.vocab_size):
        s = t.decode([i])
        if "\ufffd" in s:
            continue  # partial UTF-8; COMPRESSION_CORPUS is ASCII, so rare
        letters = [k for k, ch in enumerate(s) if cat(ch) == "L"]
        digits = [ch for ch in s if cat(ch) == "N"]
        if letters:
            first = letters[0]
            assert first <= 1, f"token {s!r}: >1 char before its letters"
            assert all(cat(ch) == "L" for ch in s[first:]), f"token {s!r}"
            assert not digits, f"token {s!r} mixes letters and digits"
            if first == 1:
                assert s[0] not in "\r\n", f"token {s!r}"
        elif digits:
            assert len(s) == len(digits) <= 3, f"token {s!r}"


# --- encode applies the same split --------------------------------------------


def test_encode_applies_the_same_split():
    """ ".x" is a chunk and learns a merge; "!.x" chunks as "!.", "x".

    An encoder that applied merges to the raw stream would produce
    [ord("!"), <".x">]. Pre-tokenized, "." and "x" are in different chunks
    and the merge cannot apply.
    """
    t = Tokenizer()
    t.train(".x" * 500, 257)
    assert t.encode(".x") == [256]
    assert t.encode("!.x") == [ord("!"), ord("."), ord("x")]


@pytest.mark.parametrize("s", ROUNDTRIP_CASES, ids=lambda s: repr(s[:24]))
def test_encoding_splits_cleanly_before_a_spaced_word(trained, s):
    """ "foo bar" chunks as "foo", " bar", so the pieces encode independently.

    For every single space between a non-space and a letter,
    encode(s) == encode(left) + encode(right).
    """
    for k, ch in enumerate(s):
        if (
            ch == " "
            and k > 0
            and not s[k - 1].isspace()
            and s[k + 1 : k + 2].isalpha()
        ):
            assert trained.encode(s) == trained.encode(s[:k]) + trained.encode(s[k:])


# --- the pattern's specific rules ---------------------------------------------


def test_digits_group_in_threes():
    """`\\p{N}{1,3}`: "1234567" can only ever become "123", "456", "7"."""
    corpus = "1234567 " * 300
    assert _max_merges(corpus) == 4

    t = Tokenizer()
    t.train(corpus, 260)
    assert _tokens(t, "1234567") == ["123", "456", "7"]


def test_contractions_split_from_their_word():
    """`'(?i:[sdmt]|ll|ve|re)`: "don't" is "don" + "'t", never "don'" + "t"."""
    corpus = "don't " * 300
    assert _max_merges(corpus) == 4

    t = Tokenizer()
    t.train(corpus, 260)
    assert _tokens(t, "don't") == ["don", "'t"]


def test_non_decimal_numbers_are_numbers_not_letters():
    """\u00bd is \\p{N} (No), not \\p{L}.

    So "x\u00bd" chunks as "x", "\u00bd" and only the two UTF-8 bytes of
    "\u00bd" can merge. An approximation like `[^\\W\\d_]` for letters would
    class "\u00bd" as a letter and allow merges across "x\u00bd".
    """
    assert _max_merges("x\u00bd" * 500) == 1


def test_information_separators_are_not_whitespace():
    """U+001F is not Unicode White_Space, but Python's `\\s` matches it.

    cl100k (Rust regex) keeps "\\x1f!" together as punctuation, so the pair
    is learnable. A pattern using stdlib `\\s` would split them and learn
    nothing.
    """
    t = Tokenizer()
    t.train("\x1f!" * 500, 257)
    assert t.encode("\x1f!") == [256]


# --- persistence of the pattern choice ----------------------------------------


def test_load_rejects_a_model_with_a_different_pattern(trained, tmp_path):
    """A model trained under another split would encode differently. Refuse."""
    prefix = tmp_path / "tok"
    trained.save(str(prefix))
    model = prefix.with_suffix(".model")
    lines = model.read_text(encoding="utf-8").split("\n")
    assert lines[1].startswith("pattern ")
    lines[1] = "pattern gpt2"
    model.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(ValueError, match="pattern"):
        Tokenizer().load(str(prefix))


def test_property_roundtrip_after_pretokenized_training():
    """decode(encode(s)) == s over arbitrary text, trained near the ceiling."""
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    tok = Tokenizer()
    tok.train(COMPRESSION_CORPUS, 600)

    @given(st.text())
    @settings(max_examples=300, deadline=None)
    def inner(s):
        assert tok.decode(tok.encode(s)) == s

    inner()
