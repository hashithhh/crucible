"""decode(encode(s)) == s. The property everything else rests on."""

import pytest

from tests.conftest import ROUNDTRIP_CASES, VOCAB_SIZE


@pytest.mark.parametrize("s", ROUNDTRIP_CASES, ids=lambda s: repr(s[:24]))
def test_roundtrip_ascii_tokenizer(trained, s):
    assert trained.decode(trained.encode(s)) == s


@pytest.mark.parametrize("s", ROUNDTRIP_CASES, ids=lambda s: repr(s[:24]))
def test_roundtrip_unicode_tokenizer(trained_unicode, s):
    assert trained_unicode.decode(trained_unicode.encode(s)) == s


def test_encode_empty_is_empty(trained):
    assert trained.encode("") == []


def test_decode_empty_is_empty(trained):
    assert trained.decode([]) == ""


def test_encode_handles_unseen_characters(trained):
    """Trained on ASCII, asked to encode CJK. Byte-level means this must work."""
    s = "\u5317\u4eac\U0001f30d"
    assert trained.decode(trained.encode(s)) == s


def test_all_ids_in_vocab_range(trained):
    for s in ROUNDTRIP_CASES:
        for i in trained.encode(s):
            assert 0 <= i < VOCAB_SIZE, f"id {i} out of range for {s!r}"


@pytest.mark.parametrize("bad", [VOCAB_SIZE, VOCAB_SIZE + 5, -1, -999])
def test_decode_rejects_out_of_range_id(trained, bad):
    """Valid ids are 0 .. vocab_size-1. Both ends are out of range."""
    with pytest.raises(ValueError):
        trained.decode([bad])


def test_decode_invalid_utf8_policy(trained):
    """Pins the documented policy for byte sequences that are not valid UTF-8.

    Your choice of policy (replace / strict / surrogateescape) is a real
    decision. Record it in an ADR, then make this test assert it.
    Requirement for now: it must NOT raise UnicodeDecodeError.
    """
    lone_continuation = [0x80, 0x81]
    try:
        out = trained.decode(lone_continuation)
    except UnicodeDecodeError:
        pytest.fail("decode must not leak UnicodeDecodeError; pick a policy")
    assert isinstance(out, str)


def test_property_roundtrip():
    """Property-based round-trip over arbitrary text."""
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from tests.conftest import COMPRESSION_CORPUS
    from tokenizer import Tokenizer

    tok = Tokenizer()
    tok.train(COMPRESSION_CORPUS, VOCAB_SIZE)

    @given(st.text())
    @settings(max_examples=200, deadline=None)
    def inner(s):
        assert tok.decode(tok.encode(s)) == s

    inner()
