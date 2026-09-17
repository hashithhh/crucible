"""Special tokens: registration, encode/decode, and persistence.

Public interface only: register_special_tokens, encode(allowed_special=...),
decode, vocab_size, save, load.
"""

import pytest

from tests.conftest import COMPRESSION_CORPUS, VOCAB_SIZE
from tokenizer import Tokenizer

EOT = "<|endoftext|>"
PAD = "<|pad|>"
EOT_ID = VOCAB_SIZE
PAD_ID = VOCAB_SIZE + 1


def _trained():
    t = Tokenizer()
    t.train(COMPRESSION_CORPUS, VOCAB_SIZE)
    return t


@pytest.fixture
def plain():
    return _trained()


@pytest.fixture
def special():
    t = _trained()
    t.register_special_tokens({EOT: EOT_ID, PAD: PAD_ID})
    return t


# --- registration ------------------------------------------------------------


def test_special_tokens_extend_vocab_size(special):
    assert special.vocab_size == VOCAB_SIZE + 2


def test_registration_may_assign_ids_in_any_order(plain):
    plain.register_special_tokens({PAD: VOCAB_SIZE + 1, EOT: VOCAB_SIZE})
    ids = plain.encode(EOT + PAD, allowed_special="all")
    assert ids == [VOCAB_SIZE, VOCAB_SIZE + 1]


def test_registration_can_happen_in_several_calls(plain):
    plain.register_special_tokens({EOT: VOCAB_SIZE})
    plain.register_special_tokens({PAD: VOCAB_SIZE + 1})
    assert plain.vocab_size == VOCAB_SIZE + 2
    assert plain.encode(PAD, allowed_special={PAD}) == [VOCAB_SIZE + 1]


@pytest.mark.parametrize(
    "tokens",
    [
        {EOT: 100},
        {EOT: 0},
        {EOT: VOCAB_SIZE + 1},
        {EOT: VOCAB_SIZE, PAD: VOCAB_SIZE + 2},
        {EOT: VOCAB_SIZE, PAD: VOCAB_SIZE},
        {"": VOCAB_SIZE},
        {EOT: -1},
        {EOT: True},
    ],
    ids=[
        "collides-with-learned-id",
        "collides-with-byte-id",
        "leaves-a-gap",
        "leaves-a-gap-between",
        "two-texts-one-id",
        "empty-text",
        "negative-id",
        "bool-id",
    ],
)
def test_registration_rejects_bad_ids_and_text(plain, tokens):
    with pytest.raises(ValueError):
        plain.register_special_tokens(tokens)
    assert plain.vocab_size == VOCAB_SIZE, "a rejected registration must not stick"


def test_registration_rejects_duplicate_text(special):
    with pytest.raises(ValueError):
        special.register_special_tokens({EOT: VOCAB_SIZE + 2})


def test_train_after_registration_raises():
    t = Tokenizer()
    t.register_special_tokens({EOT: 256})
    with pytest.raises(ValueError):
        t.train(COMPRESSION_CORPUS, VOCAB_SIZE)


# --- encode / decode -----------------------------------------------------------


def test_allowed_special_encodes_to_its_id(special):
    assert special.encode(EOT, allowed_special="all") == [EOT_ID]
    assert special.encode(PAD, allowed_special={PAD}) == [PAD_ID]


def test_special_id_decodes_to_its_literal_text(special):
    assert special.decode([EOT_ID]) == EOT
    assert special.decode([PAD_ID, EOT_ID]) == PAD + EOT


def test_special_tokens_roundtrip_inside_text(special):
    s = f"hello{EOT}world {PAD}{PAD}\n{EOT}"
    ids = special.encode(s, allowed_special="all")
    assert special.decode(ids) == s
    assert ids.count(EOT_ID) == 2
    assert ids.count(PAD_ID) == 2


def test_text_around_a_special_encodes_independently(special):
    """A special token is a hard boundary: text on each side encodes alone."""
    s = f"hello{EOT}world"
    expected = special.encode("hello") + [EOT_ID] + special.encode("world")
    assert special.encode(s, allowed_special="all") == expected


def test_literal_special_text_is_ordinary_by_default(special, plain):
    """Literal special-token text in input that did not allow it is encoded as
    ordinary text — never silently as the special id."""
    s = f"a {EOT} b {PAD}"
    ids = special.encode(s)
    assert EOT_ID not in ids and PAD_ID not in ids
    assert special.decode(ids) == s
    # ...and exactly as a tokenizer with no special tokens would encode it.
    assert ids == plain.encode(s)


def test_only_named_specials_are_allowed(special):
    ids = special.encode(EOT + PAD, allowed_special={PAD})
    assert PAD_ID in ids
    assert EOT_ID not in ids
    assert special.decode(ids) == EOT + PAD


def test_allowing_an_unregistered_special_raises(special):
    with pytest.raises(ValueError):
        special.encode("x", allowed_special={"<|nope|>"})


def test_allowed_special_as_a_bare_string_raises(special):
    """{"<|endoftext|>"} is a set of one; "<|endoftext|>" is 13 characters."""
    with pytest.raises(ValueError):
        special.encode(EOT, allowed_special=EOT)


def test_longest_special_wins_on_overlap(plain):
    plain.register_special_tokens({"<|a|>": VOCAB_SIZE, "<|a|>b": VOCAB_SIZE + 1})
    assert plain.encode("<|a|>b", allowed_special="all") == [VOCAB_SIZE + 1]
    expected = [VOCAB_SIZE] + plain.encode("c")
    assert plain.encode("<|a|>c", allowed_special="all") == expected


def test_decode_rejects_id_past_the_specials(special):
    with pytest.raises(ValueError):
        special.decode([VOCAB_SIZE + 2])


def test_every_id_in_range_decodes(special):
    for i in range(special.vocab_size):
        special.decode([i])


def test_property_roundtrip_with_specials(special):
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    pieces = st.lists(st.text() | st.sampled_from([EOT, PAD]), max_size=8)

    @given(pieces)
    @settings(max_examples=200, deadline=None)
    def inner(parts):
        s = "".join(parts)
        assert special.decode(special.encode(s, allowed_special="all")) == s
        assert special.decode(special.encode(s)) == s

    inner()


# --- persistence ---------------------------------------------------------------


def test_special_tokens_survive_save_load(special, tmp_path):
    prefix = str(tmp_path / "tok")
    special.save(prefix)
    restored = Tokenizer()
    restored.load(prefix)

    assert restored.vocab_size == special.vocab_size
    s = f"once{EOT} upon {PAD}a time {EOT}"
    for allowed in ("all", (), {PAD}):
        ids = special.encode(s, allowed_special=allowed)
        assert restored.encode(s, allowed_special=allowed) == ids
        assert restored.decode(ids) == s


def test_awkward_special_text_survives_save_load(plain, tmp_path):
    """Spaces, quotes, backslashes, newlines and non-ASCII in the token text."""
    weird = '<| caf\u00e9 "quoted" \\ back\nslash \U0001f30d |>'
    plain.register_special_tokens({weird: VOCAB_SIZE})
    prefix = str(tmp_path / "tok")
    plain.save(prefix)
    restored = Tokenizer()
    restored.load(prefix)

    assert restored.encode(weird, allowed_special="all") == [VOCAB_SIZE]
    assert restored.decode([VOCAB_SIZE]) == weird


def test_save_with_specials_is_deterministic(special, tmp_path):
    special.save(str(tmp_path / "one"))
    special.save(str(tmp_path / "two"))
    one = (tmp_path / "one.model").read_bytes()
    assert one == (tmp_path / "two.model").read_bytes()


def test_load_without_specials_clears_previous_ones(special, tmp_path):
    """load() replaces state; specials from before the load must not linger."""
    prefix = str(tmp_path / "tok")
    _trained().save(prefix)
    special.load(prefix)
    assert special.vocab_size == VOCAB_SIZE
    with pytest.raises(ValueError):
        special.encode("x", allowed_special={EOT})
