"""save/load round-trip."""

from tests.conftest import ROUNDTRIP_CASES, VOCAB_SIZE
from tokenizer import Tokenizer


def test_save_load_preserves_encoding(trained, tmp_path):
    prefix = str(tmp_path / "tok")
    trained.save(prefix)

    restored = Tokenizer()
    restored.load(prefix)

    for s in ROUNDTRIP_CASES:
        assert restored.encode(s) == trained.encode(s), f"diverged on {s!r}"


def test_save_load_preserves_decoding(trained, tmp_path):
    prefix = str(tmp_path / "tok")
    trained.save(prefix)
    restored = Tokenizer()
    restored.load(prefix)

    for s in ROUNDTRIP_CASES:
        ids = trained.encode(s)
        assert restored.decode(ids) == trained.decode(ids)


def test_save_is_deterministic(ascii_corpus, tmp_path):
    a = Tokenizer()
    a.train(ascii_corpus, VOCAB_SIZE)
    a.save(str(tmp_path / "one"))
    a.save(str(tmp_path / "two"))

    one = (tmp_path / "one.model").read_bytes()
    two = (tmp_path / "two.model").read_bytes()
    assert one == two


def test_loaded_tokenizer_roundtrips(trained, tmp_path):
    prefix = str(tmp_path / "tok")
    trained.save(prefix)
    restored = Tokenizer()
    restored.load(prefix)
    s = "the cat sat \u5317\u4eac \U0001f30d"
    assert restored.decode(restored.encode(s)) == s
