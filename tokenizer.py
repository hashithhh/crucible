"""Byte-level BPE tokenizer — Crucible Phase 1.

Implementation is written by hand. See CLAUDE.md.
Standard library only.
"""


class Tokenizer:
    """A byte-level BPE tokenizer.

    Contract notes that apply to the whole class:

    - Token ids 0..255 are reserved for raw bytes. Any learned token takes
      an id >= 256.
    - The tokenizer operates on the UTF-8 encoding of text, not on Unicode
      code points, so any `str` is representable.
    """

    def __init__(self) -> None:
        # Deliberately holds no state yet.
        #
        # What a tokenizer must remember (the merges, the vocabulary, and how
        # they are keyed) is the central design decision of this phase, and it
        # is yours. Add attributes here as a test forces you to, not before.
        pass

    @property
    def vocab_size(self) -> int:
        """Number of entries in the vocabulary.

        CONTRACT
        - Before training, this is 256 (the raw bytes).
        - After training, it reflects the ACTUAL vocabulary, which is what
          makes the `exactly vocab_size` promise testable.
        """
        raise NotImplementedError

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Fit the tokenizer on `text`.

        CONTRACT
        - After this returns, `self.vocab_size == vocab_size`.
        - OPEN DECISION (ADR-0003): what happens when the corpus runs out of
          pairs to merge before reaching `vocab_size`. Raising, stopping
          early, and padding are all defensible; pick one, write it down,
          and pin test_merge_exhaustion_behaviour to it.
        - `vocab_size` < 256 raises ValueError.
        - Training is deterministic: the same (text, vocab_size) always
          produces a tokenizer that encodes identically.
        - Training twice on the same instance is undefined; construct a new
          instance instead.
        - `verbose` may print progress. It must not change the result.
        """
        if vocab_size < 256:
            raise ValueError(
                f"vocab_size must be at least 256 (the raw bytes), got {vocab_size}"
            )
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode `text` to a list of token ids.

        CONTRACT
        - Every returned id is in range(vocab_size).
        - Accepts any `str`, including one containing characters absent from
          the training corpus.
        - `encode("")` returns `[]`.
        - Deterministic.
        """
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text.

        CONTRACT
        - `decode(encode(s)) == s` for every `str` s.
        - `decode([])` returns `""`.
        - An id outside range(vocab_size) raises ValueError.
        - Ids whose concatenated bytes are not valid UTF-8 must not raise;
          the error policy is your choice, but it must be DOCUMENTED in an
          ADR and pinned by test_decode_invalid_utf8_policy.
        """
        raise NotImplementedError

    def save(self, prefix: str) -> None:
        """Persist the tokenizer to `{prefix}.model` (and any sidecar files).

        CONTRACT
        - Output is deterministic for a given trained tokenizer.
        - Round-trips exactly through `load`.
        """
        raise NotImplementedError

    def load(self, prefix: str) -> None:
        """Restore a tokenizer previously written by `save`.

        CONTRACT
        - After loading, `encode` and `decode` behave identically to the
          tokenizer that was saved.
        """
        raise NotImplementedError
