"""Byte-level BPE tokenizer — Crucible Phase 1.

Written by Claude, not by hand; see CLAUDE.md and the git log.
Standard library only.
"""

from __future__ import annotations

import heapq
import json
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Collection
from functools import cache
from itertools import pairwise
from typing import Literal

# --- Pre-tokenization (ADR-0005) --------------------------------------------
#
# GPT-4's split pattern, copied verbatim from tiktoken's source:
#   https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken_ext/openai_public.py#L89
#   (cl100k_base, `pat_str`; commit 4e71bbe, fetched 2026-09-17)
#
# It is kept here as the reference text only. tiktoken runs it on a Rust regex
# engine; Python's `re` differs in three ways, so the pattern actually used is
# a translation, built by _split_pattern() below:
#
#   1. `\p{L}` / `\p{N}` are unsupported. Replaced by explicit classes built
#      from unicodedata general categories L* and N*.
#   2. `\s` in `re` also matches U+001C..U+001F, which are not Unicode
#      White_Space. Replaced by the exact White_Space set.
#   3. `$` in `re` also matches before a final "\n". Replaced by `\Z`.
#
# Possessive quantifiers (`?+`, `++`, `*+`, `{1,3}+`) need Python >= 3.11.
PATTERN_NAME = "cl100k"
TIKTOKEN_CL100K_PAT_STR = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""  # noqa: E501

# Unicode White_Space property, from PropList.txt. Stable since Unicode 6.3.
_WHITE_SPACE = (
    "\t\n\x0b\x0c\r\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"
)

MODEL_HEADER = "crucible-bpe v2"


def _range(lo: int, hi: int) -> str:
    esc = f"\\U{lo:08x}"
    return esc if lo == hi else f"{esc}-\\U{hi:08x}"


def _category_class(major: str) -> str:
    """Character-class body matching every code point in category `major`*."""
    parts: list[str] = []
    start = prev = -2
    for cp in range(sys.maxunicode + 1):
        if unicodedata.category(chr(cp))[0] != major:
            continue
        if cp != prev + 1:
            if start >= 0:
                parts.append(_range(start, prev))
            start = cp
        prev = cp
    if start >= 0:
        parts.append(_range(start, prev))
    return "".join(parts)


@cache
def _split_pattern() -> re.Pattern[str]:
    """The cl100k pattern, translated to stdlib `re`. Built once (~0.4s)."""
    L = _category_class("L")
    N = _category_class("N")
    S = _WHITE_SPACE
    return re.compile(
        "|".join(
            [
                r"'(?i:[sdmt]|ll|ve|re)",
                rf"[^\r\n{L}{N}]?+[{L}]++",
                rf"[{N}]{{1,3}}+",
                rf" ?[^{S}{L}{N}]++[\r\n]*+",
                rf"[{S}]++\Z",
                rf"[{S}]*[\r\n]",
                rf"[{S}]+(?![^{S}])",
                rf"[{S}]",
            ]
        )
    )


def _pretokenize(text: str) -> list[str]:
    return _split_pattern().findall(text)


# --- Tokenizer ---------------------------------------------------------------


class Tokenizer:
    """A byte-level BPE tokenizer.

    Contract notes that apply to the whole class:

    - Token ids 0..255 are reserved for raw bytes. Any learned token takes
      an id >= 256.
    - The tokenizer operates on the UTF-8 encoding of text, not on Unicode
      code points, so any `str` is representable.
    - Text is split into chunks by the cl100k pattern (ADR-0005) before any
      merge is learned or applied. No token ever spans two chunks.
    - Special tokens, if registered, take the ids directly after the learned
      vocabulary, so ids stay dense: `range(vocab_size)` is every valid id.
    """

    def __init__(self) -> None:
        # merges maps an ordered byte-pair to the id it was merged into.
        # Insertion order IS merge order, which save() relies on.
        self.merges: dict[tuple[int, int], int] = {}
        # vocab maps every ordinary id to the bytes it expands to.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # special_tokens maps literal text to id; ids follow the ordinary vocab.
        self.special_tokens: dict[str, int] = {}
        self._special_by_id: dict[int, bytes] = {}

    @property
    def vocab_size(self) -> int:
        """Number of entries in the vocabulary, special tokens included.

        CONTRACT
        - Before training, this is 256 (the raw bytes).
        - After training, it reflects the ACTUAL vocabulary, which is what
          makes the `exactly vocab_size` promise testable.
        - Each registered special token adds one.
        """
        return len(self.vocab) + len(self.special_tokens)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Fit the tokenizer on `text`.

        CONTRACT
        - After this returns, `self.vocab_size == vocab_size`.
        - ADR-0003: if the corpus runs out of pairs to merge before reaching
          `vocab_size`, raises ValueError mentioning "exhausted".
        - `vocab_size` < 256 raises ValueError.
        - Merges are learned within pre-tokenized chunks only (ADR-0005).
        - Ties in pair frequency go to the lexicographically smallest pair
          (ADR-0002).
        - Training is deterministic: the same (text, vocab_size) always
          produces a tokenizer that encodes identically.
        - Training twice on the same instance is undefined; construct a new
          instance instead. Training after registering special tokens raises
          ValueError, because learned ids would collide with theirs.
        - `verbose` may print progress. It must not change the result.
        """
        if vocab_size < 256:
            raise ValueError(
                f"vocab_size must be at least 256 (the raw bytes), got {vocab_size}"
            )
        if self.special_tokens:
            raise ValueError(
                "train() after register_special_tokens(): learned ids would "
                "collide with special token ids. Train first, then register."
            )
        n_merges = vocab_size - 256
        if n_merges == 0:
            return

        # Each distinct chunk is a word type, trained on once and weighted by
        # how often it occurs. Single-byte chunks have no pairs; drop them.
        words: list[list[int]] = []
        freqs: list[int] = []
        for chunk, count in Counter(_pretokenize(text)).items():
            ids = list(chunk.encode("utf-8"))
            if len(ids) > 1:
                words.append(ids)
                freqs.append(count)

        # pair -> weighted count, and pair -> indices of words containing it.
        # `where` may hold stale indices after merges; merging a word that no
        # longer contains the pair is a no-op, so staleness costs time only.
        counts: dict[tuple[int, int], int] = {}
        where: dict[tuple[int, int], set[int]] = {}
        for w, (ids, freq) in enumerate(zip(words, freqs, strict=True)):
            for pair in pairwise(ids):
                counts[pair] = counts.get(pair, 0) + freq
                where.setdefault(pair, set()).add(w)

        # Min-heap on (-count, pair): the top is the highest count and, among
        # equal counts, the smallest pair — the ADR-0002 winner. Entries go
        # stale as counts change; a popped entry is trusted only if it still
        # matches the live count.
        heap = [(-c, pair) for pair, c in counts.items()]
        heapq.heapify(heap)

        for i in range(n_merges):
            pair = self._pop_best(heap, counts)
            if pair is None:
                # ADR-0003: raise rather than return a smaller vocabulary.
                raise ValueError(
                    f"corpus exhausted after {i} merges; cannot reach "
                    f"vocab_size={vocab_size}"
                )
            new_id = 256 + i
            touched: set[tuple[int, int]] = set()
            for w in where.pop(pair):
                old = words[w]
                new = self._apply_merge(old, pair, new_id)
                if len(new) == len(old):
                    continue  # stale index
                freq = freqs[w]
                for p in pairwise(old):
                    counts[p] -= freq
                    touched.add(p)
                for p in pairwise(new):
                    counts[p] = counts.get(p, 0) + freq
                    where.setdefault(p, set()).add(w)
                    touched.add(p)
                words[w] = new
            touched.discard(pair)
            del counts[pair]
            for p in touched:
                c = counts[p]
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    del counts[p]
                    where.pop(p, None)

            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}/{n_merges}: {pair} -> {new_id}")

    @staticmethod
    def _pop_best(
        heap: list[tuple[int, tuple[int, int]]],
        counts: dict[tuple[int, int], int],
    ) -> tuple[int, int] | None:
        """Most frequent live pair; ties to the smallest pair (ADR-0002)."""
        while heap:
            neg, pair = heapq.heappop(heap)
            if counts.get(pair) == -neg:
                return pair
        return None

    @staticmethod
    def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every non-overlapping occurrence of `pair`, left to right."""
        out: list[int] = []
        i = 0
        n = len(ids)
        a, b = pair
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def register_special_tokens(self, tokens: dict[str, int]) -> None:
        """Add special tokens, mapping literal text to id.

        CONTRACT
        - Ids must not already be in use, and together with the existing ids
          must leave the vocabulary dense: the new ids are exactly
          `vocab_size .. vocab_size + len(tokens) - 1`, in any assignment.
          Anything else raises ValueError.
        - Text must be a non-empty str not already registered.
        - Survives save/load.
        - Registration does not change how ordinary text encodes: literal
          special-token text is only treated as special when `encode` is
          explicitly told to allow it.
        """
        if not tokens:
            return
        base = self.vocab_size
        for text, idx in tokens.items():
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"special token text must be a non-empty str: {text!r}"
                )
            if text in self.special_tokens:
                raise ValueError(f"special token already registered: {text!r}")
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ValueError(f"special token id must be an int: {idx!r}")
        if set(tokens.values()) != set(range(base, base + len(tokens))):
            raise ValueError(
                f"special token ids must be exactly {base}..{base + len(tokens) - 1}"
                f" (dense, directly after the current vocabulary); "
                f"got {sorted(tokens.values())}"
            )
        for text, idx in tokens.items():
            self.special_tokens[text] = idx
            self._special_by_id[idx] = text.encode("utf-8")

    def encode(
        self,
        text: str,
        allowed_special: Collection[str] | Literal["all"] = (),
    ) -> list[int]:
        """Encode `text` to a list of token ids.

        CONTRACT
        - Every returned id is in range(vocab_size).
        - Accepts any `str`, including one containing characters absent from
          the training corpus.
        - `encode("")` returns `[]`.
        - Deterministic.
        - Text is pre-tokenized exactly as in training; merges never cross a
          chunk boundary.
        - Special tokens: literal special-token text is encoded as ORDINARY
          text unless named in `allowed_special` (or `allowed_special="all"`).
          Naming a string that is not a registered special token raises
          ValueError. Where allowed specials overlap, the longest match wins.
        """
        if allowed_special == "all":
            allowed = set(self.special_tokens)
        else:
            if isinstance(allowed_special, str):
                raise ValueError(
                    'allowed_special takes a collection of strings or "all", '
                    f"not the string {allowed_special!r}"
                )
            allowed = set(allowed_special)
            unknown = allowed - self.special_tokens.keys()
            if unknown:
                raise ValueError(f"not registered special tokens: {sorted(unknown)}")

        if not allowed:
            return self._encode_ordinary(text)

        # Longest first, so that when one special is a prefix of another the
        # longer wins; then by text, so the alternation order is deterministic.
        specials = sorted(allowed, key=lambda s: (-len(s), s))
        splitter = re.compile("|".join(re.escape(s) for s in specials))
        out: list[int] = []
        pos = 0
        for m in splitter.finditer(text):
            out.extend(self._encode_ordinary(text[pos : m.start()]))
            out.append(self.special_tokens[m.group()])
            pos = m.end()
        out.extend(self._encode_ordinary(text[pos:]))
        return out

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in _pretokenize(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def _encode_chunk(self, raw: bytes) -> list[int]:
        ids = list(raw)
        merges = self.merges
        # Apply the earliest-learned merge present, repeatedly. Merge ids are
        # assigned in learning order, so the lowest id is the earliest merge.
        while len(ids) >= 2:
            best = min(pairwise(ids), key=lambda p: merges.get(p, sys.maxsize))
            new_id = merges.get(best)
            if new_id is None:
                break
            ids = self._apply_merge(ids, best, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text.

        CONTRACT
        - `decode(encode(s)) == s` for every `str` s, with or without
          `allowed_special`.
        - `decode([])` returns `""`.
        - An id outside range(vocab_size) raises ValueError.
        - A special token id decodes to its literal text.
        - ADR-0004: invalid UTF-8 is replaced (U+FFFD), never raised.
        """
        parts: list[bytes] = []
        for i in ids:
            piece = self.vocab.get(i)
            if piece is None:
                piece = self._special_by_id.get(i)
                if piece is None:
                    raise ValueError(f"token id {i} is outside the vocabulary")
            parts.append(piece)
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, prefix: str) -> None:
        """Persist the tokenizer to `{prefix}.model`.

        CONTRACT
        - Output is deterministic for a given trained tokenizer.
        - Round-trips exactly through `load`, special tokens included.

        FORMAT (UTF-8, "\\n" line endings)
            crucible-bpe v2
            pattern cl100k
            merges <n>
            <a> <b>            n lines; line k defines id 256+k
            specials <m>
            <id> <json str>    m lines, ascending id
        """
        lines = [MODEL_HEADER, f"pattern {PATTERN_NAME}", f"merges {len(self.merges)}"]
        lines += [f"{a} {b}" for a, b in self.merges]
        lines.append(f"specials {len(self.special_tokens)}")
        for text, idx in sorted(self.special_tokens.items(), key=lambda kv: kv[1]):
            lines.append(f"{idx} {json.dumps(text, ensure_ascii=True)}")
        with open(f"{prefix}.model", "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")

    def load(self, prefix: str) -> None:
        """Restore a tokenizer previously written by `save`.

        CONTRACT
        - After loading, `encode` and `decode` behave identically to the
          tokenizer that was saved, special tokens included.
        - A model with a different header or pre-tokenization pattern raises
          ValueError rather than silently encoding differently.
        """
        with open(f"{prefix}.model", encoding="utf-8") as f:
            lines = f.read().split("\n")

        def field(line: str, key: str) -> str:
            head, _, value = line.partition(" ")
            if head != key:
                raise ValueError(f"malformed model: expected {key!r}, got {line!r}")
            return value

        if lines[0] != MODEL_HEADER:
            raise ValueError(
                f"unrecognised model header {lines[0]!r}; expected {MODEL_HEADER!r}"
            )
        pattern = field(lines[1], "pattern")
        if pattern != PATTERN_NAME:
            raise ValueError(
                f"model uses pre-tokenization pattern {pattern!r}, "
                f"this tokenizer uses {PATTERN_NAME!r}"
            )
        n_merges = int(field(lines[2], "merges"))
        pos = 3

        merges: dict[tuple[int, int], int] = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for k in range(n_merges):
            a, b = (int(x) for x in lines[pos + k].split())
            if a not in vocab or b not in vocab:
                raise ValueError(f"malformed model: merge {k} uses an unknown id")
            merges[(a, b)] = 256 + k
            vocab[256 + k] = vocab[a] + vocab[b]
        pos += n_merges

        n_specials = int(field(lines[pos], "specials"))
        specials: dict[str, int] = {}
        for line in lines[pos + 1 : pos + 1 + n_specials]:
            idx, _, text = line.partition(" ")
            specials[json.loads(text)] = int(idx)

        self.merges, self.vocab = merges, vocab
        self.special_tokens, self._special_by_id = {}, {}
        self.register_special_tokens(specials)
