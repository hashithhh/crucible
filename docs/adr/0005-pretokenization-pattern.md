# ADR-0005 — Pre-tokenization pattern

- **Status:** accepted
- **Date:** 2026-09-17

## Context
Before this decision, `train()` merged over the raw byte stream: every merge
recounted every adjacent pair in the corpus and rewrote the whole id list.
That is O(corpus x merges). On 1 MB of TinyStories at vocab 1024 it took
**259 s**, and ADR-0001 needs sweeps over up to 20x that corpus at many
times that many merges.

Splitting text into chunks first ("pre-tokenization") changes the unit of
work: training runs over distinct chunk types weighted by frequency, and a
merge only touches the types that contain its pair. It also changes *what*
is learned: no token can span a chunk boundary, so the vocabulary is spent on
words and word pieces instead of cross-word fragments like `"e c"`.

The split pattern is therefore a modelling decision, not only a speed-up. It
fixes which token boundaries can ever exist.

## Options considered
| Option | Note |
|---|---|
| No pre-tokenization | status quo; too slow; spends vocab on cross-word fragments |
| Split on whitespace | simple; glues punctuation onto words (`"end."`); digits unbounded |
| GPT-2 pattern (r50k) | contractions case-sensitive; long digit runs become tokens |
| **GPT-4 pattern (cl100k)** | case-insensitive contractions; digits in groups of <=3; newline handling; widely studied |
| GPT-4o pattern (o200k) | adds case-aware splitting for camelCase and non-Latin scripts; more complex, less studied |

## Decision
**The cl100k (GPT-4) pattern**, taken verbatim from tiktoken's source rather
than retyped:

- Source: `tiktoken_ext/openai_public.py`, `cl100k_base()["pat_str"]`, at
  commit [`4e71bbe`](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken_ext/openai_public.py#L89),
  fetched 2026-09-17.

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s
```

This is the *current* upstream text. It differs from the version most often
quoted from memory: possessive quantifiers throughout, `\s++$` as its own
alternative, and `\s*[\r\n]` where older copies have `\s*[\r\n]+`. Retyping
would likely have produced the older one.

### Translation to stdlib `re`
`tokenizer.py` is standard library only, and tiktoken runs this pattern on a
Rust regex engine. Python's `re` differs in three ways; each is translated
exactly, not approximated:

| Construct | Rust (tiktoken) | Python `re` | Translation |
|---|---|---|---|
| `\p{L}`, `\p{N}` | Unicode general categories | unsupported | explicit classes built from `unicodedata` categories `L*`, `N*` |
| `\s` | Unicode `White_Space` (25 code points) | also matches U+001C–U+001F (29) | explicit `White_Space` class |
| `$` | end of text | end of text *or before a final `\n`* | `\Z` |

The obvious shortcuts are wrong: `[^\W\d_]` for letters counts `½` and `²`
as letters, and `\d` for numbers misses them.

Possessive quantifiers require Python >= 3.11; `requires-python` was raised
accordingly.

## Rationale
- **Measured speed-up.** Same machine, same 999,948 bytes of
  TinyStories-valid (`load_corpus(1_000_000)`), vocab 1024:

  | | train | encode 100 KB | bytes/token |
  |---|---|---|---|
  | raw stream (before) | 259.21 s | 26.54 s | 3.695 |
  | cl100k chunks (after) | 1.21–1.25 s (3 runs) | 0.28–0.30 s | 3.288 |

  The "after" train time includes the one-time 0.45 s build of the Unicode
  classes. At real scale: the full 19.4 MB validation set trains at vocab
  8192 in 13.2 s.
- **cl100k over GPT-2's pattern:** case-insensitive contractions (`'S`, `'LL`)
  and digit grouping (`\p{N}{1,3}`) are both cheap wins; unbounded digit runs
  waste vocabulary on specific numbers.
- **cl100k over o200k:** TinyStories is simple English prose. o200k's extra
  casing rules target code and multilingual text; its benefit here is
  unmeasured and it is harder to translate and verify. Revisit only with data.

### Verification of the translation
Checked once against the third-party `regex` module (which supports `\p{..}`
and possessives natively) running tiktoken's pattern, with the two Rust
semantics above normalised on the reference side:

- handpicked cases covering every alternative and every trap listed above,
- 20,000 Hypothesis-generated strings over all assigned code points,
- 2 MB of TinyStories (459,568 chunks),

all identical. Pinned in the suite by `tests/test_pretokenization.py`, which
was mutation-tested: stdlib `\s`, `[^\W\d_]` letters, ungrouped digits, and no
pre-tokenization at all are each caught.

## Consequences
- **Compression per token drops at a given vocab size.** 3.695 → 3.288
  bytes/token at vocab 1024, because tokens can no longer span words.
  ADR-0001 must be measured under this pattern; raw-stream numbers are not
  comparable.
- **Merge ceilings drop** (ADR-0003). Repetitive test corpora support fewer
  merges: `ASCII_CORPUS` 48 → 19, `UNICODE_CORPUS` 77 → 60. The ASCII
  exhaustion test and the Unicode training fixture were re-pinned to the
  measured values.
- **Unicode version.** Classes come from the running Python's `unicodedata`
  (15.1.0 on 3.13). Code points assigned in later Unicode versions — 9,039
  relative to `regex` 2026.9.10 — are treated as unassigned, i.e. neither
  letters nor numbers. Every code point Python assigns to `L*`/`N*` agrees
  with `regex`. A Python upgrade can therefore change the chunking of newly
  assigned characters; TinyStories contains none.
- **The pattern is part of the model.** `.model` files record
  `pattern cl100k`, and `load` refuses a model saved under a different
  pattern rather than silently encoding differently. The format moved to
  `crucible-bpe v2`; v1 files (raw-stream merges) are rejected.
- **First use costs ~0.45 s** to build the Unicode classes, once per process.
