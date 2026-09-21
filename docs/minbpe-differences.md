# Crucible vs karpathy/minbpe — differences

Phase 1.5 reading exercise (CLAUDE.md, definition of done item 4). Read, not
run: CLAUDE.md records that a run-based diff against minbpe would measure
nothing, since the author of both sides (Claude) has minbpe memorised.

- **minbpe:** commit [`1acefe8`](https://github.com/karpathy/minbpe/tree/1acefe89412b20245db5a22d2a02001e547dc602)
  (2024-04-22, `master` at time of reading), files `minbpe/base.py`,
  `basic.py`, `regex.py`, `gpt4.py`. Line links below are pinned to it.
- **Crucible:** `tokenizer.py` at the commit containing this document.
- **Comparison target:** minbpe's `RegexTokenizer`, the closest analogue. Where
  `BasicTokenizer` differs it is noted. `GPT4Tokenizer` (a wrapper that loads
  cl100k's merges) has no Crucible counterpart and is out of scope.

[base]: https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/base.py
[basic]: https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/basic.py
[regex]: https://github.com/karpathy/minbpe/blob/1acefe89412b20245db5a22d2a02001e547dc602/minbpe/regex.py

## Summary

| Area | minbpe | Crucible | Better |
|---|---|---|---|
| Tie-breaking | first pair in scan order | lexicographically smallest pair | Crucible, narrowly |
| Pre-tokenization pattern | older GPT-4 pattern, `regex` package | current cl100k pattern, stdlib `re` translation | same on real text; Crucible matches upstream |
| Training algorithm | full recount over every chunk occurrence, every merge | word types x counts, incremental pair index + heap | Crucible, for speed; minbpe, for readability |
| Special tokens | replace-on-register, any ids, default raises (via `assert`) | additive, dense ids, default encodes as ordinary text | mixed; see §4 |
| Exhaustion | incidental `max()` error | deliberate `ValueError` naming the counts | Crucible |
| Save / load | pattern text stored but not re-applied on load; specials unescaped | pattern name stored and enforced; specials JSON-escaped | Crucible, except no `.vocab` view |
| Encode strategy | lowest-rank pair, repeat; per chunk | same algorithm | equal |
| Decode | `errors="replace"` | `errors="replace"` (ADR-0004) | equal |
| Argument checks | `assert` | `ValueError` | Crucible |

## 1. Tie-breaking (ADR-0002)

**minbpe.** `pair = max(stats, key=stats.get)` ([regex.py:56][regex]). Among
equal counts, `max` returns the first key in dict order, and `stats` is built
by scanning the chunks left to right ([base.py:13–22][base]). So the tie goes
to **whichever tied pair occurs first in the text**.

**Crucible.** Highest count, then the **lexicographically smallest pair**
(ADR-0002), via a min-heap on `(-count, pair)`.

**Difference in practice.** On `"CD"*50 + "AB"*50`, (C,D) and (A,B) both occur
50 times. minbpe merges CD first, because it appears first; Crucible merges AB
first (Crucible verified by running it; minbpe by reading).

**Which is better.** Both are deterministic in CPython 3.7+, where dicts keep
insertion order. ADR-0002's options table lists "whatever `max()` returns" as
not deterministic; for minbpe specifically that is too strong, because its
insertion order is the scan order, which is stable. The real difference is
what the result depends on. minbpe's tie result depends on corpus order:
reorder the documents and the vocabulary can change. Crucible's depends only
on the pair counts. Order-independence is worth having for reproducibility
across data shuffles, so Crucible's rule is **narrowly better**. It is also a
property of the rule rather than of the traversal, so it survives a refactor
of the loop.

## 2. Pre-tokenization pattern (ADR-0005)

**minbpe.** Uses the third-party `regex` package ([regex.py:12][regex]) and
its own copy of the GPT-4 pattern ([regex.py:19][regex]):

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

**Crucible.** tiktoken's current cl100k pattern (commit `4e71bbe`), translated
to stdlib `re` because `tokenizer.py` is standard-library only:

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s
```

**Difference in practice.** minbpe's copy predates tiktoken's current pattern:
several quantifiers are not possessive, there is no `\s++$` alternative, and
the last alternative is `\s+` rather than `\s`. Measured with the `regex`
package (the two patterns only; minbpe itself not run):

- On all of TinyStories-valid (19.4 MB) the two split identically:
  4,502,956 chunks each.
- On 200,000 random short strings over a whitespace-heavy alphabet, 5,026
  (2.5%) split differently, and **every one** differs only in a whitespace run
  at the end of the input: e.g. `"\n\t"` is `["\n", "\t"]` in minbpe and
  `["\n\t"]` in cl100k.

**Which is better.** For this project, equivalent: TinyStories text never hits
the affected case. Crucible's is better only in that it matches what tiktoken
actually ships. minbpe's use of `regex` is simpler and avoids the translation
risk ADR-0005 had to verify; Crucible pays that cost to stay dependency-free
and to pin the Unicode semantics (`\s` = White_Space, `$` = end of text). The
`regex` package also carries its own, newer Unicode tables; Crucible uses the
running Python's (ADR-0005).

## 3. Training algorithm

**minbpe.** Splits the text into chunks and keeps **every chunk occurrence** as
its own id list ([regex.py:41–44][regex]). Each merge recounts every pair in
every chunk from scratch and rewrites every chunk ([regex.py:49–60][regex]).
Cost is O(chunk occurrences x merges); 19.4 MB of TinyStories is 4.5M chunk
lists, rescanned once per merge. `BasicTokenizer` does the same over the raw
byte stream with no chunking ([basic.py:25–39][basic]).

**Crucible.** Counts **distinct chunk types** with their frequencies (about
9,900 types in 5 MB of TinyStories), builds pair counts once, keeps a
pair → words index and a lazy max-heap, and on each merge updates only the
words that contain the merged pair.

**Difference in practice.** Same merges for the same input, apart from ties
(§1). Speed differs by orders of magnitude: Crucible's own earlier raw-stream
implementation, structurally the same as minbpe's `BasicTokenizer`, took
**259 s** for 1 MB at vocab 1024; the current one takes about 1.2 s
(ADR-0005). minbpe itself was not timed.

**Which is better.** Crucible's, for anything beyond toy corpora. minbpe's is
far easier to read and is explicitly educational: its loop states the BPE
definition directly, which Crucible's heap and stale-index bookkeeping
obscure.

*Lineage note.* Crucible's pre-ADR-0005 implementation (written in a Claude
Cowork session, and replaced before it was ever committed on its own) mirrored
minbpe's structure closely: `_count_pairs` ↔ `get_stats`, `_apply_merge` ↔
`merge`, and an encode loop identical to [basic.py:61–73][basic] down to
`float("inf")` and the membership-check `break`. That is consistent with
CLAUDE.md's note that Claude has minbpe memorised.

## 4. Special tokens

**minbpe.**
- `register_special_tokens` **replaces** the whole mapping
  ([regex.py:72–76][regex]) and validates nothing: ids may collide with
  learned ids, leave gaps, or sit far above the vocabulary (e.g. 100257).
- Training after registering is allowed. A learned id that collides with a
  special id shadows it in `decode`, which checks `vocab` first
  ([regex.py:82–85][regex]).
- `encode(text, allowed_special="none_raise")` is the default: it refuses text
  containing any registered special ([regex.py:137–139][regex]), but via
  `assert`, so the check disappears under `python -O`.
- `allowed_special` must be `"all"`, `"none"`, `"none_raise"` or exactly a
  `set` ([regex.py:140–143][regex]); a list or frozenset is rejected, and
  unknown names in the set are silently ignored.
- Splitting uses one alternation in registration order
  ([regex.py:152–153][regex]). When one special is a prefix of another, the
  one registered first wins, not the longest.

**Crucible.**
- Registration is **additive**, and ids must be exactly the next dense block
  after the current vocabulary; collisions, gaps, empty text and duplicates
  raise `ValueError`. `vocab_size` counts specials, so `range(vocab_size)` is
  every valid id.
- `train()` after registration raises.
- Default `encode(text)` treats literal special-token text as **ordinary
  text**; specials are recognised only when named in `allowed_special` or with
  `"all"`.
- Any collection of names is accepted; a bare string or an unregistered name
  raises.
- Overlapping specials: longest match wins.

**Which is better.**
- **Validation, dense ids, overlap handling, argument checking:** Crucible.
  Each minbpe behaviour listed above is a way to end up with a silently wrong
  vocabulary or encoding.
- **The default for literal special text:** minbpe's refuse-by-default is the
  safer default in general; its docstring calls anything else "a major
  footgun", and tiktoken does the same. Crucible's default never *misreads*
  text as special, which is the failure the Phase 1 tests guard against, but
  it will silently tokenize a document separator as ordinary characters if the
  caller forgets `allowed_special`. In the Phase 2 data pipeline, where
  `<|endoftext|>` appears 21,989 times in the validation file alone, minbpe's
  choice would catch that mistake and Crucible's will not. Worth revisiting
  before Phase 2, as its own decision record. minbpe's version is weakened by
  relying on `assert`.

## 5. Exhaustion (ADR-0003)

**minbpe.** No check. When no pairs remain, `max()` on an empty dict raises
`ValueError: max() arg is an empty sequence` ([regex.py:56][regex]). It stops
at the same point Crucible does, but by accident, with a message that names
neither the merges achieved nor the size requested. A vocab size below 256 is
rejected by `assert` ([regex.py:37][regex]).

**Crucible.** Raises `ValueError("corpus exhausted after N merges; cannot
reach vocab_size=V")` (ADR-0003), and `ValueError` for sizes below 256.

**Which is better.** Crucible: the same stopping point, but deliberate,
documented, tested, and with an actionable message. The vocab sweep parses
that message to find each sample's ceiling; minbpe's error would not support
that.

## 6. Save / load format

**minbpe** ([base.py:97–165][base]):

```
minbpe v1
<the full regex pattern text>
<number of specials>
<special> <id>          one per special, unescaped
<a> <b>                 one per merge; ids implied from 256 upward
```

plus a human-readable `.vocab` file for inspection.

- The `.model` file is opened without an explicit encoding
  ([base.py:106][base]), so it is written in the platform default (cp1252 on
  Windows).
- Special tokens are written raw and read back with `.split()`
  ([base.py:156][base]), so a special containing a space or newline cannot be
  loaded.
- `load()` restores `self.pattern` from the file but **does not recompile
  it**: `RegexTokenizer` compiles its pattern once in `__init__`
  ([regex.py:31–32][regex]) and does not override `load`. A model saved with a
  custom pattern and loaded into a default `RegexTokenizer` encodes with the
  default pattern.

**Crucible:**

```
crucible-bpe v2
pattern cl100k
merges <n>
<a> <b>                 n lines; ids implied from 256 upward
specials <m>
<id> <json string>      m lines, ascending id
```

- UTF-8 and `\n` line endings, explicitly.
- Specials are JSON-escaped, so any text round-trips.
- The pattern is recorded by **name**, and `load()` refuses a mismatch rather
  than encoding differently.
- There is no `.vocab` file.

**Which is better.** Crucible, for correctness: explicit encoding, escaping,
and a pattern that is enforced rather than silently ignored. minbpe's design
stores the pattern text itself, which is more flexible in principle (any
pattern is self-described) but, as implemented, is not honoured on load.
minbpe's `.vocab` file is a genuinely useful inspection aid that Crucible
lacks.

## 7. Encode strategy

**minbpe.** Per chunk: repeatedly find the adjacent pair with the lowest merge
rank, merge all its non-overlapping occurrences left to right, and stop when
no pair has a rank ([regex.py:92–109][regex]); chunk results are concatenated
([regex.py:111–121][regex]).

**Crucible.** The same algorithm. It takes the minimum over a pairwise iterator
instead of building a counts dict each step, and uses `sys.maxsize` rather
than `float("inf")` as the "no merge" rank. Neither caches encoded chunks.

**Which is better.** Equal in results. Both are pure Python and slow next to
tiktoken (C1: Crucible 0.33 MB/s vs cl100k 6.7 MB/s); a per-chunk cache would
help both.

## 8. Smaller differences

- **Decode.** Both use `errors="replace"` (Crucible: ADR-0004). For an unknown
  id, minbpe's `RegexTokenizer` raises `ValueError`, and `BasicTokenizer`
  raises a bare `KeyError` ([basic.py:53][basic]). Crucible raises
  `ValueError`, and decodes special ids.
- **Vocabulary size contract.** minbpe has no `vocab_size` and no guarantee
  that the result has exactly the requested size. Crucible has both
  (ADR-0003).
- **Argument checks.** minbpe uses `assert` throughout (`vocab_size >= 256`,
  model version, `.model` suffix), all stripped under `python -O`. Crucible
  raises `ValueError`.
