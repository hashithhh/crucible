# Crucible Phase 1 — a byte-level BPE tokenizer, and four things I got wrong first

**Status:** Phase 1 of 5 complete; Phase 2's transformer is built and green. 153 tests.
Repo: `github.com/hashithhh/crucible`
**Date:** 2026-09-25

Crucible is a small language model built from scratch — tokenizer, transformer, training run,
four ablations, and a scaling study checked against Chinchilla's prediction. Phase 1 is the
tokenizer. This is what it does, what I measured, and the four places my first answer was wrong.

**On how this was built:** the implementation was written by Claude, not by hand. That was a
deliberate change of plan partway through, and it has consequences I'll come back to at the end.
Every commit names who wrote the code in it. I'd rather say this up front than have someone find
it in the git log.

---

## What it is

A byte-level BPE tokenizer, standard library only — no `regex`, no `tiktoken`, no `tokenizers`.

- Trains on raw text, learns an ordered list of merges
- `encode` / `decode` round-trip every string, including text it has never seen
- cl100k regex pre-tokenisation, so merges never cross word boundaries
- Special tokens that survive `save`/`load`
- Vocabulary: **2,048**, chosen by measurement

Seven architecture decision records, each with the numbers behind it, live in `docs/adr/`.

---

## The decisions, and the numbers

### Vocabulary size: 2,048

I swept vocabulary sizes over 5 MB of TinyStories and measured bytes/token on a held-out
500 KB slice — 574 stories from the far end of the file, no overlap with training.

| Vocab | Bytes/token | Gain per doubling | Embedding params (d=512) |
|---|---|---|---|
| 512 | 2.365 | — | 0.26M |
| 1,024 | 3.148 | +33.1% | 0.52M |
| **2,048** | **3.726** | **+18.4%** | **1.05M** |
| 4,096 | 4.078 | +9.4% | 2.10M |
| 8,192 | 4.260 | +4.5% | 4.19M |
| 14,143 (ceiling) | 4.291 | +0.7% | 7.24M |

Compression saturates. The first few hundred merges buy most of what's available, because
language is dominated by a small number of very common sequences.

### The model is sized to the data, not to a round number

The original plan said ~100M parameters. TinyStories yields about **510M unique tokens** at
3.726 bytes/token. Chinchilla's rule of roughly 20 tokens per parameter puts the compute-optimal
model near **25M**, not 100M. Training a 100M model on that corpus would be the wrong
configuration, and it would make the Phase 5 scaling study measure the wrong thing.

Final config — **S25: d_model 512, 8 layers, 8 heads, 26.2M parameters**, 19.5 tokens/param.

I could have fetched more data instead. I chose not to: more parameters on the same corpus is the
one option that makes the scaling study meaningless, and the corpus is what I actually have.

---

## Four things I got wrong first

### 1. I wrote a test that no implementation could pass

The first test suite asserted that compression improves monotonically up to vocabulary 600 — 344
merges — on a corpus that was one sentence repeated forty times. That corpus has **24 distinct
byte pairs**. It collapses to a single token after about 90 merges, and then there are no pairs
left to count at all.

Not a hard test. An impossible one. It was caught by counting bigrams before writing any
implementation, which cost about two minutes and saved an evening of debugging code that was fine.

Merge ceilings turn out to be a property of the corpus, not the algorithm:

| Corpus | Max merges, raw stream | Max merges, cl100k chunks |
|---|---|---|
| One sentence, repeated | 48 | 19 |
| Mixed-Unicode sample | 77 | 60 |
| Lexically varied text | >1,744 | 348 |
| 5 MB TinyStories | — | 13,887 |

The ceiling is not even a property of the corpus alone: confining merges to
word-like chunks (the pre-tokenisation decision below) cut every one of these.
Two test fixtures had to be re-pinned when that landed.

That's also why vocabulary 16,384 and 32,768 were unreachable on a 5 MB sample, and are recorded
as unreachable rather than quietly skipped.

### 2. My knee-detection result wasn't robust

I picked 2,048 using the knee of the compression curve — the point furthest above the line joining
the endpoints. Then I checked whether the answer survived changing the setup. It didn't:

| Variant | Knee |
|---|---|
| Full grid, log2 axis | 2,048 |
| Drop the 512 point | 2,048 (margin 0.017) |
| Drop 512, include the ceiling point | **4,096** |
| Linear vocab axis | **4,096** |
| Drop the 8,192 point | 2,048 (margin 0.0042 — a tie) |

The method measures distance from the chord between the **first and last** points, so the lowest
point in the grid has outsized leverage, and 512 was an arbitrary place to start. On a linear
vocabulary axis — arguably the decision-relevant one, since embedding cost is linear in vocabulary
— 4,096 wins most variants.

A rule that changes its answer when you change the axis cannot carry a decision. I re-grounded the
choice on something stable instead: the corpus is token-limited, and the smaller vocabulary yields
more training tokens from it.

### 3. My pre-registered gate was badly specified

Before measuring anything I registered C1: *our tokenizer within 10% of `tiktoken` on compression;
worse than 30% means the implementation is broken.*

The measurement:

| Tokenizer | Vocab | Bytes/token | vs cl100k | Encode MB/s |
|---|---|---|---|---|
| **Ours** | **2,048** | **3.726** | **+13.6% tokens** | 0.33 |
| Ours, at sample ceiling | 14,143 | 4.291 | −1.4% | 0.35 |
| cl100k_base | 100,277 | 4.232 | — | 6.69 |
| o200k_base | 200,019 | 4.263 | −0.7% | 11.23 |

13.6% worse. Misses the target. And it means nothing, because a 2,048-token vocabulary against
100,277 is a **49× gap** — that comparison was never like-for-like. At a vocabulary large enough to
be comparable, the same code is 1.4% *better* than cl100k.

So the gap is the vocabulary choice I made on purpose, not a defect in the code. A gate that
"failed" here would have been overruling a recorded decision, not catching a bug. I've recorded C1
as **measured, not a valid gate**, and left it visible rather than quietly rewriting the target
after seeing the number.

The throughput gap is real and separate: tiktoken is 20–34× faster — a compiled Rust core against
pure Python. At 0.33 MB/s, encoding the 1.9 GB training file takes about 97 minutes on one core.
That's a Phase 3 data-prep cost, and it parallelises across stories.

### 4. The decision survived, but its reasoning inverted

ADR-0001 chose 2,048 over 4,096 *primarily* because a token-limited corpus makes extra tokens
valuable. That was written against a 100M-parameter budget, where both options landed near 25% of
Chinchilla-optimal — hopeless either way, so the token argument carried real weight.

At 25M the picture flips:

| | 2,048 | 4,096 |
|---|---|---|
| Training tokens | 510M | 466M |
| Share of Chinchilla target (~524M) | **97%** | **89%** |
| Embedding, share of params | 4.0% | 7.7% |

Both are nearly adequate on tokens now, so the argument that decided it almost vanishes. Meanwhile
the parameter-cost argument it treated as secondary got about **2.7× stronger**.

Same answer, different reasons. I amended the ADR rather than rewriting it, because the interesting
part is that the justification moved. When a budget changes, re-check *why* a decision held — not
just whether it still holds.

---

## What differs from minbpe

I read `karpathy/minbpe` only after the implementation was finished and green. Nine differences:

| Area | minbpe | Crucible |
|---|---|---|
| Tie-breaking | first pair in scan order | lexicographically smallest pair |
| Pre-tokenisation | older GPT-4 pattern, `regex` package | current cl100k pattern, stdlib `re` |
| Training | full recount every merge | word types × counts, incremental index + heap |
| Special tokens | replace-on-register, `assert` | additive, dense ids, JSON-escaped |
| Exhaustion | incidental `max()` error | deliberate `ValueError` |
| Save/load | pattern stored, not enforced | pattern enforced on load |
| Argument checks | `assert` | `ValueError` |
| Encode strategy | lowest-rank pair, repeat | identical |
| Decode | `errors="replace"` | identical |

The tie-break one matters most. `max()` over a counts dictionary inherits insertion order on ties,
which makes training non-reproducible across runs in a way nothing visibly complains about.
Ordering on the pair itself is a total order and doesn't.

The `assert` one is worth knowing generally: assertions are stripped under `python -O`, so a
library that validates arguments with `assert` silently stops validating in optimised runs.

minbpe wins on readability, and that's not a small thing for a teaching repo.

---

## The honest part

The original plan was to hand-write the tokenizer, then diff against minbpe as an independent
check. I dropped the hand-writing partway through and let Claude write it.

That kills the check. Claude has minbpe memorised; diffing Claude's implementation against it
measures nothing. So Phase 1.5 became a reading exercise rather than a verification step, and the
comparison gate came out of the definition of done rather than being left in to be quietly ignored.

The plan was that Phase 2 would restore the check: I would write the transformer, and G1 would
have me rebuild from memory, a week later, blank file, no reference — code I had written. Then I
asked Claude to write Phase 2 as well. So G1 now tests recall of code I *read*, which is a weaker
thing to measure. I am keeping the gate and recording it as that, rather than quietly scoring it
as if I had written the code.

Two of the five phases are now Claude-authored. The pattern is worth naming: each time, the
decision was mine and the typing was not, and each time the check that would have measured my
understanding is the thing that got dropped.

---

## What's next

Phase 2's transformer is built: RMSNorm, RoPE, causal attention with a KV cache, and sampling,
under 50 tests that pin the properties that matter — RoPE dot products depending only on relative
offset, cached decode matching a full forward, and gradients at step *t* unaffected by later
tokens. S25 trains on a 6 GB laptop GPU at about 40,000 tokens/s, peaking at 3.9 GB at
micro-batch 32, so one epoch over 510M tokens is roughly 3.5 hours locally.

Still to come: the G1 rebuild, then training S25, four ablations (learning rate, warmup, dense vs
MoE, full vs hybrid attention), and a four-point scaling study against Chinchilla.

Everything above is reproducible from the repo: `scripts/sweep_vocab.py`,
`scripts/compare_tiktoken.py`, `scripts/model_budget.py`, with raw output in `results/`.
