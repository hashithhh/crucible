# ADR-0001 — Tokenizer vocabulary size

- **Status:** accepted
- **Date:** 2026-09-16 (proposed), 2026-09-17 (decided from measurement)

## Amendment, 2026-09-22 — the parameter percentages used a superseded budget

Everything below is left as written on 2026-09-17. One input has since
changed: this ADR costed the vocabulary against a ~100M-parameter model at
`d_model=768`, and **ADR-0006 replaced that with ~25M at `d_model=512`**
(S25: 26,223,616 parameters). Every "% of budget" figure below is therefore
stale. Corrected:

| Vocab | Embedding params | Was (100M, d=768) | Now (25M, d=512) | Share of the actual S25 model |
|---:|---:|---:|---:|---:|
| 2,048 | 1,049,088 | 1.57% (1,572,864 params) | **4.20%** | 4.00% |
| 4,096 | 2,097,664 | 3.15% (3,145,728 params) | **8.39%** | 7.69% |

The "Now" column uses the round 25M, for comparability with the old figures;
the last column divides by each model's own total, which is how ADR-0006
quotes them (4.0% and 7.7%).

`scripts/sweep_vocab.py` has been repointed at S25 (d_model 512, 26,223,616
parameters) and `results/vocab_sweep.{csv,json,png}` regenerated on
2026-09-25, so the figure and CSV now report shares on that basis: 4.00% at
2,048, 8.00% at 4,096. Bytes/token is unchanged, as it must be. The Results
table below still shows the original 100M / d_model 768 figures, as written.

**The decision does not change, and the reasoning behind it gets stronger.**
The secondary argument here was that the choice between 2,048 and 4,096 was
cheap: ~1.6 points of the parameter budget. Under ADR-0006 the same step
costs 1,048,576 parameters, **4.2 points of a 25M budget**: the parameter
cost of a larger vocabulary is now **2.7x more significant** (4.20% against
1.57%; 2.5x measured against each model's own total). That argues for the
smaller vocabulary, which is what was chosen.

The primary argument, the training-token budget, moved the other way and is
weaker. ADR-0006's Consequences section records it: 510.5M tokens is 97% of
the Chinchilla target for 25M, and 4,096's 466.6M is 89%.

## Context
Vocabulary size is not a tokenizer-local choice. The embedding table is
`vocab_size x d_model` parameters, and at `d_model=768` a GPT-2-sized 50,257
vocab costs ~38M parameters — roughly 38% of the ~100M budget — on a corpus
(TinyStories) built around a deliberately small vocabulary.

This decision propagates into Phase 3's parameter count and Phase 5's
scaling study. Choosing it by default pollutes both.

## Decision
**2,048 learned tokens** (256 bytes + 1,792 merges), under the cl100k
pre-tokenization of ADR-0005. Special tokens (at minimum `<|endoftext|>`) are
registered on top and take ids 2,048 onward, so the embedding table has
2,048 + (number of special tokens) rows.

- **Measured compression at 2,048:** **3.73 bytes/token** (3.7258) on held-out
  TinyStories.
- **Parameter cost:** 2,048 x 768 = **1,572,864 input-embedding parameters,
  1.57% of the 100M budget.** An untied output projection would double that
  to 3.1M (3.1%).

## Method
`scripts/sweep_vocab.py`, outputs in `results/vocab_sweep.{csv,json,png}`.

| Constant | Value | Why |
|---|---|---|
| `VOCAB_SIZES` | 512 … 32,768, powers of two | spans "too small to hold common words" to "beyond this corpus"; doubling gives even spacing on the log axis the knee is found on |
| `TRAIN_BYTES` | 5,000,000 | its merge ceiling (13,887) clears every grid size up to 8,192, and bytes/token at 2,048 and 4,096 agrees with the 2.5 MB and 10 MB samples to within 0.016; small enough to sweep in under 2 minutes |
| `HELDOUT_BYTES` | 500,000 | 116k–211k tokens per size: a stable ratio, and ~1 s to encode |
| `D_MODEL` | 768 | the Phase 3 planning width |
| `PARAM_BUDGET` | 100,000,000 | the project's model size |
| `TIMING_REPEATS` | 3 (median) | single timings vary run to run |

- **Corpus:** `TinyStories-valid.txt` (SHA-256 `94e43181…`), split into 21,989
  stories on `<|endoftext|>` lines. The separator is **removed**: it will be a
  special token, one id at any vocab size, and learning its text would both
  spend merges on it and inflate bytes/token. The first story is dropped
  because the file starts mid-story.
- **Train / held-out:** 6,101 stories (4,999,982 bytes) from the start of the
  file; 574 stories (499,260 bytes) from the end. Disjoint by construction.
- **Knee:** the point at **maximum distance above the endpoint chord**, over
  (log2 vocab, held-out bytes/token) for the grid sizes that could be trained.
  Both axes are normalised to [0, 1] by the first and last points; the knee is
  the point furthest above the straight line joining them. This is only the
  core idea of Kneedle (after Satopaa et al., 2011) — no smoothing, no
  sensitivity parameter — and, as Robustness shows, it inherits that idea's
  sensitivity to the endpoints.
- **Hardware:** AMD Ryzen 7 7735HS (16 logical CPUs; training is
  single-threaded), 15.3 GB RAM, Windows 11, Python 3.13.5.

## Results

| Vocab | Bytes/token | Gain vs previous | Embedding params | Share of 100M |
|---:|---:|---:|---:|---:|
| 512 | 2.3653 | — | 393,216 | 0.39% |
| 1,024 | 3.1478 | +33.1% | 786,432 | 0.79% |
| **2,048** | **3.7258** | **+18.4%** | **1,572,864** | **1.57%** |
| 4,096 | 4.0775 | +9.4% | 3,145,728 | 3.15% |
| 8,192 | 4.2596 | +4.5% | 6,291,456 | 6.29% |
| 14,143 *(ceiling)* | 4.2908 | +0.7% | 10,861,824 | 10.86% |
| 16,384 | unreachable | — | 12,582,912 | 12.58% |
| 32,768 | unreachable | — | 25,165,824 | 25.17% |

Normalised distance above the endpoint chord: 512 → 0.000, 1,024 → 0.163,
**2,048 → 0.218**, 4,096 → 0.154, 8,192 → 0.000.

**16,384 and 32,768 cannot be trained on this sample.** With merges confined
to words (ADR-0005), 5 MB of TinyStories supports 13,887 merges; past that
every word in the training text is already a single token, and ADR-0003
raises rather than padding. The ceiling itself (vocab 14,143) was measured as
an extra row and gains only 0.7% over 8,192. It is excluded from the knee
computation because it is a property of the sample size, not a grid point.

**Training time is not a discriminator.** Median wall-clock was flat across
sizes within a run (2.44–2.83 s in the recorded run), because pre-tokenizing
5 MB dominates and the merges themselves are cheap. An earlier identical run
measured 3.47–4.03 s, so between-session variance (~40%) is larger than any
difference between sizes.

### Robustness
**The knee is not robust.** An earlier draft of this ADR said it "did not
move" under perturbation; that was wrong, because it reported only the
perturbations that happened to agree. All variants below are recomputed from
saved sweep outputs (no re-run). *Margin* is the winner's distance above the
chord minus the runner-up's.

Recorded sweep (5 MB, `results/vocab_sweep.csv`):

| Grid | Knee, log2 x (the method) | Margin | Knee, linear x |
|---|---:|---:|---:|
| 512 – 8,192 (as recorded) | 2,048 | 0.055 | 2,048 |
| drop 512 | 2,048 | 0.017 | 4,096 |
| drop 8,192 | 2,048 | 0.004 | 2,048 |
| drop 512 and 8,192 | 2,048 | (only candidate) | 2,048 |
| add ceiling (14,143) | 2,048 | 0.026 | 4,096 |
| drop 512, add ceiling | **4,096** | 0.044 | 4,096 |

Same method on the 2.5 MB and 10 MB training samples:

| Sample | Grid | Full grid | Drop 512 | Add ceiling | Drop 512 + ceiling |
|---|---|---:|---:|---:|---:|
| 2.5 MB | 512 – 8,192 | 2,048 (0.057) | 2,048 (0.028) | 2,048 (0.048) | **4,096** (0.010) |
| 10 MB | 512 – 16,384 | 2,048 (0.015) | **4,096** (0.061) | 2,048 (0.012) | **4,096** (0.066) |

(log2 x; margin in parentheses. At 10 MB the ceiling is above 16,384, so
16,384 is a grid point there.)

**Why it moves: endpoint leverage.** The chord runs from the first grid point
to the last, and both axes are normalised by those same two points. Every
other point's distance is measured against that line, so changing an endpoint
re-scores every candidate at once. The lowest grid entry has the most
leverage because the curve is steepest there: 512 → 1,024 is a +33% jump, so
starting the chord at 512 instead of 1,024 tilts it sharply and favours
earlier points. Drop 512 and the chord flattens, and 4,096 gains on 2,048;
extend the far end (the ceiling, or 16,384 at 10 MB) and it gains further.
The linear-x column shows the same method is also sensitive to the choice of
axis: on a linear vocabulary axis 4,096 wins in 9 of the 14 variants across
the three samples.

What *is* stable is the measurement, not the knee: bytes/token at 2,048 was
3.7263 / 3.7258 / 3.7314 at 2.5 / 5 / 10 MB, and the per-doubling gains
(+33%, +18%, +9%, +4.5%) repeat across all three samples. The ambiguity is
in which of two adjacent sizes the rule calls "the knee", 2,048 or 4,096.

The ceiling grows with the sample (11,036 → 13,887 → 17,181 merges from
2.5 → 5 → 10 MB) as rarer words appear.

## Rationale
### What the sweep settles, and what it does not
- **It rules out the ends.** Below 2,048 each doubling still buys a large
  gain (+33% from 512 to 1,024, +18% from 1,024 to 2,048). Above 4,096 the
  gains are small (+4.5% to 8,192, +0.7% to the ceiling) while embedding cost
  keeps doubling, and 16,384 and GPT-2's 50,257 cannot be trained from this
  sample at all. The original options table started at 4,096; the data puts
  the decision at 2,048 or 4,096.
- **It does not resolve 2,048 vs 4,096.** The knee rule this ADR planned to
  use is unstable (see Robustness): it picks 2,048 on the recorded grid with
  a log2 axis, but flips to 4,096 when an endpoint changes, and **on a linear
  vocabulary axis 4,096 wins most variants (9 of 14)**. A rule that changes
  its answer with the axis scale cannot carry the decision, so it does not.

### The tie-breaker: the corpus is token-limited
The model trains on TinyStories-train, which is fixed at 1,924,281,556 bytes
and cannot be extended. At a fixed number of bytes, a smaller vocabulary
yields more training tokens.

| | 2,048 | 4,096 |
|---|---:|---:|
| Held-out bytes/token (CSV) | 3.7258 | 4.0775 |
| Training tokens, file bytes / bytes-per-token | 516.5M | 471.9M |
| Training tokens, separator-adjusted\* | **510.5M** | **466.6M** |
| Share of ~2.0B Chinchilla-optimal tokens\*\* | 25.5% | 23.3% |

\* Matches how bytes/token was measured: `<|endoftext|>` lines (1.58% of
bytes in the valid file) are removed from the text and each counted as one
special-token id (~2.18M of them).
\*\* ~20 training tokens per parameter (Hoffmann et al., 2022) x 100M
parameters = ~2.0B.

Either vocabulary leaves the model at about a quarter of the
compute-optimal token count for one pass over the data. **2,048 buys ~9.4%
more training tokens (~44M) from the same fixed corpus.** In a data-limited
regime that is the scarce resource, so it decides the tie.

Two limits on this argument. The figures assume full-corpus bytes/token
matches the 5 MB-sample measurement (see Consequences). And a token at 2,048
carries fewer bytes of text than a token at 4,096; the extra tokens are more,
shorter prediction steps over the same text, not more text. The claim is
only that, measured in tokens (the unit Chinchilla-style budgets use), the
smaller vocabulary gets closer to the target.

### Secondary: the decision is cheap either way
2,048 and 4,096 cost 1.57% and 3.15% of the 100M budget for the input
embedding: the whole choice is worth ~1.6 points of parameters. Getting it
wrong in either direction is not expensive, which is why a secondary
consideration like token count is allowed to settle it.

If Phase 3 finds sequence length, or parameters, rather than training tokens
to be the binding constraint, supersede this ADR with that evidence, not by
drift.

## Consequences
- **Phase 3 parameter count:** input embedding 1.57M (+ special-token rows);
  output head tied (0 extra) or untied (+1.57M). Either way under 3.2% of the
  budget.
- **Sequence lengths:** at 3.73 bytes/token, a 1,024-token context covers
  ~3.8 KB of text. Stories in this sample average ~820 bytes
  (4,999,982 B / 6,101), i.e. ~220 tokens.
- **The ceiling, and possibly the knee, will move on the full corpus.** The
  13,887-merge ceiling is a property of the 5 MB sample, not of TinyStories.
  Robustness was tested over 2.5 – 10 MB, a 4x range. Phase 2 trains the
  tokenizer on TinyStories-train, 1.9 GB: ~385x the 5 MB sample and ~190x the
  largest sample tested. More text means more rare word types, so the ceiling
  will rise, 16,384 and larger will become trainable, and the grid's upper
  end will extend. Since the knee has already moved when an endpoint changed
  (Robustness), it may move again. The decision does not rest on the knee,
  but it does rest on the measured bytes/token at 2,048 and 4,096 and the
  token counts derived from them. Re-run the sweep against the tokenizer
  training corpus in Phase 2 and record the result here; do not assume 2,048
  carries over. If a later measurement favours 4,096, switching costs ~1.6
  points of budget; each further doubling doubles the step (8,192 would be
  +4.7 points over 2,048).
- **Embedding-table rounding** (e.g. padding rows to a multiple of 64 for
  kernel efficiency) is a Phase 3 implementation detail and does not change
  this decision.
- **Reproducible:** `python scripts/sweep_vocab.py` regenerates every number
  above except wall-clock times.
