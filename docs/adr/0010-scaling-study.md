# ADR-0010 — Phase 5 scaling study

- **Status:** accepted
- **Date:** 2026-09-28
- **Applies to:** Phase 5. Sizes from ADR-0006 §3; training config from ADR-0008.
- **Scope:** the 2026-09-28 direction applies — a resume project, "don't overcook".

## Decision

Train the ladder at **Chinchilla's compute-optimal ratio, ~20 tokens per
parameter**, and reuse the Phase 3 S25 run as the fourth point.

| Size | d / layers / heads | Params | Tokens | Steps | Local time (measured rate) |
|---|---|---:|---:|---:|---:|
| S3 | 256 / 4 / 4 | 3,672,576 | 73.5M | 1,121 | ~7.5 min |
| S7 | 320 / 6 / 5 | 8,032,640 | 160.7M | 2,451 | ~28 min |
| S13 | 384 / 8 / 6 | 14,949,120 | 299.0M | 4,562 | ~77 min |
| S25 | 512 / 8 / 8 | 26,223,616 | 430.0M | 6,561 | Phase 3, not re-run |

Steps are `round(20 × params / 65,536)`. Everything else is ADR-0008, including
the 6e-4 peak learning rate and the 200-step warmup, with the cosine landing at
each run's own final step. The metric is held-out loss at the final step on the
full validation split.

**This revises ADR-0006 §3**, which planned one full epoch for every size
(139 down to 19.5 tokens per parameter, ~3.8 hours locally). One epoch each
measures loss against size at *fixed data*, which is not what Chinchilla
claims; 20 tokens per parameter is its compute-optimal frontier, and it is also
half the cost.

## What is compared with Chinchilla, and what isn't

Chinchilla's fitted law is `L(N, D) = E + A/N^α + B/D^β` with α = 0.34 and
β = 0.28 (Hoffmann et al., approach 3). Its absolute constants belong to its own
data and tokenizer, so **absolute losses are not compared**. Along the 20:1
frontier the law reduces to `L ≈ E + K·N^(−γ)` with an effective exponent
between β and α, so **the prediction checked here is that γ lands in
[0.28, 0.34]**.

The fit is `L = E + K·N^(−γ)` over the four points: three parameters from four
points, one degree of freedom. That's weak, and it's stated as weak: γ is
reported with that caveat, and the robust claims are the qualitative ones —
loss falls with size, and by less at each step up.

## Known confounds, accepted

- **S25 sits at 16.5 tokens per parameter**, not 20: the corpus caps it
  (ADR-0006 amendment). It is slightly undertrained relative to the others,
  which can only flatten the top of the curve.
- **One learning rate for all sizes.** Smaller models usually tolerate larger
  ones, so 6e-4 may slightly handicap S3 and S7. Tuning per size is a sweep,
  and sweeps are out of scope.
- **One run per size**, no seeds. Phase 4 measured seed noise at 0.0033 on this
  setup, far below the gaps expected between sizes.

## Sources

- Hoffmann et al., *Training Compute-Optimal Large Language Models*
  (Chinchilla, 2022): the ~20 tokens/param ratio and the approach-3 exponents.
- ADR-0006 §3: the ladder shapes. ADR-0008: the training config.
- `results/c3_result.json`: the S25 point (1.3383).
