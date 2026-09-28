# ADR-0009 — Phase 4 ablation protocol

- **Status:** accepted
- **Date:** 2026-09-28
- **Applies to:** Phase 4 (CLAUDE.md). Base config is ADR-0008; model is S25
  (ADR-0006).
- **Scope:** set by Hashith on 2026-09-28 — a resume project, "don't overcook".
  Everything below is the smallest version that still answers each question.

## Protocol

| Constant | Value | Why |
|---|---:|---|
| `steps` | 820 | 1/8 of ADR-0008's 6,561; ~21 min locally. The cosine lands at 820. |
| Tokens per run | 53.7M | `820 × 65,536` |
| Everything else | ADR-0008 | batch, betas, decay, clip, precision (bf16) |
| Seeds | 1337, and 1338 for the noise run | 1337 is ADR-0008's; 1338 only needs to differ |
| Metric | held-out loss at step 820 | full val split, `evaluate()` in `scripts/train.py` |

Every variant shares the baseline's seed, so the two runs see the same data in
the same order from the same initial weights. What differs is only the thing
being ablated.

**The noise rule, fixed before any run.** The baseline is trained twice, with
seeds 1337 and 1338. Their gap, `|a − b|`, is the noise floor. A variant whose
difference from the baseline is smaller than that is reported as **no
difference**, whatever its sign. This is weak by design: one pair of seeds is a
single sample of the noise, so results above the floor are *indicative*, not
significant, and the writeup will say so.

**What 820 steps can and cannot show.** A short run ranks variants by how fast
they learn early. It can't show where each would end after a full epoch;
warmup's effect in particular is front-loaded, so it is overstated here
relative to a long run. Differences are reported as "at 1/8 epoch".

## Variants

| Run | Change from baseline | Why this value |
|---|---|---|
| 4.2 LR | peak 2e-4, floor 2e-5 | ADR-0008 named 6e-4 its riskiest constant and ~2e-4 the argued alternative (square-root batch scaling). The floor stays at peak/10, ADR-0008's convention. |
| 4.3 Warmup | 0 steps | Whether warmup matters at all. Not a sweep. |
| 4.4 MoE | each MLP → 4 experts, top-1 routing | Below. |
| 4.5 Hybrid attention | odd layers use a sliding window of 128 | Below. |

**MoE: 4 experts, top-1.** ADR-0006 §2 sized 8 experts with top-2 routing and
noted that it isn't compute-matched to dense: 43.0M active parameters against
26.2M. It left the fix to this ADR. **Top-1 routing** makes each token pass
through exactly one full-width expert, so active parameters and FLOPs per token
equal the dense model's (26.2M). That turns it into a fair comparison at equal
compute and unequal capacity. **Four experts rather than eight** because four
answers the same question at 76.6M total parameters (26,223,616 + 3 × 8 layers ×
2,097,152 + a 16,384-parameter router); eight would be ~143M and too tight for
6 GB.

- Router: one linear layer, `d_model → 4`, softmax. The chosen expert's output
  is scaled by its router probability so the router receives a gradient.
- Load balancing: the auxiliary loss from Fedus et al. (Switch Transformer),
  coefficient **0.01**, their value. Without it top-1 routing tends to collapse
  onto one expert, which would quietly turn the MoE into a dense model with
  dead weight.
- No capacity limit and no token dropping. Every token is processed by its
  expert. That's simpler and loses nothing at this size; capacity limits are a
  throughput optimisation for many-device training that this project doesn't
  do.

**Hybrid attention: alternate full and sliding-window layers.** Layers 1, 3, 5
and 7 (zero-indexed) attend only to the previous 128 positions; layers 0, 2, 4
and 6 stay full-context. The alternating local/global pattern is the one Gemma 2
uses. The window of **128 is a quarter of the 512-token context**, short enough
that the local layers genuinely can't see across a whole story (~220 tokens on
average, ADR-0001) and long enough to cover several sentences. Parameters are
unchanged. The mask is the only difference, which is what makes it an
ablation.

## Consequences

- Six runs, about 2–2.5 GPU-hours. The MoE run is slower per token than dense
  despite equal FLOPs, because its experts run as separate small matmuls.
  Wall-clock is reported, not hidden.
- Two pieces of new model code: an MoE MLP and a windowed mask. Each gets tests
  only for the way it could be silently wrong: MoE routing (every token goes to
  exactly one expert, and gradients reach the router) and the window (no
  attention beyond 128 positions, and still causal).
- **Not in this ADR:** more seeds, full-epoch confirmations, LR or window
  sweeps. Any of those is a new decision, not an extension of this one.

## Sources

- ADR-0006 §2: the MoE sizing and the compute-matching problem it left open.
- ADR-0008: the base configuration and the 2e-4 alternative.
- Fedus, Zoph and Shazeer, *Switch Transformers* (2021): top-1 routing and the
  load-balancing loss with coefficient 0.01.
- Gemma Team, *Gemma 2* (2024): alternating local sliding-window and global
  attention layers.
