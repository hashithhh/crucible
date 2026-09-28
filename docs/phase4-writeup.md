# Crucible Phase 4 — four ablations on a 26M model, in 2.2 GPU-hours

**Status:** done. Repo: `github.com/hashithhh/crucible` · **Date:** 2026-09-28

Phase 4 asks what four design choices in the Phase 3 model are actually worth: the learning
rate, the warmup, dense versus mixture-of-experts, and full versus hybrid attention. It was scoped
deliberately small — one short run per question on a 6 GB laptop GPU — and this page is the result.

**On how this was built:** the code was written by Claude, not by hand, and every commit says so.
The protocol (ADR-0009) was written and committed before any run started.

---

## The protocol

- **Every run is 820 steps** — one eighth of the Phase 3 run, 53.7M tokens, about 21 minutes.
  Same config, same seed, same data order as the baseline; only the thing being tested differs.
- **The metric** is held-out loss at step 820, on the full 2.18M-token validation split.
- **The noise rule, fixed before any run:** the baseline was trained twice with different seeds.
  Their gap is the noise floor, and any variant closer to the baseline than that counts as **no
  difference**.

The two baselines landed at 1.8552 and 1.8585, so **the noise floor is 0.0033** — small enough
that every difference below is larger than seed noise.

## Results

| Variant | Held-out loss | vs baseline | Verdict | Throughput | Wall-clock |
|---|---:|---:|---|---:|---:|
| **Baseline** (Phase 3 config) | **1.8552** | — | — | 43.5k tok/s | 21.4 min |
| Learning rate 2e-4 (vs 6e-4) | 2.3268 | +0.472 | worse | 43.6k | 21.0 min |
| No warmup (vs 200 steps) | 2.0478 | +0.193 | worse | 43.5k | 21.0 min |
| **MoE, 4 experts, top-1** | **1.7721** | **−0.083** | **better** | **32.9k** | **28.7 min** |
| Hybrid attention (window 128) | 1.8484 | −0.007 | better, marginally | 43.6k | 21.3 min |

Six runs, 2.24 GPU-hours, none interrupted.

## What each one says

**Learning rate: the risky choice was the right one.** ADR-0008 flagged 6e-4 as the constant most
likely to be wrong, with ~2e-4 as the argued alternative. At 2e-4 the model is 0.47 worse. The
caveat is built in: a lower learning rate always starts slower, and a short run rewards whatever
learns fastest, so the gap would narrow over a full epoch. It would have to close entirely to
change the conclusion, and nothing here suggests it would.

**Warmup: it matters.** Removing the 200-step warmup costs 0.19. Without warmup, the first
optimizer steps are taken at a full learning rate while Adam's estimates still rest on a handful
of batches, and the run never makes up the damage within 820 steps. ADR-0009 predicted in advance
that a short run overstates warmup's effect, and that caveat stands.

**MoE: better per step, at a price.** Four experts with top-1 routing give each token exactly the
same compute as the dense model, so the 0.083 improvement is what extra *capacity* buys at equal
FLOPs — 76.6M parameters instead of 26.2M. It isn't free: the experts run as separate small matrix
multiplies, so throughput drops 24% and the run took 1.35× as long. Whether a dense model given
that extra 35% of wall-clock would catch up is exactly the question a single short run can't
answer. The experts were genuinely used: the balancing loss stayed between 1.1× and 1.5× its
perfectly-balanced value, far from the 4× that a collapse onto a single expert would give.

**Hybrid attention: no measurable cost, and no speed-up either.** Alternating full-context layers
with layers that see only the last 128 tokens came out 0.007 *better* — about twice the noise
floor, which clears the pre-registered rule but is too small to lean on. The honest reading is
"no measurable cost". It also ran at exactly the baseline's speed, which contradicts ADR-0009's
"less attention compute": this implementation masks the full attention matrix rather than skipping
the hidden part, so no compute is saved. Getting the speed-up would need a kernel that exploits
the window. Here, hybrid attention is a result about *quality*, not efficiency.

## The honest part

- **One seed pair is one sample of the noise.** Results clearly above the floor (learning rate,
  warmup, MoE) are solid for this setup; the hybrid result is at the edge and is reported as
  marginal.
- **Short horizon.** Every number is "at 1/8 of an epoch". It ranks how fast variants learn, not
  where they would finish.
- **Scope was cut on purpose** to keep this a portfolio project: no extra seeds per variant, no
  full-length confirmations, no sweeps. Each of those would be a new decision, not a gap in this one.
- **The code is Claude's.** "Ran ablations on a small language model with AI assistance" is true.

## Reproduce

```
python scripts/ablate.py            # runs whatever has not finished; resumable
python scripts/ablate.py --summary  # rebuilds results/phase4_ablations.json
```

Per-run logs and results are in `results/phase4/`; the protocol is
`docs/adr/0009-ablation-protocol.md`.
