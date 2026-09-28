# Crucible Phase 5 — a four-point scaling study, and a Chinchilla prediction that didn't hold

**Status:** done. Repo: `github.com/hashithhh/crucible` · **Date:** 2026-09-28

Phase 5 asks how held-out loss changes with model size when each model gets Chinchilla's
compute-optimal share of data, about 20 tokens per parameter, and whether the shape matches
Chinchilla's prediction. Three new models were trained on a laptop GPU in under two hours; the
fourth point is the Phase 3 model.

**On how this was built:** the code was written by Claude, not by hand, and every commit says so.
The design and the prediction being tested (ADR-0010) were committed before any run.

---

## The runs

| Model | Params | Tokens | Tokens/param | Held-out loss | Time |
|---|---:|---:|---:|---:|---:|
| S3 | 3.7M | 73.5M | 20.0 | 2.1616 | 7.7 min |
| S7 | 8.0M | 160.7M | 20.0 | 1.6871 | 27.9 min |
| S13 | 14.9M | 299.0M | 20.0 | 1.4723 | 78.3 min |
| S25 | 26.2M | 430.0M | 16.4 | 1.3383 | Phase 3 |

Same training recipe for every size (ADR-0008), full held-out split, one run per size. Phase 4
measured seed noise on this setup at 0.0033, far below any gap here.

**Loss falls with every step up in size, and by less each time** — 0.47, then 0.21, then 0.13 —
which is the shape any scaling law predicts.

## The prediction

Chinchilla's law is `L = E + A/N^0.34 + B/D^0.28`. Its absolute constants belong to its own data
and tokenizer, so they aren't comparable here. What does carry over is the *shape*: with data
growing in step with size (D = 20N), the law behaves like `L = E + K·N^(−γ)` with γ between 0.28
and 0.34. ADR-0010 recorded that band as the prediction before any run.

**The fit gives γ = 0.74, with a floor E = 1.09 nats. The prediction is not confirmed.**

## Is 0.74 real?

Four points and three fitted parameters leave one degree of freedom, and the floor and exponent
trade off against each other, so this was checked before being believed:

| Floor E held fixed at | Fitted γ | RMS error |
|---:|---:|---:|
| 0.0 | 0.24 | 0.043 |
| 0.5 | 0.35 | 0.035 |
| 0.8 | 0.47 | 0.024 |
| 1.0 | 0.63 | 0.011 |
| **1.09 (best fit)** | **0.74** | **0.002** |

- **Chinchilla's band is reachable only with a floor at or below ~0.5 nats**, and those curves
  miss the data 15–18× worse than the best fit.
- **The under-trained S25 point isn't responsible.** Fitting only the three models trained at
  exactly 20:1 gives γ = 0.79, steeper still.

So the data genuinely prefer a curve that bends sharply toward a floor near 1.1 nats. The most
likely reason is the corpus. TinyStories is deliberately narrow — simple vocabulary, formulaic
stories — so there is less left to learn, and models approach its irreducible loss quickly. The
curve then drops steeply and flattens early. Chinchilla's exponents were fitted on web text with
models from 70M to over 16B parameters, where the floor is far away. Its prediction not
transferring to a tiny model on a toy corpus is an informative result, not a broken one.

What this does *not* show: that Chinchilla is wrong, or that 0.74 is the "true" exponent for
TinyStories. One degree of freedom is too little for either claim. The robust findings are the
qualitative ones and the fact that no reasonable floor puts γ inside the predicted band.

## The honest part

- **One degree of freedom.** The exponent is fragile; its sensitivity table is above so nobody has
  to take 0.74 on trust.
- **One learning rate for all sizes.** Smaller models usually tolerate a higher learning rate, so
  S3 and S7 may be slightly handicapped — which would, if anything, *steepen* the curve, in the
  direction observed.
- **S25 at 16.4 tokens/param**, capped by the corpus. Checked above: removing it doesn't help.
- **Scope was cut on purpose:** no seeds per size, no learning-rate tuning per size, no sizes above
  S25. This is a portfolio-sized study.
- **The code is Claude's.** "Ran a small scaling study with AI assistance" is true.

## Reproduce

```
python scripts/scale.py            # trains whatever has not finished; resumable
python scripts/scale.py --summary  # refits and rewrites results/phase5_scaling.json
```

Design: `docs/adr/0010-scaling-study.md`. Results, fit and sensitivity: `results/phase5_scaling.json`.
