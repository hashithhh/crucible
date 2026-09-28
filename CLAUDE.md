# Crucible — operating rules for this repo

Crucible is a ~25M-parameter LLM trained from scratch (ADR-0006; the target
was ~100M until the data budget was measured). **All five phases are complete
as of 2026-09-28**: tokenizer, transformer, training run, ablations, scaling
study.

## Authorship — read this first

**The Phase 1 tokenizer was written by Claude, not by hand.** The original
plan was for Hashith to hand-write it, with an independent comparison against
`karpathy/minbpe` as the check. That plan was abandoned on 2026-09-17 at his
explicit instruction, after the implementation had not been started.

This file previously carried a "do not write the BPE internals" constraint.
That constraint is no longer in force and has been removed rather than left
in place to be quietly violated.

What this changes, concretely:

- **The `minbpe` comparison gate is dropped from the definition of done.**
  Claude has `minbpe` memorised; diffing Claude's implementation against it
  measures nothing. Phase 1.5 becomes a reading exercise — read `minbpe`,
  write down the differences — not a verification step.
- **Describe this project accurately.** "Built a BPE tokenizer with AI
  assistance" is true and unremarkable. "Hand-wrote a BPE tokenizer from
  scratch" is not true of this repo. The git log is the record; keep commit
  authorship honest and this stays a non-issue.

## Current rules

- `tokenizer.py` depends on the **standard library only**. `pytest` and
  `hypothesis` are test-only dependencies.
- Tests touch the **public interface only** (`vocab_size`, `train`, `encode`,
  `decode`, `register_special_tokens`, `save`, `load`). Internal structure is
  unconstrained.
- ADR for every constant. No magic numbers. ADRs 0001–0010 are
  accepted.
- Apache-2.0.
- Every commit message states who wrote the code in it.

## Phase 1 definition of done

1. ~~Suite green~~ **done 2026-09-17** — 51/51.
2. ~~`docs/adr/0001-vocab-size.md` decided from measured data~~ **done
   2026-09-17** — 2,048 (3.73 bytes/token; ~510M training tokens).
3. ~~The bytes-per-token vs vocab-size figure exists~~ **done 2026-09-17** —
   `results/vocab_sweep.png`.
4. ~~Differential check against `minbpe`~~ **dropped** — see Authorship.
   Replaced by: read `minbpe` and `tiktoken`, record the differences.
   minbpe **done 2026-09-22** — `docs/minbpe-differences.md` (commit 1acefe8).
   tiktoken compared by measurement (C1, `results/c1_tiktoken.md`); its
   source has not been read.
5. ~~A public writeup with those numbers, stating how the code was produced~~
   **done 2026-09-25** — `docs/phase1-writeup.md`; the repo is public.

## Phase 2 definition of done

1. ~~Transformer: attention, MLP, residuals, RMSNorm~~ **done 2026-09-24**.
2. ~~RoPE~~ **done 2026-09-24**.
3. ~~KV cache~~ **done 2026-09-24**.
4. ~~Sampling: greedy, temperature, top-k, top-p~~ **done 2026-09-24**.
5. ~~Shape-and-gradient tests~~ **done 2026-09-23** — 50 tests, all green.
6. ~~G1 GATE: rebuild from memory, record C2~~ **WITHDRAWN 2026-09-26,
   unrun** — `results/LEDGER.md`, C2. Registered (8ddbb64), amended to a
   two-day interval, then withdrawn without being attempted. **Nothing was
   measured.** Phase 2 closes with five of six items done and the sixth struck,
   not passed.

   This is the second verification step this project has dropped, after the
   minbpe comparison in Phase 1, and for the same reason both times: the code
   was Claude's, so the check that would have measured understanding was the
   part that became optional. Recorded here rather than quietly omitted.

   **Describe Phase 2 accordingly.** "Built a transformer with AI assistance"
   is true. "Rebuilt it from memory as a check" is not true of this repo.

## Phase 3 definition of done

Train S25 (26.2M params, ADR-0006) to convergence on the full corpus, and be
able to show the loss curve, the samples, and how both were produced.

1. ~~**3.0** Tokenizer trained on the full 1.9 GB corpus~~ **done 2026-09-25**
   — 3.7333 bytes/token held out, 0.2% off ADR-0001
   (`results/tokenizer_train.json`).
2. ~~**3.1 Data pipeline.** Dedup, shard, memmap, held-out split,
   round-trip verified, realised token count against ADR-0006~~ **done
   2026-09-26** — `data/tokens/meta.json`, `results/shard_verification.json`.
   432,175,881 tokens from 1,799,248 unique stories after removing 320,241
   exact duplicates (15.1%); 7 train shards and 1 val shard, uint16,
   memmap-able. Encoding ran at 1.0M tokens/s across 16 workers, 7.2 min
   against ~97 min single-threaded. **The measured corpus is 15.3% smaller
   than ADR-0006's 510.5M estimate, so S25 trains at 16.48 tokens/param, not
   19.5.** ADR-0006 is amended 2026-09-26: S25 holds, and what that costs is
   stated there rather than smoothed over.
3. ~~**3.2 Training loop.** AdamW, cosine schedule with warmup, gradient
   clipping~~ **done 2026-09-26** — `scripts/train.py`, `crucible/training.py`,
   ADR-0008. The schedule is a pure function with 20 tests on the properties
   ADR-0008 relies on, because a wrong schedule does not crash: it trains,
   looks plausible, and costs three hours.
4. ~~**3.3 Mixed precision,** with the loss scaler understood rather than
   copied~~ **done 2026-09-26** — `pick_precision` picks bf16 or fp16+scaler
   from the hardware; the paragraph on why the scaler exists is in ADR-0008,
   Precision. Verified on the local card: bf16, no scaler. **The fp16 + scaler
   path was exercised 2026-09-28** by forcing it on the local card
   (`--precision fp16`): 100 real S25 steps from the same seed track bf16 to
   within 0.0076 (`results/fp16_check.json`), and a GPU test runs it end to
   end. Not yet exercised: the scaler's overflow branch (no overflow occurred)
   and the path on actual Turing hardware.
5. ~~**3.4 Checkpointing and resume**, exact rather than approximate~~
   **done 2026-09-26** — `tests/test_train_resume.py` trains 8 steps
   uninterrupted, then 4 + resume, and asserts the losses match. The sampler's
   RNG state is in the checkpoint: without it a resumed run re-sees batches it
   already trained on while looking perfectly healthy, and a second test
   deliberately breaks it to prove the first one can fail. `--stop-at` pauses
   without moving the schedule's horizon.
6. ~~**3.5 Logging:** loss, LR, grad norm, tokens/sec, GPU memory~~ **done
   2026-09-26** — JSONL to `results/train_log.jsonl`, one record per logged
   step plus `run` / `eval` / `paused` / `done` lines, asserted by a test.
7. ~~**3.6 Train S25 to convergence**~~ **done 2026-09-28** — all 6,561
   steps, final held-out loss **1.3383** (0.517 bits/byte). **C3: pass**
   (bar 2.60, stretch 2.00 also met; `results/c3_result.json`). The run
   survived three stops for low memory and a 285-minute machine sleep,
   resuming exactly each time; 360 steps were recomputed. In hindsight C3's
   bars were easy — crossed by step 750 — and were not moved.
8. ~~**3.7 Sample from it**~~ **done 2026-09-28 — C4: pass, 9/10** (7/10 on
   the strictest reading), **scored by Claude at Hashith's request**, departing
   from the registration that named Hashith as judge; recorded in the ledger.
   Ten samples from the registered prompts, one each, no re-rolls, every
   call reasoned in `results/c4_samples.md`; only sample 10 fails outright.

Carried over because nothing else covers them:

9. ~~**ADR-0008 — training hyperparameters**~~ **done 2026-09-26** —
   `docs/adr/0008-training-hyperparameters.md`. Borrowed values say where
   from; `peak_lr` 6e-4 is the one carrying real risk and is the only
   constant with a pre-stated revision trigger. It did not fire.
10. ~~**Tests, public behaviour only**~~ **done** — exact resume (with a
    control that must fail), shard-reader boundaries, LR schedule shape, the
    C3/C4 ledger links, and MFU throughput under a stall and a resume.
11. **MFU — local half done; T4 half DEFERRED 2026-09-28.** Local RTX 4050: **32.3%**
    (`results/mfu.json`) against a *measured* dense-bf16 matmul ceiling of
    24.66 TFLOP/s, 43,662 tokens/s, inside the 25-40% band ADR-0006 assumed.
    The earlier note here that the band looked optimistic was wrong; it used
    no measured ceiling. **The T4 figure is still owed** — no Turing GPU on
    this machine. It is now one command in a Kaggle T4 notebook:
    `scripts/kaggle_t4.py` writes `results/t4_check.json`, then
    `python scripts/model_budget.py --t4-mfu <mfu_vs_datasheet>` is the
    re-run. The script was dry-run locally (31.9% MFU against the measured
    ceiling, agreeing with the 32.3% from the run log). **Deferred, not done:**
    Hashith scoped the project down on 2026-09-28 ("keep this a resume
    project, don't overcook"), and a Kaggle session is not worth it for a
    number the local measurement already bounds. Not measured, not claimed.
12. ~~**A Phase 3 writeup**~~ **done 2026-09-28** — `docs/phase3-writeup.md`,
    complete; no pending markers.

**Phase 3 closes 2026-09-28**: eleven items done, item 11's T4 half deferred.

**C3 and C4 were registered 2026-09-26** (`de8ae8a`), before the run, with
thresholds proposed by Claude and set without Hashith's review. Both entries
say so.

## Phase 4 definition of done

Four ablations against S25, sized to a 6 GB laptop and a resume: one short run
per variant, one shared baseline, one noise check. **Scope set by Hashith on
2026-09-28: "don't overcook or do over-engineering."** Not a research
programme; if a step starts growing, cut it.

**Protocol (ADR-0009, one page, written before any run):** every run is 820
steps (1/8 epoch, 53.7M tokens, ~21 min locally) on ADR-0008's config with the
cosine landing at step 820, same seed and data order as the baseline. The
metric is held-out loss at step 820 on the full val split. A difference
smaller than the baseline's seed-to-seed gap is reported as **no difference**.
That rule, fixed before the runs, is the pre-registration; no ledger entries.

1. ~~**4.0 ADR-0009**~~ **done 2026-09-28**, committed before any run.
2. ~~**4.1 Baseline, two seeds**~~ **done** — 1.8552 and 1.8585; noise
   floor **0.0033**.
3. ~~**4.2 Learning rate 2e-4**~~ **done** — 2.3268, +0.472, worse. ADR-0008's
   6e-4 confirmed at this horizon.
4. ~~**4.3 No warmup**~~ **done** — 2.0478, +0.193, worse.
5. ~~**4.4 MoE, 4 experts top-1**~~ **done** — 1.7721, −0.083, better at equal
   FLOPs per token, but 24% lower throughput (1.35× wall-clock). Routing
   stayed balanced (aux 1.1-1.5× ideal).
6. ~~**4.5 Hybrid attention**~~ **done** — 1.8484, −0.007: clears the noise
   rule by 2× but reported as marginal. **No speed-up**: the mask-based
   implementation computes full attention, contrary to ADR-0009's "less
   attention compute" (amended there).
7. ~~**4.6 Results**~~ **done** — `results/phase4_ablations.json`,
   `results/phase4/`, `docs/phase4-writeup.md`; 12 tests on the new code.

Budget: 6 runs, 2.24 GPU-hours actual, none interrupted (run as a detached
process after the session's memory reaper stopped every in-session attempt). **Not in Phase 4:** multiple seeds per
variant, full-epoch confirmations, LR sweeps. Phase 5 is the scaling ladder.

## Phase 5 definition of done

A four-point scaling study against Chinchilla, sized like Phase 4
(ADR-0010, written before any run).

1. ~~**5.0 ADR-0010**~~ **done 2026-09-28** — S3/S7/S13 at ~20 tokens/param,
   S25 reused from Phase 3; revises ADR-0006 §3's one-epoch-per-size plan.
2. ~~**5.1 Train the ladder**~~ **done 2026-09-28** — S3 2.1616, S7 1.6871,
   S13 1.4723, S25 1.3383; 1.9 GPU-hours, none interrupted.
3. ~~**5.2 Fit and compare**~~ **done** — `L = 1.090 + K·N^−0.742`. **The
   Chinchilla prediction (γ in [0.28, 0.34]) is NOT confirmed.** Checked for
   fragility: band-compatible fits need a floor ≤ ~0.5 and fit 15-18× worse,
   and dropping the under-trained S25 gives γ = 0.79. Loss falls with size
   with shrinking gains. `results/phase5_scaling.json`.
4. ~~**5.3 Writeup**~~ **done** — `docs/phase5-writeup.md`.

**Not in Phase 3:** the ablations and the scaling ladder — Phases 4 and 5,
sized in ADR-0006 §2 and §3.
