# Crucible — operating rules for this repo

Crucible is a ~25M-parameter LLM trained from scratch (ADR-0006; the target
was ~100M until the data budget was measured). Phase 1 is the byte-level BPE
tokenizer.

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
- ADR for every constant. No magic numbers. ADRs 0001–0007 are
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
   Precision. Verified on the local card: bf16, no scaler. **The fp16 path is
   unexercised** — no Turing GPU here — so it is written and reasoned, not
   demonstrated.
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
7. **3.6 Train S25 to convergence** (ADR-0006). **C3 and C4 registered in
   `results/LEDGER.md` before the run starts**, per the ledger's rule 1. Their
   targets are not recorded anywhere in this repo; see the note below.
8. **3.7 Sample from it.** Coherent text, or the run failed — this is the
   whole point of Phase 2's sampler and the first evidence the pipeline works
   rather than merely runs.

Carried over because nothing else covers them:

9. ~~**ADR-0008 — training hyperparameters**~~ **done 2026-09-26** —
   `docs/adr/0008-training-hyperparameters.md`. Borrowed values say where
   from; `peak_lr` 6e-4 is the one carrying real risk and is the only
   constant with a pre-stated revision trigger.
10. **Tests, public behaviour only.** Exact resume (checkpoint, resume,
    identical loss), shard-reader boundaries, LR schedule shape. Same standard
    as Phases 1 and 2.
11. **MFU measured on a T4, `model_budget.py` re-run with it.** ADR-0006's T4
    hours assume 25-40% MFU and it records this as the first thing Phase 3
    owes. The local 4050 figure (~40k tokens/s, ~7.3 TFLOP/s) suggests that
    band is optimistic.
12. **A Phase 3 writeup** with the numbers, stating how the code was produced.

**C3 and C4 are undefined.** The checklist says "record C3, C4" but no target
for either exists in this repo or in any vault on this machine — the same gap
C1 and C2 hit. Candidates, given 3.6 and 3.7: **C3 = held-out loss** against a
pre-registered value, **C4 = sample coherence** under a stated judgement.
Both must be written down before the run, or they are not checks.

**Not in Phase 3:** the ablations and the scaling ladder — Phases 4 and 5,
sized in ADR-0006 §2 and §3.
