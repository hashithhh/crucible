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
