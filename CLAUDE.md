# Crucible — operating rules for this repo

Crucible is a learn-by-building project: a ~100M-parameter LLM trained from
scratch. **Phase 1 is the BPE tokenizer, and Hashith writes it by hand.**
That is the entire point of the phase.

## Hard constraints — these override default helpfulness

- Do **not** write, sketch, pseudocode, or describe the BPE algorithm's
  internals: not the pair-counting loop, not the merge step, not `encode`,
  not `decode`, not the regex split pattern.
- Do **not** reference, quote, paraphrase, or reconstruct `karpathy/minbpe`,
  `tiktoken`, HuggingFace `tokenizers`, or any existing BPE implementation.
  These are memorised; reproducing one makes the Phase 1 comparison gate
  meaningless.
- Do **not** fill in a stub, complete an empty function, or "fix" the
  algorithm even if it looks wrong.
- If Hashith is stuck, give a **socratic hint only**: name the property being
  violated or the case not handled. Never the fix.
- If a request would require breaking these rules, say so and stop.

## What you ARE free to write

Tests, harnesses, CLI scaffolding, file I/O, plotting, benchmarking,
packaging, docs, ADRs, CI.

## Implementation rules

- `tokenizer.py` depends on the **standard library only**. `pytest` and
  `hypothesis` are test-only dependencies.
- Tests touch the **public interface only** (`train`/`encode`/`decode`/
  `save`/`load`). Internal structure is deliberately unconstrained.
- ADR for every constant. No magic numbers.
- Apache-2.0.

## Phase 1 is done when

1. The full suite is green.
2. `docs/adr/0001-vocab-size.md` is decided from measured data, not a default.
3. The bytes-per-token vs vocab-size figure exists.
4. The differential check against `minbpe` has been run **once, afterwards**.
5. A public writeup with those numbers is published.
