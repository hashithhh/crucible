# Crucible — Phase 1: BPE Tokenizer

Rung 1 of the Crucible ladder: a ~25M-parameter LLM trained from scratch.
This phase is a byte-level BPE tokenizer, standard library only.

The target was ~100M until ADR-0006 (2026-09-22): TinyStories supplies 510M
unique tokens, which at ~20 tokens/parameter supports ~25M, not 100M.

## Status

Phase 1 of 5 — **103/103 green** as of 2026-09-22, including the corpus-wide
round-trip over all 21,990 stories (`pytest --runslow`, 52 s; the default run
skips it and reports 102 passed, 1 skipped). Tokenizer implementation written
by Claude, not hand-written; see `CLAUDE.md` for what that changes.

Vocabulary size is decided: **2,048** (3.73 bytes/token held-out, 1.05M
embedding parameters at ADR-0006's d_model=512). See ADR-0001 and
`results/vocab_sweep.png`; regenerate with `python scripts/sweep_vocab.py`.

Compared against tiktoken by measurement (C1, `results/c1_tiktoken.md`) and
against minbpe by reading its source (`docs/minbpe-differences.md`).

Remaining in Phase 1: the writeup.

## Run the tests

```
pip install -e ".[dev]"
pytest
```

## Watch one test

```
.\scripts\watch.ps1
.\scripts\watch.ps1 tests/test_train.py::test_vocab_size_is_exact
```

## Decisions

| ADR | Subject | Status |
|---|---|---|
| 0001 | Vocabulary size | accepted — 2,048; sweep narrows to 2,048/4,096, token-limited corpus breaks the tie |
| 0002 | Merge tie-break rule | accepted — lexicographically smallest pair |
| 0003 | Merge exhaustion | accepted — raise ValueError |
| 0004 | Invalid UTF-8 on decode | accepted — errors="replace" |
| 0005 | Pre-tokenization pattern | accepted — GPT-4 (cl100k), translated to stdlib `re` |
| 0006 | Model size and data | accepted — ~25M (S25: d512, 8 layers, 8 heads) on TinyStories only |

## Rules

See `CLAUDE.md`.
