# Crucible — Phase 1: BPE Tokenizer

Rung 1 of the Crucible ladder: a ~100M-parameter LLM trained from scratch.
This phase is a byte-level BPE tokenizer, standard library only.

## Status

Phase 1 of 5 — **102/102 green** as of 2026-09-17. Tokenizer implementation
written by Claude, not hand-written; see `CLAUDE.md` for what that changes.

Vocabulary size is decided: **2,048** (3.73 bytes/token held-out, 1.57M
embedding parameters at d_model=768). See ADR-0001 and
`results/vocab_sweep.png`; regenerate with `python scripts/sweep_vocab.py`.

Remaining in Phase 1: the `minbpe`/`tiktoken` reading pass, and the writeup.

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

## Rules

See `CLAUDE.md`.
