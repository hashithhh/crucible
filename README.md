# Crucible

A ~25M-parameter language model trained from scratch: byte-level BPE
tokenizer, decoder-only transformer, training run, then ablations and a
scaling study.

The target was ~100M until ADR-0006 (2026-09-22) sized the model to the data.
TinyStories measured **432M unique tokens** after deduplication (the estimate
was 510M; 15% of stories are exact duplicates), which supports S25 at 26.2M
parameters.

**The code in this repo was written by Claude, not by hand.** Every commit
says so; see `CLAUDE.md` for what that changes.

## Status

**255 passed, 1 skipped** (`pytest`; the corpus-wide round-trip runs under
`--runslow`).

| Phase | What | State |
|---|---|---|
| 1 | Byte-level BPE tokenizer, 2,048 merges | done — [writeup](docs/phase1-writeup.md) |
| 2 | Transformer: RMSNorm, RoPE, KV cache, sampling | done; the from-memory rebuild check was withdrawn unrun |
| 3 | Train S25 on TinyStories | done — [writeup](docs/phase3-writeup.md) |
| 4 | Ablations on S25: LR, warmup, MoE, hybrid attention | done — [writeup](docs/phase4-writeup.md) |
| 5 | Scaling study vs Chinchilla | done — [writeup](docs/phase5-writeup.md) |

Phase 3 result: one epoch, 6,561 steps, final held-out loss **1.3383
nats/token (0.517 bits/byte)** against a bigram baseline of 3.6190. Checks and
their pre-registered bars are in [`results/LEDGER.md`](results/LEDGER.md):
C3 (held-out loss) **passed**; C4 (sample coherence) **passed, 9/10**, scored by
Claude at the author's request — see the ledger for why that is noted. MFU on the local card was
32.3%; the T4 measurement was deferred as out of scope.

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
| 0007 | Architecture constants | accepted — RoPE base 10k, RMSNorm eps 1e-6, init 0.02, GELU, pre-norm, no biases |
| 0008 | Training hyperparameters | accepted — 65,536-token batch, AdamW, peak LR 6e-4 cosine to 6e-5, bf16 / fp16 + loss scaler |
| 0009 | Phase 4 ablation protocol | accepted — 820-step runs, seed-gap noise floor, 4-expert top-1 MoE, alternating 128-window attention |
| 0010 | Phase 5 scaling study | accepted — S3/S7/S13 at ~20 tokens/param plus S25; predicted exponent 0.28–0.34, measured 0.74 |

## Rules

See `CLAUDE.md`.
