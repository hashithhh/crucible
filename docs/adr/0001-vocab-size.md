# ADR-0001 — Tokenizer vocabulary size

- **Status:** proposed — BLOCKED on measurement
- **Date:** 2026-09-16

## Context
Vocabulary size is not a tokenizer-local choice. The embedding table is
`vocab_size x d_model` parameters, and at `d_model=768` a GPT-2-sized 50,257
vocab costs ~38M parameters — roughly 38% of the ~100M budget — on a corpus
(TinyStories) built around a deliberately small vocabulary.

This decision propagates into Phase 3's parameter count and Phase 5's
scaling study. Choosing it by default pollutes both.

## Options considered
| Option | Embedding params @ d_model=768 | Note |
|---|---|---|
| 4,096 | 3.1M | |
| 8,192 | 6.3M | |
| 16,384 | 12.6M | |
| 50,257 (GPT-2) | 38.6M | 38% of budget |

## Decision
PENDING. To be set from `scripts/sweep_vocab.py`: pick the knee of the
bytes-per-token curve, not the largest vocabulary that fits.

## Rationale
To be filled with measured bytes/token per vocab size, and embedding
parameter share at each.

## Consequences
To be filled.
