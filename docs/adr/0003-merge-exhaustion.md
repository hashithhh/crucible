# ADR-0003 — Behaviour when merges are exhausted

- **Status:** accepted
- **Date:** 2026-09-17

## Context
A corpus supports only a finite number of useful merges. `ASCII_CORPUS` has
24 distinct byte pairs and collapses to a single token after roughly 90
merges; after that there are no pairs left to count. Asking it for a
vocabulary of 600 asks for merges that do not exist.

This is not hypothetical — the original compression test in this repo asked
for 344 merges on that corpus and was therefore unsatisfiable. It was caught
before implementation, not after.

The contract says `self.vocab_size == vocab_size` after training. That
promise and merge exhaustion are in direct tension.

## Options considered
| Option | vocab_size promise | Note |
|---|---|---|
| Raise ValueError | held | loudest; forces the caller to pick a real size |
| Stop early, expose smaller vocab_size | broken | silent; downstream code may assume the requested size |
| Pad with unused placeholder tokens | held | wastes embedding parameters — see ADR-0001 |

## Decision
**Raise `ValueError`.** The message names how many merges were achieved and
what was requested.

## Rationale
Stopping early silently breaks `self.vocab_size == vocab_size`, and anything
downstream that sized an embedding table from the requested number would be
wrong without ever being told. Padding keeps the promise but spends
`d_model` parameters per unused token — the exact waste ADR-0001 exists to
prevent. Raising is the only option that keeps the contract and stays loud.

Measured ceilings. They depend on merge granularity, so they changed when
merges were confined to pre-tokenized chunks (ADR-0005, 2026-09-17):

| Corpus | Raw stream: max vocab (merges) | cl100k chunks: max vocab (merges) |
|---|---|---|
| ASCII_CORPUS | 304 (48) | 275 (19) |
| UNICODE_CORPUS | 333 (77) | 316 (60) |
| COMPRESSION_CORPUS | >2000 (>1744) | 604 (348) |

## Consequences
The default training fixture moved off ASCII_CORPUS, which could not reach
VOCAB_SIZE=320. Under ADR-0005, UNICODE_CORPUS cannot reach it either; its
fixture trains at UNICODE_VOCAB_SIZE=300. ASCII_CORPUS is now used only where its thinness is the
point: the 256 case and `test_merge_exhaustion_behaviour`.

Practically this means vocabulary size is bounded by corpus richness, which
is worth remembering when ADR-0001 picks a real vocab size against
TinyStories.
