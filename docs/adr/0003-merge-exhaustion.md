# ADR-0003 — Behaviour when merges are exhausted

- **Status:** proposed — BLOCKED on your decision
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
PENDING.

## Rationale
Note the interaction with ADR-0001: padding costs `d_model` parameters per
unused token in the embedding table, which is the exact waste ADR-0001 exists
to avoid.

## Consequences
Pin `test_merge_exhaustion_behaviour` to the chosen behaviour. Amend the
`train` docstring to state it as settled rather than open.
