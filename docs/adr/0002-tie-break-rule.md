# ADR-0002 — Merge tie-break rule

- **Status:** proposed — BLOCKED on your decision
- **Date:** 2026-09-17

## Context
When two byte pairs occur the same number of times, something must decide
which is merged first. If that something is dict or set iteration order, the
tokenizer is not deterministic across runs or Python versions, and every
downstream artifact — the trained model, the ablation table, the scaling
study — becomes unreproducible.

`tests/conftest.py::TIE_CORPUS` forces the case: two pairs with exactly equal
counts.

## Options considered
| Option | Deterministic | Note |
|---|---|---|
| First encountered in a stable left-to-right scan | yes | depends on scan order being defined |
| Lexicographically smallest pair | yes | total order, no scan dependency |
| Lowest token id | yes | cheap |
| Whatever `max()` returns | **no** | depends on insertion order |

## Decision
PENDING.

## Rationale
To be filled.

## Consequences
Pin `test_tie_break_rule_is_pinned` to the chosen rule once decided.
