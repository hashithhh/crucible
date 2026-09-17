# ADR-0002 — Merge tie-break rule

- **Status:** accepted
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
**Highest count; on a tie, the lexicographically smallest pair.**

Implemented as `max(stats, key=lambda p: (stats[p], -p[0], -p[1]))`.

## Rationale
The pair itself provides a total order, so the rule never consults dict or
set iteration order and is stable across runs, Python versions and platforms.
"First encountered in a scan" would also be deterministic but only as long as
the scan order is never refactored — it makes determinism a property of the
traversal rather than of the rule.

This diverges from `minbpe`, which relies on `max()` over a counts dict and
therefore inherits insertion order on ties. That is a real difference to
record in Phase 1.5, not a bug in either.

## Consequences
`test_tie_break_rule_is_pinned` asserts it against TIE_CORPUS: (65,66)="AB"
and (67,68)="CD" both occur 50 times, "AB" wins id 256. Changing the rule
breaks that test loudly, which is the point.
