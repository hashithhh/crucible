# C2 reconstruction — non-qualifying Codex simulation

**Date:** 2026-09-26  
**Candidate:** `results/c2_rebuild.py`  
**Seal:** `d9f382ed3ea02cd7112edcf1f0a89d59996f1e15`  
**Score:** 21 / 22 (Group A 10 / 10; Group B 11 / 12; no shim)

`python scripts/score_c2.py results/c2_rebuild.py` ran after the candidate was
sealed and reported a pass against the registered threshold of 14. The 21
applicable tests passed. `test_runs_on_gpu_and_matches_cpu` was skipped because
the scoring machine had no CUDA device, rather than failing.

## Validity

This is **not a C2/G1 result** and does not close Phase 2. The candidate was
written by Codex at Hashith's explicit request, and Codex consulted ADR-0007,
`scripts/score_c2.py`, and `tests/test_model.py` before writing it. It was also
not a timed, human, from-memory sitting. Those conditions violate C2's
no-reference and no-model-assistance procedure, so 21 / 22 is only an
implementation compatibility check.

The original human no-reference rebuild remains pending. Its eventual report
must be separate from this file and must carry its actual interval and any
exposure recorded before the attempt.
