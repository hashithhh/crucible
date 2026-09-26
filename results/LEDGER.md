# Crucible results ledger

Checks that gate a phase, with the pass mark written down **before** the check
runs.

This file was created on **2026-09-26**, which is later than it should have
been. Both checks defined so far went looking for a pre-registered target and
did not find one: C1 recorded that the Obsidian vault on this machine has no
Crucible entry, and CLAUDE.md records the same for C2. A gate whose pass mark
is chosen after the result is visible is not a gate, so the pass marks now live
in the repository, under version control, where the git log dates them.

## The rule

1. **Register before running.** An entry states what is measured, the
   procedure, the pass mark, and the remedy on failure — and is committed
   before the check is run.
2. **A late entry is a backfill and says so.** A backfill records a number. It
   cannot pass or fail anything, because the threshold and the result were
   never separated in time.
3. **Thresholds carry a rationale.** CLAUDE.md requires an ADR for every
   constant; a pass mark is a constant. The reasoning goes in the entry rather
   than a separate ADR, because the entry *is* the decision record.
4. **The remedy must change what was measured.** "Repeat the phase" is only a
   remedy if repeating the phase acts on the thing that failed.

## Entries

| ID | Check | Registered | Run | Status |
|---|---|---|---|---|
| C1 | Tokenizer compression vs tiktoken | backfilled 2026-09-26 | 2026-09-22 | recorded, non-gating |
| C2 | G1 — rebuild `crucible/model.py` from memory | 2026-09-26, signed | 2026-09-26, **2-day interval** | pending |

---

# C1 — tokenizer compression vs tiktoken

**Backfill.** Full result: [`results/c1_tiktoken.md`](c1_tiktoken.md), raw
output in [`results/c1_tiktoken.json`](c1_tiktoken.json). Run 2026-09-22,
committed in d23d8cb.

- **Measured:** bytes/token and extra tokens against `cl100k_base` and
  `o200k_base` on the ADR-0001 held-out slice.
- **Target, as stated in the Phase 1.4 request:** within 10% of cl100k_base;
  kill threshold >30% worse.
- **Result:** +13.6% extra tokens vs cl100k_base. Misses the 10% target, well
  inside the kill threshold.

**Status: recorded, non-gating.** Two reasons, both already argued in the C1
document and neither softened here:

- The target was never pre-registered anywhere that could be produced, so per
  rule 2 this entry is a backfill and settles nothing.
- The check does not isolate what it appears to. C1 compares a 2,048-token
  vocabulary against pretrained vocabularies 49x and 98x larger; at the largest
  vocabulary the sample supports (14,143) the same code beats cl100k_base by
  1.4%. The gap is ADR-0001's vocabulary choice, which ADR-0001 made
  deliberately and for stated reasons. A kill here would have overruled a
  recorded decision rather than caught a defect.

If a compression gate is wanted later, re-register it at matched vocabulary
size. Do not read the number above as a pass or a fail.

---

# C2 — G1: rebuild the transformer from memory

**Status: registered 2026-09-26, before the attempt.** Drafted by Claude and
accepted by Hashith without amendment — see [Sign-off](#sign-off). How the
thresholds were set is itself part of the record: Claude proposed them, Hashith
did not change them. That is weaker than Hashith setting his own bar, and is
noted here rather than left to be inferred.

## What this measures

Whether the transformer's structure and its non-obvious properties can be
reproduced without reference, **two days** after exposure. The registered
interval was one week; see [Interval](#interval) for why it changed and what
that costs.

## What it does not measure

`crucible/model.py` was written by Claude, not by hand (CLAUDE.md, Authorship).
So C2 measures **recall of code read, not code written**, which is a weaker
claim than the one G1 was originally designed to support, and the number must
be reported as such wherever it appears. This is the same substitution that
removed the minbpe check in Phase 1; it is recorded here so the pattern stays
visible rather than being absorbed into a passing grade.

## Scope

`crucible/model.py` only — `ModelConfig`, `RMSNorm`, `apply_rope`, `KVCache`,
`CausalSelfAttention`, `MLP`, `Block`, `Transformer`. That is Phase 2.1–2.3,
324 lines.

`crucible/sampling.py` (Phase 2.4) is **out of scope**. Greedy / temperature /
top-k / top-p is a different kind of recall — a list of independent procedures
rather than one interlocking structure — and mixing the two would make a single
score uninterpretable. If it is wanted, register it separately as C3.

## Procedure

1. Blank file, single sitting, **2 hours** (proposed), no reference material:
   no `crucible/`, no `tests/`, no ADRs, no notes, no documentation, no model
   or search assistance. PyTorch's own API docs (`torch.nn`, `F.*` signatures)
   **are** allowed — the target is architecture, not API memorisation.
2. Save as `results/c2_rebuild.py` and **commit it unmodified** before running
   anything against it. The seal is what makes the score meaningful; a file
   edited after the first traceback measures debugging, not recall.
3. Then, and only then, score it:

   ```
   python scripts/score_c2.py results/c2_rebuild.py [--shim results/c2_shim.py]
   ```

   The script substitutes the rebuild for `crucible.model` so
   `tests/test_model.py` runs unmodified, and writes `results/c2_rebuild.json`.
   It refuses to run on a file that is untracked or has uncommitted changes, so
   the seal in step 2 is enforced rather than trusted. A rebuild whose names
   differ enough that the tests cannot import it scores 0 — that is what the
   shim below exists to rescue.

## Interval

G1 was registered at one week after the 2026-09-24 build, earliest 2026-10-01.
On 2026-09-26 Hashith elected to run it at **two days** instead, to close Phase
2 without waiting. Amended and committed before the attempt, not after.

**What this costs.** Two days is a materially easier test than seven; a good
result at this interval does not support the claim the seven-day version would
have. Wherever C2's number appears it carries the interval with it. The gate is
not thereby worthless — a two-day cold rebuild still separates recall from
recognition — but it is a weaker instrument than the one originally registered,
and the writeup should say two days rather than "the G1 gate".

**Recorded exposure.** During registration on 2026-09-26, in the session that
produced this file, Hashith saw: the names of the eight public symbols in
`crucible/model.py` (Scope, above), and all 22 test names in
`tests/test_model.py` (Measure, below). Neither reveals an implementation, but
both are more than a cold start, and the Group A / Group B split signposts which
properties matter. This is real contamination and is recorded rather than
discounted, per the clause below.

**Exposure before the attempt.** The procedure above bars reference material
*during* the rebuild and says nothing about the days before it, which leaves
the "one week later" bar doing no work: re-reading `model.py` on 2026-09-30
would make this a one-day test wearing a one-week label. Proposed clause, for
the sign-off: **no contact with `crucible/model.py`, `tests/test_model.py` or
ADR-0007 after 2026-09-26.** Incidental exposure through unrelated work is
possible in a repository this small; if it happens, record it in the result
rather than discounting it silently.

## Measure

Tests passed out of the **22** in `tests/test_model.py`, unedited. The tests
are already written and already green against the real file, so the instrument
exists and needs no judgement call at scoring time.

They fall into two groups, which is where the threshold comes from.

**Group A — reachable from a generic transformer (10).** Any competent
from-memory transformer should pass these; they say little about this
implementation.

`test_config_rejects_indivisible_head_split`,
`test_rmsnorm_is_scale_invariant_and_shape_preserving`,
`test_rmsnorm_survives_zeros_and_normalises_last_dim_only`,
`test_rmsnorm_has_one_parameter_vector`, `test_rope_preserves_shape_and_norm`,
`test_mlp_is_position_wise`, `test_block_with_zeroed_outputs_is_identity`,
`test_forward_shape_and_dtype`, `test_single_step_forward_works`,
`test_forward_is_deterministic`

**Group B — specific to what was read (12).** These are the properties a
plausible-but-wrong implementation breaks silently: relative-offset RoPE,
causality under both paths, cache/uncached agreement, gradient masking.

`test_s25_parameter_count_matches_adr_0006`,
`test_rope_at_position_zero_is_identity`,
`test_rope_dot_product_depends_only_on_offset`, `test_attention_is_causal`,
`test_model_is_causal`, `test_context_limit_and_id_range_are_enforced`,
`test_every_parameter_receives_gradient`,
`test_gradients_ignore_later_tokens`,
`test_cache_reports_length_and_rejects_overflow`,
`test_kv_cache_matches_full_forward`,
`test_cache_reset_gives_a_fresh_sequence`,
`test_runs_on_gpu_and_matches_cpu`

One caveat on Group B: `test_s25_parameter_count_matches_adr_0006` asserts
26,223,616 exactly, which is reciting ADR-0006's constants rather than
understanding the architecture. **Score it, but report it separately** — a
rebuild that gets everything else right and misremembers `vocab_size` as 2,048
instead of 2,049 has not misunderstood anything.

## Thresholds (proposed)

| Band | Tests passed | Meaning |
|---|---:|---|
| **Pass** | **≥ 14 / 22** | Phase 2 closes. |
| **Partial fail** | 7 – 13 | Re-read, re-attempt once. |
| **Hard fail** | < 7 | The material did not land at all. |

**Why 14.** Group A is 10 tests and is reachable without having read this
repository. A pass mark at or below 10 would be satisfied by generic knowledge
and would measure nothing about the week-old exposure. 14 requires all of
Group A plus 4 of the 12 properties that are specific to what was read — enough
to distinguish recall from background knowledge, without demanding that every
invariant survive a two-hour reconstruction.

**Why 7.** Below a third of the suite the rebuild has failed on shapes and
scaffolding rather than on subtleties, and the useful response is to re-read
rather than to re-attempt with the same preparation.

**Adapter shim.** A rebuild that is conceptually right but names things
differently would score near zero for the wrong reason. After the rebuild is
sealed, a shim may be written to reconcile names and signatures with what the
tests import — **renaming only, no logic, no behaviour**. Record its line count
in the result: a 5-line shim and a 60-line shim say different things about how
close the recall was, and that number is itself a datum.

## Remedy

Not "repeat Phase 2". Repeating Phase 2 means Claude writes the transformer
again, which does not act on what C2 measures. Per rule 4:

- **Partial fail (7–13):** re-read `crucible/model.py` and ADR-0007, wait one
  week, re-attempt under identical conditions. Record as C2b; both attempts
  stay in the ledger.
- **Hard fail (<7):** the same, but preceded by rebuilding one component at a
  time with the tests visible, until each passes individually. Then re-attempt
  whole and cold.
- Either way, **Phase 2 stays open** and the writeup says so.

## A note on sequencing

Phase 3 began on 2026-09-25 (fc7d6f7, 6e9c568), before this gate ran. That was
deliberate and is on the record in CLAUDE.md. C2 therefore cannot "unblock"
Phase 3 — that already happened. What C2 still decides is whether Phase 2
closes, and what the writeup is allowed to claim about how well the
architecture was understood. Note also that commit fc7d6f7's subject line reads
"Phase 2 closed out", which the gate has not granted; its body states the
deferral correctly.

## Sign-off

Registered before the attempt, 2026-09-26.

- Time box: **2 hours**
- Pass: **14 / 22**
- Hard-fail floor: **7 / 22**
- Interval: **2 days** (amended from 7 — see [Interval](#interval))
- No contact with `model.py` / `test_model.py` / ADR-0007 from now until the
  attempt, beyond the exposure recorded above
- Thresholds proposed by: Claude · Accepted unamended by: Hashith · Date:
  2026-09-26

Every number above is Claude's proposal that Hashith did not change. That is
recorded because the alternative — presenting them as Hashith's own bar — is
the substitution this file exists to prevent. What the registration does
guarantee is the thing that matters most: the bar was fixed and committed
before the result existed.
