"""C2: score a from-memory rebuild of `crucible/model.py` against tests/test_model.py.

    python scripts/score_c2.py results/c2_rebuild.py   # writes results/c2_rebuild.json
    python scripts/score_c2.py --self-test             # scores the real model.py: 22/22

The rebuild is substituted for `crucible.model` in `sys.modules`, so the test
file runs unmodified. Group A / Group B and the pass bands come from
`results/LEDGER.md`; see that file for what the numbers mean and why.

REFUSES TO RUN if the rebuild is untracked or has uncommitted changes. The seal
is what makes the score meaningful: a file edited after the first traceback
measures debugging, not recall. Override with --no-seal-check only for a dry
run, which is recorded in the output and is not a C2 result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = ROOT / "tests" / "test_model.py"
RESULTS_DIR = ROOT / "results"

# LEDGER.md, C2 -> Measure. Group A is reachable from a generic transformer;
# Group B is specific to what was read. The split is the argument for the pass
# mark, so it is pinned here and verified against collection below.
GROUP_A = (
    "test_config_rejects_indivisible_head_split",
    "test_rmsnorm_is_scale_invariant_and_shape_preserving",
    "test_rmsnorm_survives_zeros_and_normalises_last_dim_only",
    "test_rmsnorm_has_one_parameter_vector",
    "test_rope_preserves_shape_and_norm",
    "test_mlp_is_position_wise",
    "test_block_with_zeroed_outputs_is_identity",
    "test_forward_shape_and_dtype",
    "test_single_step_forward_works",
    "test_forward_is_deterministic",
)
GROUP_B = (
    "test_s25_parameter_count_matches_adr_0006",
    "test_rope_at_position_zero_is_identity",
    "test_rope_dot_product_depends_only_on_offset",
    "test_attention_is_causal",
    "test_model_is_causal",
    "test_context_limit_and_id_range_are_enforced",
    "test_every_parameter_receives_gradient",
    "test_gradients_ignore_later_tokens",
    "test_cache_reports_length_and_rejects_overflow",
    "test_kv_cache_matches_full_forward",
    "test_cache_reset_gives_a_fresh_sequence",
    "test_runs_on_gpu_and_matches_cpu",
)

# LEDGER.md, C2 -> Thresholds. PROPOSED until the sign-off block is filled in.
PASS_AT = 14
HARD_FAIL_BELOW = 7

# Recall of ADR-0006's constants rather than of the architecture; scored, but
# reported on its own so it cannot be mistaken for a structural failure.
CONSTANT_RECALL = "test_s25_parameter_count_matches_adr_0006"


class Recorder:
    """Collects one pass/fail per test from the pytest run."""

    def __init__(self) -> None:
        self.outcomes: dict[str, str] = {}
        self.errors: dict[str, str] = {}

    def pytest_runtest_logreport(self, report) -> None:
        name = report.nodeid.split("::")[-1]
        if report.when == "call" or (
            report.when == "setup" and report.outcome != "passed"
        ):
            if self.outcomes.get(name) == "failed":
                return
            self.outcomes[name] = report.outcome
            if report.outcome == "failed" and report.longrepr is not None:
                self.errors[name] = str(report.longrepr).strip().splitlines()[-1][:300]


def seal_state(path: Path) -> dict[str, object]:
    """Whether the rebuild is committed and unmodified, per LEDGER.md step 2."""

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )

    tracked = git("ls-files", "--error-unmatch", str(path)).returncode == 0
    clean = tracked and git("diff", "--quiet", "HEAD", "--", str(path)).returncode == 0
    commit = git("log", "-1", "--format=%H", "--", str(path)).stdout.strip() or None
    return {"tracked": tracked, "unmodified": clean, "sealed_in": commit}


def load_rebuild(path: Path, shim: Path | None) -> None:
    """Install `path` as `crucible.model` so test_model.py imports it instead."""
    import crucible  # the package; only `crucible.model` is replaced

    for name, file in (("crucible.model", path), ("c2_shim", shim)):
        if file is None:
            continue
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if name == "crucible.model":
            crucible.model = module  # type: ignore[attr-defined]


def rel_to_root(path: Path) -> str:
    """Repo-relative, forward-slashed; absolute if the file sits outside the repo."""
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def band(passed: int) -> str:
    if passed >= PASS_AT:
        return "pass"
    if passed < HARD_FAIL_BELOW:
        return "hard fail"
    return "partial fail"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rebuild", nargs="?", type=Path, help="the sealed rebuild file")
    parser.add_argument(
        "--shim", type=Path, help="renaming-only adapter, scored for length"
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "c2_rebuild.json")
    parser.add_argument(
        "--self-test", action="store_true", help="score crucible/model.py"
    )
    parser.add_argument(
        "--no-seal-check", action="store_true", help="dry run; not a result"
    )
    args = parser.parse_args(argv)

    if args.self_test:
        target = ROOT / "crucible" / "model.py"
    elif args.rebuild is not None:
        target = args.rebuild
    else:
        parser.error("give a rebuild file, or --self-test")
    if not target.exists():
        print(f"no such file: {target}", file=sys.stderr)
        return 1

    dry_run = args.self_test or args.no_seal_check
    seal = seal_state(target)
    if not dry_run and not seal["unmodified"]:
        why = "is not committed" if not seal["tracked"] else "has uncommitted changes"
        print(
            f"refusing to score: {target} {why}.\n"
            "LEDGER.md requires the rebuild to be sealed before it is run. "
            "Commit it unmodified, then score.",
            file=sys.stderr,
        )
        return 2

    sys.path.insert(0, str(ROOT))
    load_rebuild(target, args.shim)

    recorder = Recorder()
    pytest.main(
        [str(TEST_FILE), "-q", "--no-header", "-p", "no:cacheprovider"],
        plugins=[recorder],
    )

    collected = set(recorder.outcomes)
    expected = set(GROUP_A) | set(GROUP_B)
    # A rebuild whose public names differ from what the tests import fails at
    # collection, so nothing runs. That is a real 0, not a broken harness --
    # and it is exactly what the shim exists to rescue.
    collection_failed = not collected
    if collection_failed:
        print(
            f"\n{TEST_FILE.name} could not import {target.name}: "
            f"scored 0/{len(expected)}.\n"
            "If the structure is there and only the names differ, write a "
            "renaming-only shim (LEDGER.md, C2 -> Adapter shim) and re-run with "
            "--shim; its line count is recorded.",
            file=sys.stderr,
        )
        recorder.outcomes = dict.fromkeys(expected, "not collected")
        collected = set(expected)
    elif collected != expected:
        print(
            "refusing to score: test_model.py no longer matches the ledger's groups.\n"
            f"  only in tests : {sorted(collected - expected)}\n"
            f"  only in ledger: {sorted(expected - collected)}\n"
            "Update results/LEDGER.md and this script together; the denominator "
            "must not drift silently.",
            file=sys.stderr,
        )
        return 3

    def tally(group: tuple[str, ...]) -> int:
        return sum(recorder.outcomes[n] == "passed" for n in group)

    a, b = tally(GROUP_A), tally(GROUP_B)
    total = a + b
    shim_lines = (
        len(args.shim.read_text(encoding="utf-8").splitlines()) if args.shim else 0
    )

    result = {
        "c2": {
            "rebuild": rel_to_root(target),
            "dry_run": dry_run,
            "seal": seal,
            "shim_lines": shim_lines,
            "collection_failed": collection_failed,
            "passed": total,
            "of": len(expected),
            "group_a": {"passed": a, "of": len(GROUP_A)},
            "group_b": {"passed": b, "of": len(GROUP_B)},
            "constant_recall_test": recorder.outcomes[CONSTANT_RECALL],
            "pass_at": PASS_AT,
            "hard_fail_below": HARD_FAIL_BELOW,
            "verdict": band(total),
            "measures": "recall of code read, not code written (CLAUDE.md, Authorship)",
        },
        "tests": [
            {
                "name": n,
                "outcome": recorder.outcomes[n],
                "error": recorder.errors.get(n),
            }
            for n in GROUP_A + GROUP_B
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nGroup A {a}/{len(GROUP_A)}   Group B {b}/{len(GROUP_B)}")
    print(f"C2: {total}/{len(expected)} -> {band(total).upper()}", end="")
    print("  (dry run, not a C2 result)" if dry_run else "")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
