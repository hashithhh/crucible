# Crucible — Phase 1: BPE Tokenizer

Rung 1 of the Crucible ladder: a ~100M-parameter LLM trained from scratch.
This phase is a byte-level BPE tokenizer, written by hand, standard library
only.

## Status

Phase 1 of 5 — **red**. The interface and test suite exist; the
implementation does not. That is intentional.

## Run the tests

```
pip install -e ".[dev]"
pytest
```

Everything fails with `NotImplementedError`. Make them pass one at a time.

To re-run one test on every save of `tokenizer.py`:

```
.\scripts\watch.ps1                                            # 256-is-legal test
.\scripts\watch.ps1 tests/test_train.py::test_vocab_size_is_exact
```

Watch settings are in `[tool.pytest-watcher]` in `pyproject.toml`.

Start with `tests/test_train.py::test_vocab_size_below_256_raises`. It needs
only `__init__` and an argument check — no encoding, no merges, no state.

Then immediately `test_vocab_size_exactly_256_is_legal`. At vocab_size 256
there are no merges, so it needs only byte-level `encode`/`decode`. Run the
two together: a `train` that always raises would pass the first on its own,
and the second is what proves you did not cheat.

After that: merges in `train` -> `encode` -> `save`/`load`.

Two tests are deliberately skipped, pending decisions only you can make:
ADR-0002 (tie-break rule) and ADR-0003 (merge exhaustion).
