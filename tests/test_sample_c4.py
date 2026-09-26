"""Phase 3.7 / C4: the sampling script must use what C4 registered.

C4's value rests on the prompts and settings having been fixed before the
model existed. That guarantee is only as strong as the link between the ledger
and the script, so the link itself is tested: prompts are read from the
ledger, and the settings the script hard-codes are checked back against the
ledger's own text.
"""

from __future__ import annotations

import pytest

from scripts.sample import (
    EXPECTED_PROMPTS,
    LEDGER,
    MAX_NEW_TOKENS,
    SEED,
    TEMPERATURE,
    TOP_P,
    load_prompts,
)


def test_the_registered_prompts_parse_out_of_the_ledger():
    prompts = load_prompts()
    assert len(prompts) == EXPECTED_PROMPTS
    assert all(p.strip() for p in prompts)
    assert len(set(prompts)) == EXPECTED_PROMPTS, "C4's prompts must be distinct"


def test_the_first_prompt_is_the_one_c4_registered():
    """A canary: if the ledger's list is reordered or rewritten, this fails."""
    assert load_prompts()[0] == "Once upon a time, there was a little girl named Lily."


def test_the_hard_coded_settings_match_the_ledger_entry():
    """The script's constants and C4's registered ones cannot drift apart."""
    text = LEDGER.read_text(encoding="utf-8")
    c4 = text[text.find("# C4 ") :]
    for value in (str(TEMPERATURE), str(TOP_P), str(MAX_NEW_TOKENS), str(SEED)):
        assert value in c4, f"C4 does not register {value}"


def test_a_ledger_with_the_wrong_number_of_prompts_is_refused(tmp_path):
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text("# C4 - coherence\n\n1. `only one prompt`\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered"):
        load_prompts(ledger)


def test_a_ledger_without_a_c4_entry_is_refused(tmp_path):
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text("# C1 - something else\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no C4"):
        load_prompts(ledger)


def test_a_missing_ledger_says_where_the_prompts_live(tmp_path):
    with pytest.raises(FileNotFoundError, match="C4"):
        load_prompts(tmp_path / "absent.md")
