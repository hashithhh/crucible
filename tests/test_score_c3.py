"""C3: the scorer must read the bar the ledger registered, and only that.

C3's worth comes from the thresholds having been fixed before a number
existed. What can quietly undo that is not someone editing the ledger -- it is
the scorer keeping its own copy and drifting from it. So the parse is what is
tested, including that it refuses rather than guesses.
"""

from __future__ import annotations

import pytest

from scripts.score_c3 import BASELINES, bits_per_byte, parse_thresholds, verdict

REGISTERED = """# C3 - held-out loss

- **Hard fail: ≥ 3.6190** - the bigram baseline.
- **Pass: ≤ 2.60** (approx 1.00 bits/byte).
- **Stretch, recorded but NOT gating: ≤ 2.00** (approx 0.77 bits/byte).

# C4 - something else
- **Pass: ≤ 9.99**
"""


def test_the_registered_bounds_parse_out_of_the_real_ledger():
    bounds = parse_thresholds()
    assert bounds["hard_fail_at"] == 3.6190
    assert bounds["pass_at"] == 2.60
    assert bounds["stretch_at"] == 2.00


def test_the_hard_fail_bar_is_the_measured_bigram_baseline():
    """C3's argument: at or above the bigram table, the transformer did nothing."""
    assert parse_thresholds()["hard_fail_at"] == BASELINES["bigram"]


def test_parsing_stops_at_the_c4_entry(tmp_path):
    """C4 has a 'Pass:' line too; taking it would score against the wrong bar."""
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text(REGISTERED, encoding="utf-8")
    assert parse_thresholds(ledger)["pass_at"] == 2.60


def test_a_reworded_entry_is_refused_rather_than_guessed(tmp_path):
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text(
        REGISTERED.replace("**Pass: ≤ 2.60**", "Pass: about 2.6 or so"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pass at"):
        parse_thresholds(ledger)


def test_thresholds_out_of_order_are_refused(tmp_path):
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text(
        REGISTERED.replace("**Pass: ≤ 2.60**", "**Pass: ≤ 5.00**"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not ordered"):
        parse_thresholds(ledger)


def test_a_ledger_without_c3_is_refused(tmp_path):
    ledger = tmp_path / "LEDGER.md"
    ledger.write_text("# C1 - something\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no C3"):
        parse_thresholds(ledger)


@pytest.mark.parametrize(
    "loss,expected",
    [
        (1.40, "pass"),
        (2.60, "pass"),  # the bar is inclusive, as registered
        (2.61, "fail"),
        (3.50, "fail"),
        (3.6190, "hard fail"),  # at the bigram baseline, not merely above it
        (6.10, "hard fail"),
    ],
)
def test_verdict_bands(loss, expected):
    assert verdict(loss, parse_thresholds()) == expected


def test_bits_per_byte_matches_the_registered_conversion():
    """The ledger quotes 1.399 bits/byte for the 3.6190 bigram baseline."""
    assert bits_per_byte(BASELINES["bigram"]) == pytest.approx(1.399, abs=5e-4)
    assert bits_per_byte(BASELINES["uniform"]) == pytest.approx(2.947, abs=5e-4)
