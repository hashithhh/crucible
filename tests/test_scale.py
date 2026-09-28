"""Phase 5: the fit and the plan, the two things that could be silently wrong.

A fit that returns a plausible but wrong exponent would produce a confident,
false comparison with Chinchilla. So it is checked against data whose answer
is known.
"""

from __future__ import annotations

import pytest

from scripts.scale import SIZES, fit_power_law, plan

LADDER_N = [3_672_576, 8_032_640, 14_949_120, 26_223_616]


@pytest.mark.parametrize("gamma", [0.28, 0.34, 0.5])
def test_fit_recovers_a_known_exponent(gamma):
    e, k = 1.0, 50.0
    losses = [e + k * n**-gamma for n in LADDER_N]
    fit = fit_power_law(LADDER_N, losses)
    assert fit["gamma"] == pytest.approx(gamma, abs=0.02)
    assert fit["E"] == pytest.approx(e, abs=0.05)


def test_fit_predicts_the_points_it_was_given():
    losses = [2.20, 1.80, 1.55, 1.34]
    fit = fit_power_law(LADDER_N, losses)
    for n, loss in zip(LADDER_N, losses, strict=True):
        predicted = fit["E"] + fit["K"] * n ** -fit["gamma"]
        assert predicted == pytest.approx(loss, abs=0.05)


def test_step_counts_match_adr_0010():
    """round(20 x params / 65,536), as tabulated in ADR-0010."""
    expected = {"S3": 1121, "S7": 2451, "S13": 4562}
    assert {name: plan(cfg).steps for name, cfg in SIZES.items()} == expected
