"""Tests for the payroll calculator.

A calculator differs from the rest of this system in one dangerous way: it returns
a number rather than a citation, and a wrong number looks exactly like a right
one. These tests are therefore about arithmetic that can be checked by hand
against the statute, and about the calculator being honest when it is out of date
or missing a parameter.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.calc import payroll


@pytest.mark.parametrize("gross,expected", [
    (0, 0),
    (40_000, 0),
    (50_000, 0),          # top of the exempt band
    (55_000, 2_600),      # 13% of (55 000 − 35 000)
    (60_000, 3_250),      # 13% of (60 000 − 35 000)
    (60_001, 3_900.13),   # band changes: 13% of (60 001 − 30 000)
    (85_000, 7_150),      # 13% of (85 000 − 30 000)
    (200_000, 22_100),    # 13% of 170 000 — the fixed amount in the statute
    (250_000, 33_600),    # 22 100 + 23% of 50 000
])
def test_monthly_tax_matches_the_statute_table(gross, expected):
    tax, _ = payroll.monthly_income_tax(gross)
    assert tax == pytest.approx(expected, abs=0.5)


def test_the_fixed_amount_replaces_the_lower_band_rather_than_adding_to_it():
    """Regression: the top step reads "22 100 lekë + 23% të shumës mbi 200 000".

    The 22 100 *is* the tax on the 30 001–200 000 band. An earlier version added
    it on top of that band, overstating tax on high salaries by exactly 22 100 —
    a figure that still looked plausible on screen.
    """
    at_threshold, _ = payroll.monthly_income_tax(200_000)
    just_above, _ = payroll.monthly_income_tax(200_001)
    assert just_above == pytest.approx(at_threshold + 0.23, abs=0.01)


def test_contributions_use_the_rates_of_the_decision_in_force():
    result = payroll.calculate(100_000, min_wage=40_000, max_wage=176_416,
                               today=date(2024, 6, 1))
    assert result.named("Sigurime shoqërore (punëmarrësi)") == pytest.approx(9_500)
    assert result.named("Sigurime shoqërore (punëdhënësi)") == pytest.approx(15_000)
    assert result.named("Sigurime shëndetësore (punëmarrësi)") == pytest.approx(1_700)
    assert result.named("Sigurime shëndetësore (punëdhënësi)") == pytest.approx(1_700)


def test_contribution_base_is_capped_at_the_maximum_wage():
    result = payroll.calculate(500_000, min_wage=40_000, max_wage=176_416,
                               today=date(2024, 6, 1))
    assert result.named("Sigurime shoqërore (punëmarrësi)") == pytest.approx(
        176_416 * 0.095)


def test_contribution_base_is_floored_at_the_minimum_wage():
    result = payroll.calculate(10_000, min_wage=40_000, max_wage=176_416,
                               today=date(2024, 6, 1))
    assert result.named("Sigurime shoqërore (punëmarrësi)") == pytest.approx(
        40_000 * 0.095)


def test_missing_wage_bounds_are_declared_not_guessed():
    """Those bounds live in a separate decision the corpus does not contain."""
    result = payroll.calculate(85_000, today=date(2024, 6, 1))
    assert any("kufirin minimal" in w for w in result.warnings)


def test_an_expired_table_says_so():
    """The monthly table is transitional and lapsed on 31 December 2024.

    Applying a lapsed rate silently is the single worst thing this module could
    do, so the expiry is checked against the date the calculation is run.
    """
    result = payroll.calculate(85_000, min_wage=40_000, max_wage=176_416,
                               today=date(2026, 9, 10))
    assert any("31.12.2024" in w for w in result.warnings)


def test_a_calculation_inside_the_validity_period_raises_no_expiry_warning():
    result = payroll.calculate(85_000, min_wage=40_000, max_wage=176_416,
                               today=date(2024, 6, 1))
    assert not any("31.12.2024" in w for w in result.warnings)


def test_every_line_carries_a_source():
    """The whole point: no figure without the provision it rests on."""
    result = payroll.calculate(85_000, min_wage=40_000, max_wage=176_416)
    assert result.lines
    for line in result.lines:
        assert line.source.cite
        assert line.source.doc_id
        assert line.source.url.startswith("https://www.tatime.gov.al/")


def test_net_and_employer_cost_are_consistent():
    r = payroll.calculate(85_000, min_wage=40_000, max_wage=176_416,
                          today=date(2024, 6, 1))
    assert r.net == pytest.approx(
        85_000 - r.named("Sigurime shoqërore (punëmarrësi)")
        - r.named("Sigurime shëndetësore (punëmarrësi)")
        - r.named("Tatimi mbi të ardhurat nga paga"))
    assert r.employer_total > r.gross
