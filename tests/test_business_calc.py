"""Tests for the VAT and business income tax calculators.

Same concern as the payroll module: a wrong number looks exactly like a right one,
so the arithmetic is pinned to values that can be checked by hand against the
articles, and the provisions that expire are tested for saying so.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.calc import business


def test_vat_added_to_a_net_amount():
    r = business.vat(100_000)
    assert r.named("TVSH") == pytest.approx(20_000)
    assert r.named("Vlera me TVSH") == pytest.approx(120_000)


def test_vat_extracted_from_a_gross_amount():
    """VAT inside a gross price is 20/120 of it, not 20% — the common mistake."""
    r = business.vat(120_000, amount_includes_vat=True)
    assert r.named("Vlera e tatueshme (pa TVSH)") == pytest.approx(100_000)
    assert r.named("TVSH") == pytest.approx(20_000)


def test_extracting_then_adding_vat_round_trips():
    gross = 87_654.0
    net = business.vat(gross, amount_includes_vat=True).named("Vlera e tatueshme (pa TVSH)")
    assert business.vat(net).named("Vlera me TVSH") == pytest.approx(gross)


def test_reduced_rates_are_declared_not_applied():
    """Which supply qualifies for 6% is a legal classification, not arithmetic."""
    assert any("reduktuar" in w for w in business.vat(1_000).warnings)


def test_small_business_is_zero_rated_within_the_transitional_period():
    r = business.business_income_tax(3_000_000, turnover=8_000_000,
                                     today=date(2026, 9, 10))
    assert r.named("Tatimi mbi të ardhurat nga biznesi") == 0.0
    assert any("31.12.2029" in w for w in r.warnings)


def test_the_zero_rate_depends_on_turnover_not_profit():
    """A small profit on large turnover is taxed; conflating the two would not."""
    r = business.business_income_tax(1_000_000, turnover=60_000_000,
                                     today=date(2026, 9, 10))
    assert r.named("Tatimi mbi të ardhurat nga biznesi") == pytest.approx(150_000)


def test_progressive_bands_above_the_ceiling():
    r = business.business_income_tax(20_000_000, turnover=60_000_000,
                                     today=date(2026, 9, 10))
    # 15% of 14 000 000, then 23% of the remaining 6 000 000.
    assert r.named("Tatimi mbi të ardhurat nga biznesi") == pytest.approx(3_480_000)


def test_bands_join_without_a_step_at_the_ceiling():
    below = business.business_income_tax(14_000_000, turnover=60_000_000).named(
        "Tatimi mbi të ardhurat nga biznesi")
    above = business.business_income_tax(14_000_001, turnover=60_000_000).named(
        "Tatimi mbi të ardhurat nga biznesi")
    assert above == pytest.approx(below + 0.23, abs=0.01)


def test_expired_zero_rate_falls_back_and_says_so():
    r = business.business_income_tax(3_000_000, turnover=8_000_000,
                                     today=date(2030, 6, 1))
    assert r.named("Tatimi mbi të ardhurat nga biznesi") == pytest.approx(450_000)
    assert any("afat deri" in w for w in r.warnings)


def test_missing_turnover_is_declared():
    r = business.business_income_tax(3_000_000, today=date(2026, 9, 10))
    assert any("Qarkullimi vjetor nuk u dha" in w for w in r.warnings)


def test_every_line_carries_a_source():
    for r in (business.vat(1_000),
              business.business_income_tax(1_000_000, turnover=2_000_000)):
        assert r.lines
        for line in r.lines:
            assert line.source.cite and line.source.doc_id
