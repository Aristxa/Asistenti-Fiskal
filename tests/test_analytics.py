"""Tests for act metadata parsing and obligation extraction.

Both layers are rule-based, which means their failure mode is silent: a pattern
that stops matching produces fewer rows, not an error. These pin the parses that
matter, using real title and article strings from the corpus.
"""

from __future__ import annotations

import pytest

from src.analytics.acts import classify, parse_act
from src.extract.obligations import (
    RELATIVE_DEADLINE,
    extract_from_chunk,
    find_subject,
    parse_relative_deadline,
)


def act(title: str, **extra) -> dict:
    return {"doc_id": 1, "title": title, "domain": "tatime", "authority": "DPT",
            "category": extra.pop("category", "tatime"), "n_articles": 0,
            "regime": "neni", "chars": 0, "url": "", **extra}


class TestActParsing:
    def test_parses_a_standard_vkm_title(self):
        row = parse_act(act('VKM Nr.783 datë 10.11.2011 "Për procedurat e ndarjes" i ndryshuar'))
        assert row.act_type == "VKM"
        assert row.number == 783
        assert row.date == "2011-11-10"
        assert row.year == 2011
        assert row.is_amended is True
        assert row.subject == "Për procedurat e ndarjes"

    def test_parses_slash_numbered_law(self):
        row = parse_act(act("Ligji nr. 25/2018 PËR KONTABILITETIN"))
        assert row.act_type == "Ligj"
        assert row.number == 25
        assert row.year == 2018

    def test_single_digit_day_and_month(self):
        row = parse_act(act("UMF Nr.1 datë 12.01.2007 'Për diçka'"))
        assert row.date == "2007-01-12"

    def test_rejects_impossible_dates(self):
        row = parse_act(act("VKM Nr.5 datë 45.99.2011 xxx"))
        assert row.date is None

    def test_amendment_and_repeal_flags(self):
        amending = parse_act(act("Për disa ndryshime në ligjin nr.9920 datë 19.5.2008"))
        assert amending.is_amendment is True
        assert 9920 in amending.amends

        repeal = parse_act(act("Shfuqizim i VKM Nr. 37 datë 21.1.2016"))
        assert repeal.is_repeal is True

    def test_an_act_does_not_amend_itself(self):
        row = parse_act(act("Ligji nr.9920 për ndryshime në ligjin nr.9920"))
        assert 9920 not in row.amends

    def test_country_titled_treaties_fall_back_to_category(self):
        """Regression: double-taxation treaties are titled with only the country
        name, so 43 of them were unclassified until the category was consulted."""
        assert classify("Franca", "marreveshje-nderkombetare") == "Marrëveshje"
        assert classify("Greqi", "marreveshje-nderkombetare") == "Marrëveshje"

    def test_title_still_wins_over_category(self):
        assert classify("Ligj Nr.1 datë 1.1.2020", "marreveshje-nderkombetare") == "Ligj"

    def test_unknown_stays_unknown(self):
        assert classify("diçka pa lloj", "") is None


class TestDeadlines:
    @pytest.mark.parametrize("text,days", [
        ("brenda 15 ditëve", 15),
        ("brenda 5 ditëve", 5),
        ("Brenda 30 ditëve", 30),
        ("brenda 3 muajve", 90),
        ("brenda dy javëve", 14),
        ("jo më vonë se 10 ditë", 10),
    ])
    def test_relative_deadlines_convert_to_days(self, text, days):
        match = RELATIVE_DEADLINE.search(text)
        assert match is not None, f"no match for {text!r}"
        _, parsed = parse_relative_deadline(match)
        assert parsed == days

    def test_word_numbers_are_understood(self):
        match = RELATIVE_DEADLINE.search("brenda tre ditëve")
        assert match is not None
        assert parse_relative_deadline(match)[1] == 3


class TestSubjects:
    @pytest.mark.parametrize("text,expected", [
        ("punëdhënësi duhet të paguajë", "punëdhënësi"),
        ("personi i tatueshëm është i detyruar", "personi i tatueshëm"),
        ("tatimpaguesi duhet të deklarojë", "tatimpaguesi"),
        ("njësia ekonomike duhet të mbajë", "njësia ekonomike"),
    ])
    def test_identifies_the_duty_holder(self, text, expected):
        assert find_subject(text) == expected

    def test_returns_none_when_no_subject_present(self):
        assert find_subject("duhet të bëhet brenda afatit") is None


class TestObligationExtraction:
    @staticmethod
    def chunk(text: str) -> dict:
        return {"doc_id": 42, "cite": "Neni 7", "heading": "Afatet", "text": text}

    meta = {"domain": "tatime", "authority": "DPT", "url": "http://x"}

    def test_extracts_a_complete_obligation(self):
        rows = extract_from_chunk(self.chunk(
            "Personi i tatueshëm është i detyruar të njoftojë organet tatimore "
            "brenda 15 ditëve nga data e ndryshimit."
        ), self.meta)
        assert len(rows) == 1
        row = rows[0]
        assert row.subject == "personi i tatueshëm"
        assert row.deadline_days == 15
        assert row.cite == "Neni 7"
        assert "subject" in row.matched_by and "deadline_relative" in row.matched_by

    def test_bare_duty_without_attributes_is_dropped(self):
        """A lone 'duhet të' is a sentence, not an obligation worth tabulating."""
        rows = extract_from_chunk(self.chunk("Ky rregull duhet të zbatohet."), self.meta)
        assert rows == []

    def test_captures_rate_and_penalty(self):
        rows = extract_from_chunk(self.chunk(
            "Tatimpaguesi duhet të paguajë 15% të vlerës, përndryshe zbatohet gjobë."
        ), self.meta)
        assert rows and rows[0].rates == ["15%"]
        assert rows[0].penalty is not None

    def test_span_does_not_start_mid_word(self):
        text = ("x" * 400) + " Tatimpaguesi duhet të paguajë brenda 10 ditëve afatin."
        rows = extract_from_chunk(self.chunk(text), self.meta)
        assert rows
        assert not rows[0].duty_span.startswith("x")
