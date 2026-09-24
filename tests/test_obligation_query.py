"""Tests for the obligations query layer.

This layer answers what retrieval cannot: "which duties fall due within fifteen
days" is a filter over structured fields, not a similarity search. The tests below
guard the two things that would quietly mislead a user — a filter that does not
filter, and a view that overstates what the extractor actually found.
"""

from __future__ import annotations

import json

import pytest

from src.extract import query as q


@pytest.fixture
def rows(tmp_path, monkeypatch):
    source = tmp_path / "obligations.jsonl"
    records = [
        # Same article twice: the extractor emits one row per matched duty phrase.
        {"obligation_id": "1:Neni 5:0", "doc_id": 1, "cite": "Neni 5",
         "heading": "Regjistrimi", "domain": "tatime", "authority": "DPT",
         "subject": "tatimpaguesi", "duty_span": "duhet te regjistrohet",
         "deadline_text": "brenda 15 diteve", "deadline_days": 15,
         "rates": [], "penalty": None, "form": None},
        {"obligation_id": "1:Neni 5:1", "doc_id": 1, "cite": "Neni 5",
         "heading": "Regjistrimi", "domain": "tatime", "authority": "DPT",
         "subject": "tatimpaguesi", "duty_span": "duhet te regjistrohet",
         "deadline_text": "brenda 15 diteve", "deadline_days": 15,
         "rates": ["20%"], "penalty": "gjobe 10000 leke", "form": "formular"},
        {"obligation_id": "2:Neni 9:0", "doc_id": 2, "cite": "Neni 9",
         "heading": "Pushimet", "domain": "pune", "authority": "ISHPSH",
         "subject": "punëdhënësi", "duty_span": "duhet te japi pushim",
         "deadline_text": None, "deadline_days": None,
         "rates": [], "penalty": None, "form": None},
        {"obligation_id": "3:Neni 3:0", "doc_id": 3, "cite": "Neni 3",
         "heading": "Afati i gjate", "domain": "tatime", "authority": "DPT",
         "subject": "tatimpaguesi", "duty_span": "duhet te deklaroje",
         "deadline_text": "brenda 90 diteve", "deadline_days": 90,
         "rates": [], "penalty": "sanksionon", "form": None},
    ]
    source.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records),
                      encoding="utf-8")
    monkeypatch.setattr(q, "SOURCE", source)
    q.load.cache_clear()
    yield
    q.load.cache_clear()


def test_one_row_per_article(rows):
    """Regression: three copies of Neni 23 filled the top of a 15-day query.

    The extractor emits a row per matched duty phrase, so an article can repeat
    with identical fields. Duplicates crowd out other articles without adding
    information — the same defect that once affected retrieval results.
    """
    found = q.search(limit=50)
    cites = [r.cite for r in found]
    assert len(cites) == len(set(cites))


def test_deduplication_keeps_the_richest_row(rows):
    """Of two rows for one article, the one carrying more facts must survive."""
    neni5 = next(r for r in q.search(limit=50) if r.cite == "Neni 5")
    assert neni5.rates == ("20%",)
    assert neni5.has_real_penalty


def test_max_days_excludes_longer_deadlines(rows):
    found = q.search(max_days=15)
    assert {r.cite for r in found} == {"Neni 5"}
    assert all(r.deadline_days <= 15 for r in found)


def test_max_days_excludes_rows_without_a_deadline(rows):
    """A missing deadline is unknown, not "within N days"."""
    assert all(r.deadline_days is not None for r in q.search(max_days=90))


def test_a_bare_verb_is_not_a_penalty(rows):
    """`penalty="sanksionon"` is the grammar of a sanction without its content.

    Offering such a row under "has a penalty" would report a finding the extractor
    did not actually make.
    """
    found = q.search(with_penalty=True)
    assert {r.cite for r in found} == {"Neni 5"}


def test_text_search_folds_diacritics(rows):
    """Albanian is typed without ë and ç — the finding behind F8.

    "punedhenesi" and "punëdhënësi" must reach the same row, or the box silently
    fails for the majority of users who omit the diacritics.
    """
    folded = {r.cite for r in q.search(text="punedhenesi")}
    exact = {r.cite for r in q.search(text="punëdhënësi")}
    assert folded == exact == {"Neni 9"}


def test_text_search_covers_the_subject_field(rows):
    """A search box that ignores a word the user can see is worse than none."""
    assert {r.cite for r in q.search(text="tatimpaguesi")} == {"Neni 5", "Neni 3"}


def test_domain_and_subject_filters(rows):
    assert {r.cite for r in q.search(domain="pune")} == {"Neni 9"}
    assert {r.cite for r in q.search(subject="punëdhënësi")} == {"Neni 9"}


def test_deadlines_are_sorted_shortest_first(rows):
    found = [r for r in q.search(with_deadline=True) if r.deadline_days]
    assert found == sorted(found, key=lambda r: r.deadline_days)


def test_coverage_reports_totals_not_just_hits(rows):
    cov = q.coverage()
    assert cov["afat"] == (3, 4)
    assert cov["sanksion"] == (1, 4)


def test_routing_sends_enumeration_questions_to_the_table():
    assert q.looks_like_a_table_question("Cilat detyrime kam si punëdhënës?")
    assert q.looks_like_a_table_question("Çfarë duhet dorëzuar brenda 30 ditësh?")


def test_routing_leaves_explanatory_questions_to_retrieval():
    assert not q.looks_like_a_table_question("Sa është norma e TVSH-së?")
    assert not q.looks_like_a_table_question("Kur duhet të regjistrohem për TVSH?")
