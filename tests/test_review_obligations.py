"""Tests for the obligations review round trip.

This workbook is the interface to the only step that needs accounting expertise.
A silent failure here wastes the author's time rather than a machine's, and two of
these lock in bugs that already happened once on the question workbook: text handed
to a reviewer in truncated form, and a re-export destroying judgements.
"""

from __future__ import annotations

import csv
import json

import pytest

from src.extract import review_obligations as ro


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    (processed / "index/article").mkdir(parents=True)
    monkeypatch.setattr(ro, "PROCESSED", processed)
    monkeypatch.setattr(ro, "SOURCE", processed / "obligations.jsonl")
    monkeypatch.setattr(ro, "WORKBOOK", tmp_path / "detyrimet.csv")
    monkeypatch.setattr(ro, "SAMPLE_SIZE", 2)

    obligations = [
        {"obligation_id": "1:Neni 5:0", "doc_id": 1, "cite": "Neni 5",
         "domain": "tatime", "subject": "tatimpaguesi", "duty_span": "fragment i shkurtër",
         "deadline_text": "brenda 15 ditëve", "rates": ["20%"], "penalty": None},
        {"obligation_id": "1:Neni 6:0", "doc_id": 1, "cite": "Neni 6",
         "domain": "tatime", "subject": None, "duty_span": "fragment tjetër",
         "deadline_text": None, "rates": [], "penalty": None},
    ]
    (processed / "obligations.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in obligations),
        encoding="utf-8")

    # Neni 5 is stored in two parts, out of order, to check they are reassembled.
    chunks = [
        {"doc_id": 1, "cite": "Neni 5", "order": 1, "text": "PJESA E DYTË"},
        {"doc_id": 1, "cite": "Neni 5", "order": 0, "text": "PJESA E PARË"},
        {"doc_id": 1, "cite": "Neni 6", "order": 0, "text": "Teksti i nenit 6"},
    ]
    (processed / "index/article/chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    return tmp_path


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    head = next(i for i, r in enumerate(rows) if r and r[0].strip() == "obligation_id")
    header = [c.strip() for c in rows[head]]
    return [dict(zip(header, r + [""] * (len(header) - len(r))))
            for r in rows[head + 1:] if r and r[0].strip()], rows, head


def test_export_uses_the_whole_article_not_the_extracted_span(workspace):
    """Regression: the span was truncated at 600 chars and was a fragment anyway.

    A reviewer shown a fragment is being asked to judge blind — the same defect
    that made the question review unworkable until it was fixed there.
    """
    ro.do_export()
    records, _, _ = read(ro.WORKBOOK)
    neni5 = next(r for r in records if r["neni"] == "Neni 5")
    assert "PJESA E PARË" in neni5["teksti"]
    assert "PJESA E DYTË" in neni5["teksti"]
    assert "fragment" not in neni5["teksti"]


def test_article_parts_are_reassembled_in_order(workspace):
    ro.do_export()
    records, _, _ = read(ro.WORKBOOK)
    text = next(r for r in records if r["neni"] == "Neni 5")["teksti"]
    assert text.index("PJESA E PARË") < text.index("PJESA E DYTË")


def test_reexport_preserves_existing_judgements(workspace):
    """Re-running export must never throw away work already done by hand."""
    ro.do_export()
    records, rows, head = read(ro.WORKBOOK)
    rows[head + 1][1] = "po"
    rows[head + 1][2] = "shënim i rëndësishëm"
    with ro.WORKBOOK.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle, delimiter=";").writerows(rows)
    judged_id = rows[head + 1][0]

    ro.do_export()

    records, _, _ = read(ro.WORKBOOK)
    kept = next(r for r in records if r["obligation_id"] == judged_id)
    assert kept["e_sakte"] == "po"
    assert kept["shenime"] == "shënim i rëndësishëm"


def test_score_reports_precision_over_judged_rows_only(workspace, capsys):
    ro.do_export()
    records, rows, head = read(ro.WORKBOOK)
    rows[head + 1][1] = "po"
    rows[head + 2][1] = "jo"
    with ro.WORKBOOK.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle, delimiter=";").writerows(rows)

    ro.do_score()
    out = capsys.readouterr().out
    assert "1/2" in out and "50%" in out


def test_falls_back_to_the_span_when_the_article_is_missing(workspace):
    """A missing index entry must not produce an empty cell to judge against."""
    (workspace / "processed/index/article/chunks.jsonl").write_text("", encoding="utf-8")
    ro.do_export()
    records, _, _ = read(ro.WORKBOOK)
    assert all(r["teksti"].strip() for r in records)
