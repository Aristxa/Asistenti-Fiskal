"""Tests for the benchmark review round trip.

The workbook is the interface between the pipeline and the only step that needs
tax expertise, so a silent failure here wastes the author's time rather than a
machine's. Two of these lock in bugs that actually occurred.
"""

from __future__ import annotations

import csv
import json

import pytest

from src.eval import review


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the module at a temporary gold directory."""
    gold = tmp_path / "gold"
    gold.mkdir()
    monkeypatch.setattr(review, "GOLD", gold)
    monkeypatch.setattr(review, "CANDIDATES", gold / "candidates.jsonl")
    monkeypatch.setattr(review, "WORKBOOK", gold / "rishikim.csv")
    monkeypatch.setattr(review, "QUESTIONS", gold / "questions.jsonl")
    # Point the out-of-scope slice at the temp directory too. Left unpatched it
    # resolves to the real 22-question file and every import test inherits those
    # rows, so assertions about what import kept would silently be about the wrong
    # set.
    monkeypatch.setattr(review, "OUT_OF_SCOPE", gold / "jashte_teme.jsonl")

    candidates = [
        {"id": "tvsh-1", "question": "", "gold_doc_id": 14534,
         "gold_article": "Neni 117", "category": "tvsh", "answerable": True,
         "reviewed": False, "source_url": "http://x/1",
         "source_heading": "Regjistrimi", "source_text": "Teksti i nenit 117 ..."},
        {"id": "tvsh-2", "question": "", "gold_doc_id": 14534,
         "gold_article": "Neni 118", "category": "tvsh", "answerable": True,
         "reviewed": False, "source_url": "http://x/2",
         "source_heading": "Çregjistrimi", "source_text": "Teksti i nenit 118 ..."},
        {"id": "tap-1", "question": "", "gold_doc_id": 9999,
         "gold_article": "12.3", "category": "tap", "answerable": True,
         "reviewed": False, "source_url": "http://x/3",
         "source_heading": "Afatet", "source_text": "Teksti me pikë 12.3 ..."},
    ]
    review.CANDIDATES.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in candidates),
        encoding="utf-8",
    )
    return gold


def read_sheet(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle, delimiter=";"))


def write_sheet(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle, delimiter=";").writerows(rows)


def header_index(rows):
    return next(i for i, r in enumerate(rows) if r and r[0].strip() == "id")


def test_export_writes_every_candidate(workspace):
    review.do_export()
    rows = read_sheet(review.WORKBOOK)
    assert len(rows) - header_index(rows) - 1 == 3


def test_export_preserves_albanian_diacritics(workspace):
    review.do_export()
    text = review.WORKBOOK.read_text(encoding="utf-8-sig")
    assert "Çregjistrimi" in text


def test_export_uses_bom_so_excel_reads_utf8(workspace):
    """Without the BOM Excel guesses the system codepage and mangles ë and ç."""
    review.do_export()
    assert review.WORKBOOK.read_bytes().startswith(b"\xef\xbb\xbf")


def test_header_is_found_despite_quoted_instruction_lines(workspace):
    """Regression: instruction lines contain ';', so csv quotes them and they
    start with '"' rather than '#'. Filtering raw lines on '#' dropped nothing
    and the import silently produced zero rows."""
    review.do_export()
    rows = read_sheet(review.WORKBOOK)
    index = header_index(rows)
    assert index > 0                      # there really is a preamble
    assert rows[index][0].strip() == "id"


def test_import_keeps_only_reviewed_rows_with_questions(workspace):
    review.do_export()
    rows = read_sheet(review.WORKBOOK)
    h = header_index(rows)
    rows[h + 1][2] = "Kur duhet të regjistrohem?"
    rows[h + 1][6] = "po"
    rows[h + 2][6] = "po"                 # reviewed but no question
    # rows[h + 3] left untouched
    write_sheet(review.WORKBOOK, rows)

    review.do_import()
    kept = [json.loads(l) for l in
            review.QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["id"] for r in kept] == ["tvsh-1"]
    assert kept[0]["question"] == "Kur duhet të regjistrohem?"


def test_import_honours_a_corrected_gold_article(workspace):
    review.do_export()
    rows = read_sheet(review.WORKBOOK)
    h = header_index(rows)
    rows[h + 1][2] = "Pyetje"
    rows[h + 1][3] = "Neni 120"           # author corrects the pre-filled label
    rows[h + 1][6] = "po"
    write_sheet(review.WORKBOOK, rows)

    review.do_import()
    kept = json.loads(review.QUESTIONS.read_text(encoding="utf-8").splitlines()[0])
    assert kept["gold_article"] == "Neni 120"


def test_import_marks_out_of_scope_questions(workspace):
    review.do_export()
    rows = read_sheet(review.WORKBOOK)
    h = header_index(rows)
    rows[h + 1][2] = "Kush fiton zgjedhjet?"
    rows[h + 1][5] = "jo"                 # e_pergjigjshme
    rows[h + 1][6] = "po"
    write_sheet(review.WORKBOOK, rows)

    review.do_import()
    kept = json.loads(review.QUESTIONS.read_text(encoding="utf-8").splitlines()[0])
    assert kept["answerable"] is False


def test_import_carries_source_url_through(workspace):
    review.do_export()
    rows = read_sheet(review.WORKBOOK)
    h = header_index(rows)
    rows[h + 1][2] = "Pyetje"
    rows[h + 1][6] = "po"
    write_sheet(review.WORKBOOK, rows)

    review.do_import()
    kept = json.loads(review.QUESTIONS.read_text(encoding="utf-8").splitlines()[0])
    assert kept["source_url"] == "http://x/1"


def test_import_rejects_a_sheet_without_a_header(workspace):
    review.do_export()
    write_sheet(review.WORKBOOK, [["# vetëm udhëzime"], ["asgjë"]])
    with pytest.raises(SystemExit):
        review.do_import()
