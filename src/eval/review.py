"""Export candidates to a spreadsheet for review, and read the result back.

The gold labels are tax judgements, so the review step belongs to the author, not
to a script. Asking someone to hand-edit 120 rows of JSONL would make the format
the obstacle rather than the work. This exports a CSV that opens directly in
Excel — with a UTF-8 BOM, so Albanian diacritics survive Excel's default
encoding guess — and reads the filled-in file back into the benchmark.

    python -m src.eval.review export     # candidates -> eval/gold/rishikim.csv
    python -m src.eval.review import     # rishikim.csv -> eval/gold/questions.jsonl

The round trip is deliberately lossless in one direction only: `import` trusts the
spreadsheet for the question, the gold article and the flags, and carries
everything else through unchanged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

GOLD = Path("eval/gold")
CANDIDATES = GOLD / "candidates.jsonl"
WORKBOOK = GOLD / "rishikim.csv"
QUESTIONS = GOLD / "questions.jsonl"

COLUMNS = [
    "id",
    "kategoria",
    "pyetja",              # to be written by the author
    "neni_gold",           # pre-filled; correct if wrong
    "dokumenti",
    "e_pergjigjshme",      # po / jo
    "rishikuar",           # po / jo
    "shenime",
    "titulli_i_nenit",     # read-only context
    "teksti_i_nenit",      # read-only context
]

INSTRUCTIONS = [
    ["# UDHËZIME — mos e fshij këtë rresht, importi e kapërcen"],
    ["# 1. Lexo 'titulli_i_nenit' dhe 'teksti_i_nenit' në të djathtë."],
    ["# 2. Shkruaj te 'pyetja' atë që do ta pyeste një tatimpagues i zakonshëm."],
    ["#    MOS kopjo fjalët e nenit — përdor fjalor të thjeshtë, ndryshe testi "
     "matet i lehtë dhe s'provon asgjë."],
    ["# 3. Kontrollo 'neni_gold': a përgjigjet vërtet ky nen? Nëse jo, korrigjoje."],
    ["# 4. Vendos 'rishikuar' = po kur rreshti është gati."],
    ["# 5. Nëse neni nuk jep pyetje të dobishme, lëre 'rishikuar' = jo; do të hiqet."],
    [],
]


def load_candidates() -> list[dict]:
    if not CANDIDATES.exists():
        raise SystemExit(
            f"nuk ka {CANDIDATES} — ekzekuto `python -m src.eval.make_candidates`"
        )
    return [
        json.loads(line)
        for line in CANDIDATES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def do_export() -> None:
    rows = load_candidates()
    GOLD.mkdir(parents=True, exist_ok=True)

    # utf-8-sig: Excel assumes the system codepage without a BOM and mangles ë/ç.
    with WORKBOOK.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        for line in INSTRUCTIONS:
            writer.writerow(line)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([
                row["id"],
                row["category"],
                row.get("question", ""),
                row.get("gold_article", ""),
                row.get("gold_doc_id", ""),
                "po" if row.get("answerable", True) else "jo",
                "po" if row.get("reviewed") else "jo",
                "",
                row.get("source_heading", ""),
                (row.get("source_text", "") or "")[:900],
            ])

    print(f"shkruar: {WORKBOOK}  ({len(rows)} rreshta)")
    print("Hape me Excel, plotëso kolonën 'pyetja', pastaj:")
    print("  python -m src.eval.review import")


def do_import() -> None:
    if not WORKBOOK.exists():
        raise SystemExit(f"nuk ka {WORKBOOK} — ekzekuto `export` më parë")

    by_id = {row["id"]: row for row in load_candidates()}

    # Parse first, then find the header row. Filtering raw lines on a leading "#"
    # does not work: instruction lines contain semicolons, so csv quotes them and
    # they begin with a quote character instead. Locating the header after parsing
    # is robust to whatever Excel does to the preamble on save.
    with WORKBOOK.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.reader(handle, delimiter=";"))

    header_index = next(
        (i for i, row in enumerate(raw) if row and row[0].strip() == "id"), None
    )
    if header_index is None:
        raise SystemExit(
            f"{WORKBOOK}: nuk u gjet rreshti i kokës (kolona e parë duhet 'id')"
        )

    header = [cell.strip() for cell in raw[header_index]]
    records = [
        dict(zip(header, row + [""] * (len(header) - len(row))))
        for row in raw[header_index + 1:]
        if row and any(cell.strip() for cell in row)
    ]

    kept, skipped_unreviewed, skipped_empty, unknown = [], 0, 0, 0
    for record in records:
        row_id = (record.get("id") or "").strip()
        if not row_id:
            continue
        source = by_id.get(row_id)
        if source is None:
            unknown += 1
            continue
        if (record.get("rishikuar") or "").strip().lower() != "po":
            skipped_unreviewed += 1
            continue
        question = (record.get("pyetja") or "").strip()
        if not question:
            skipped_empty += 1
            continue

        kept.append({
            "id": row_id,
            "question": question,
            "gold_doc_id": source["gold_doc_id"],
            "gold_article": (record.get("neni_gold") or "").strip()
                            or source["gold_article"],
            "category": source["category"],
            "answerable": (record.get("e_pergjigjshme") or "po").strip().lower() == "po",
            "notes": (record.get("shenime") or "").strip(),
            "source_url": source.get("source_url", ""),
            "drafted_by_model": bool(source.get("drafted_by_model")),
        })

    with QUESTIONS.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    answerable = sum(1 for r in kept if r["answerable"])
    print(f"shkruar: {QUESTIONS}")
    print(f"  pyetje të pranuara     : {len(kept)}")
    print(f"    të përgjigjshme      : {answerable}")
    print(f"    jashtë teme          : {len(kept) - answerable}")
    print(f"  të parishikuara        : {skipped_unreviewed}")
    print(f"  të rishikuara pa pyetje: {skipped_empty}")
    if unknown:
        print(f"  id të panjohura        : {unknown}")

    if answerable < 60:
        print(f"\n  Minimumi i mbrojtshëm është 60 pyetje të përgjigjshme "
              f"(ke {answerable}).")
    if len(kept) - answerable == 0:
        print("\n  Nuk ka pyetje jashtë teme. Pa to nuk matet dot refuzimi —")
        print("  një sistem që refuzon gjithçka do të dilte i përsosur.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Rishikimi i bazës së testimit")
    parser.add_argument("action", choices=["export", "import"])
    args = parser.parse_args()

    if args.action == "export":
        do_export()
    else:
        do_import()


if __name__ == "__main__":
    main()
