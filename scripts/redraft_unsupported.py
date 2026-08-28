"""Re-draft only the questions whose form the source article cannot support.

The first drafting run assigned a question shape by rotation. On an article with
no penalty in it, a "what happens if you fail to" question has no answer; the
same holds for deadline questions against articles with no deadline and amount
questions against articles with no figure. Measured on the first set: 25 of 26
consequence-form questions, 20 of 26 deadline-form, 21 of 26 amount-form.

Re-drafting everything would discard review work already done. This re-drafts
only rows that are both unsupported and not yet reviewed, so the author's
judgements survive.

The detectors are deliberately crude and one-directional: they look for evidence
in the article that the question's premise exists. A question can be answerable
without matching them, so this over-selects rather than under-selects — the cost
of a needless re-draft is a fraction of a cent, the cost of leaving an
unanswerable question in the benchmark is a false failure in Chapter IV.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

# Run as a script rather than a module, so the project root is not on sys.path
# and `import src...` fails. Added here instead of requiring a PYTHONPATH the
# caller has to remember.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GOLD = Path("eval/gold")
CANDIDATES = GOLD / "candidates.jsonl"
WORKBOOK = GOLD / "rishikim.csv"

# (question opener, evidence the article must contain for it to be answerable)
FORM_REQUIREMENTS: tuple[tuple[str, str, str], ...] = (
    ("pasojë",
     r"(?i)^\s*[ÇC]far[ëe]\s+ndodh",
     r"(?i)\b(gjob|sanksion|d[ëe]noh|d[ëe]nim|kamat|p[ëe]rgjegj[ëe]si|shkelje"
     r"|kund[ëe]rvajtje|mas[ëa]\s+administrative|nuk\s+njihet|refuzoh|humb)"),
    ("afat",
     r"(?i)^\s*(Kur|Deri\s+kur)\b",
     r"(?i)\b(brenda|deri m[ëe]|afat|dat[ëe]s|\d+\s*dit|\d+\s*muaj|çdo\s+muaj|vjetor)"),
    ("shumë",
     r"(?i)^\s*Sa\b",
     r"(\d+\s?%|\blek[ëe]\b|\bshkall[ëe]\b|\bnorm[ëa]\b|\bp[ëe]rqindj)"),
)


def load_candidates() -> list[dict]:
    return [json.loads(l) for l in CANDIDATES.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def reviewed_ids() -> set[str]:
    if not WORKBOOK.exists():
        return set()
    with WORKBOOK.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    head = next(i for i, r in enumerate(rows) if r and r[0].strip() == "id")
    header = rows[head]
    ri, ii = header.index("rishikuar"), header.index("id")
    return {r[ii] for r in rows[head + 1:]
            if len(r) > ri and (r[ri] or "").strip().lower() == "po"}


def unsupported(row: dict) -> str | None:
    """Name of the form whose premise the article does not support, if any."""
    question = row.get("question") or ""
    text = row.get("source_text") or ""
    for name, opener, evidence in FORM_REQUIREMENTS:
        if re.search(opener, question) and not re.search(evidence, text):
            return name
    return None


def main() -> None:
    from src.eval.make_candidates import draft_questions, flag_paraphrases, report_overlap

    candidates = load_candidates()
    protected = reviewed_ids()

    targets = []
    kept_reviewed = 0
    for row in candidates:
        reason = unsupported(row)
        if not reason:
            continue
        if row["id"] in protected:
            kept_reviewed += 1
            continue
        row["_reason"] = reason
        targets.append(row)

    print(f"kandidate: {len(candidates)}")
    print(f"  pa mbështetje në nen : {len(targets) + kept_reviewed}")
    print(f"  tashmë të rishikuara (nuk preken): {kept_reviewed}")
    print(f"  do të ridraftohen    : {len(targets)}")
    if not targets:
        print("asgjë për të bërë")
        return

    from collections import Counter
    for reason, count in Counter(t["_reason"] for t in targets).most_common():
        print(f"    {reason:<10} {count}")

    # draft_questions only fills rows whose question is empty
    for row in targets:
        row["question"] = ""
        row.pop("_reason", None)

    print()
    draft_questions(targets)

    report_overlap(candidates)
    flag_paraphrases(candidates)

    with CANDIDATES.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nshkruar: {CANDIDATES}")

    still = [r for r in candidates if unsupported(r)]
    print(f"ende pa mbështetje: {len(still)} (ishin {len(targets) + kept_reviewed})")


if __name__ == "__main__":
    main()
