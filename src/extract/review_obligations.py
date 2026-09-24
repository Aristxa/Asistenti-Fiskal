"""Export a sample of extracted obligations for hand-checking.

Precision of a rule-based extractor is a measurable quantity, not an impression.
This draws a reproducible random sample, stratified so the interesting rows —
those carrying a deadline, a rate or a penalty — are actually represented, and
writes a spreadsheet with one judgement column.

    python -m src.extract.review_obligations export
    python -m src.extract.review_obligations score
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

PROCESSED = Path("data/processed")
SOURCE = PROCESSED / "obligations.jsonl"
WORKBOOK = Path("eval/gold/detyrimet_rishikim.csv")

SAMPLE_SIZE = 60
SEED = 20260827

COLUMNS = ["obligation_id", "e_sakte", "shenime", "domeni", "neni", "subjekti",
           "afati", "norma", "sanksioni", "teksti"]


def load() -> list[dict]:
    if not SOURCE.exists():
        raise SystemExit(f"nuk ka {SOURCE} — ekzekuto `python -m src.extract.obligations`")
    return [json.loads(l) for l in SOURCE.read_text(encoding="utf-8").splitlines() if l.strip()]


def sample(rows: list[dict]) -> list[dict]:
    """Stratified: half from rows with a concrete attribute, half from the rest.

    A uniform sample would be dominated by bare duty statements and would not
    measure precision where it matters.
    """
    rng = random.Random(SEED)
    rich = [r for r in rows if r["deadline_text"] or r["rates"] or r["penalty"]]
    plain = [r for r in rows if r not in rich]
    half = SAMPLE_SIZE // 2
    picked = rng.sample(rich, min(half, len(rich)))
    picked += rng.sample(plain, min(SAMPLE_SIZE - len(picked), len(plain)))
    rng.shuffle(picked)
    return picked


def article_texts() -> dict[tuple[int, str], str]:
    """Full text of every article, keyed by (doc_id, cite).

    Long articles are stored as several chunks under one citation, so the parts are
    concatenated in stored order. Taking the first part instead would hand the
    reviewer a fragment — and judging an extraction against a fragment is exactly
    the situation that made the earlier question review unworkable.
    """
    index = PROCESSED / "index/article/chunks.jsonl"
    if not index.exists():
        return {}
    parts: dict[tuple[int, str], list[tuple[int, str]]] = {}
    for line in index.open(encoding="utf-8"):
        if not line.strip():
            continue
        chunk = json.loads(line)
        key = (chunk["doc_id"], chunk.get("cite") or chunk.get("label", ""))
        parts.setdefault(key, []).append((chunk.get("order", 0), chunk["text"]))
    return {k: "\n".join(t for _, t in sorted(v)) for k, v in parts.items()}


def existing_judgements() -> dict[str, tuple[str, str]]:
    """Keep any judgement already made, so re-exporting never destroys work."""
    if not WORKBOOK.exists():
        return {}
    with WORKBOOK.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.reader(handle, delimiter=";"))
    head = next((i for i, r in enumerate(raw) if r and r[0].strip() == "obligation_id"), None)
    if head is None:
        return {}
    header = [c.strip() for c in raw[head]]
    kept = {}
    for row in raw[head + 1:]:
        if not row or not row[0].strip():
            continue
        record = dict(zip(header, row + [""] * (len(header) - len(row))))
        verdict = (record.get("e_sakte") or "").strip()
        if verdict:
            kept[record["obligation_id"]] = (verdict, (record.get("shenime") or "").strip())
    return kept


def do_export() -> None:
    rows = sample(load())
    articles = article_texts()
    previous = existing_judgements()
    WORKBOOK.parent.mkdir(parents=True, exist_ok=True)
    missing = 0

    with WORKBOOK.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["# Shëno 'po' te e_sakte nëse rreshti përshkruan vërtet një"])
        writer.writerow(["# detyrim real dhe atributet e nxjerra janë të sakta. Ndryshe 'jo'."])
        writer.writerow([])
        writer.writerow(COLUMNS)
        for row in rows:
            verdict, note = previous.get(row["obligation_id"], ("", ""))
            # Whole article, never a truncated span.
            text = articles.get((row["doc_id"], row["cite"]), "") or row["duty_span"]
            if (row["doc_id"], row["cite"]) not in articles:
                missing += 1
            writer.writerow([
                row["obligation_id"], verdict, note, row["domain"], row["cite"],
                row["subject"] or "", row["deadline_text"] or "",
                ", ".join(row["rates"]), row["penalty"] or "",
                text,
            ])

    print(f"shkruar: {WORKBOOK}  ({len(rows)} rreshta për rishikim)")
    if previous:
        print(f"  u ruajtën {len(previous)} vlerësime ekzistuese")
    if missing:
        print(f"  {missing} rreshta pa nen në indeks — u përdor vetëm pjesa e nxjerrë")


def do_score() -> None:
    if not WORKBOOK.exists():
        raise SystemExit(f"nuk ka {WORKBOOK} — ekzekuto `export` më parë")
    with WORKBOOK.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.reader(handle, delimiter=";"))
    head = next(i for i, r in enumerate(raw) if r and r[0].strip() == "obligation_id")
    header = [c.strip() for c in raw[head]]
    records = [dict(zip(header, r + [""] * (len(header) - len(r))))
               for r in raw[head + 1:] if r and any(c.strip() for c in r)]

    judged = [r for r in records if (r.get("e_sakte") or "").strip().lower() in ("po", "jo")]
    correct = sum(1 for r in judged if r["e_sakte"].strip().lower() == "po")
    if not judged:
        print("asnjë rresht i vlerësuar ende")
        return
    print(f"saktësia e nxjerrjes: {correct}/{len(judged)} = "
          f"{100 * correct / len(judged):.0f}%")
    print(f"  të pavlerësuara: {len(records) - len(judged)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["export", "score"])
    args = parser.parse_args()
    do_export() if args.action == "export" else do_score()
