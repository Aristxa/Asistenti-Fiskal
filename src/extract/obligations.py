"""Extract obligations from the corpus into a structured, citable table.

Retrieval answers *what does the law say*. It cannot answer *what do I owe, by
when, at what rate, and what happens if I am late* — those need the duties pulled
out of prose and put into rows. That is information extraction, and it is the
part that makes this an accountant's tool rather than a search box.

**Rule-based, deliberately.** A language model would extract more, but every row
would be unverifiable and unreproducible, and the thesis would have to defend
extractions nobody can audit. Patterns are auditable: each row records which
pattern matched and the exact span, so a wrong row can be traced to the rule that
produced it and the rule can be fixed. Recall is lower and precision is checkable,
which is the right trade for a professional tool that cites its sources.

Every row carries its source article. An obligation without a citation is exactly
the kind of confident, unverifiable claim this whole system exists to avoid.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROCESSED = Path("data/processed")
OUTPUT = PROCESSED / "obligations.jsonl"

MONTHS = ("janar", "shkurt", "mars", "prill", "maj", "qershor", "korrik",
          "gusht", "shtator", "tetor", "nëntor", "dhjetor")

# --- who is obliged -------------------------------------------------------
# Ordered: the most specific subject wins, because "punëdhënësi" and "personi i
# tatueshëm" can both appear in one article and the first is the actual duty-holder.
SUBJECTS: tuple[tuple[str, str], ...] = (
    ("punëdhënësi", r"(?i)\bpun[ëe]dh[ëe]n[ëe]s"),
    ("punëmarrësi", r"(?i)\bpun[ëe]marr[ëe]s"),
    ("i vetëpunësuari", r"(?i)\bi?\s*vet[ëe]pun[ëe]suar"),
    ("personi i tatueshëm", r"(?i)\bperson[a-zëç]*\s+i?\s*tatuesh[ëe]m"),
    ("tatimpaguesi", r"(?i)\btatimpagues"),
    ("subjekti i TVSH-së", r"(?i)\bsubjekt[a-zëç]*\s+i\s+TVSH"),
    ("njësia ekonomike", r"(?i)\bnj[ëe]si[a-zëç]*\s+ekonomike"),
    ("organi tatimor", r"(?i)\borgan[a-zëç]*\s+tatimor"),
    ("shoqëria tregtare", r"(?i)\bshoq[ëe]ri[a-zëç]*\s+tregtare"),
)

# --- the duty itself ------------------------------------------------------
DUTY = re.compile(
    r"(?i)\b(detyroh[ea]n?|[ëe]sht[ëe]\s+i\s+detyruar|duhet\s+t[ëe]|ka\s+detyrimin)\b"
)

# --- deadlines ------------------------------------------------------------
RELATIVE_DEADLINE = re.compile(
    r"(?i)\b(?:brenda|jo\s+m[ëe]\s+von[ëe]\s+se)\s+"
    r"(\d+|nj[ëe]|dy|tre|tri|kat[ëe]r|pes[ëe]|gjasht[ëe]|shtat[ëe]|tet[ëe]|n[ëe]nt[ëe]|dhjet[ëe])"
    r"\s*(dit[ëe]ve?|dit[ëe]sh|muaj[ie]?v?e?|jav[ëe]ve?|vite?ve?)"
)
ABSOLUTE_DEADLINE = re.compile(
    r"(?i)\b(?:deri\s+m[ëe]|jo\s+m[ëe]\s+von[ëe]\s+se|brenda\s+dat[ëe]s)\s+"
    r"(\d{1,2})\s+(" + "|".join(MONTHS).replace("ë", "[ëe]") + r")"
)

# --- rates and amounts ----------------------------------------------------
RATE = re.compile(r"\b(\d{1,2}(?:[.,]\d+)?)\s?%")
AMOUNT = re.compile(r"(?i)\b(\d[\d\.\s]{2,}\d|\d)\s*(lek[ëe]|milion[ëe]?\s+lek[ëe]?)")

# --- penalties ------------------------------------------------------------
PENALTY = re.compile(
    r"(?i)\b(gjob[ëeaèt]{1,3}|sanksion[a-zëç]*|d[ëe]nohet|d[ëe]nim\s+me|"
    r"kamat[ëevo]{1,3}|interes[a-zëç]*\s+vonese)\b"
)

# --- forms ----------------------------------------------------------------
FORM = re.compile(r"(?i)\b(formular[a-zëç]*(?:\s+nr\.?\s*\d+)?|deklarat[ëeaèn]{1,3}[a-zëç]*)\b")

WORD_NUMBERS = {
    "një": 1, "nje": 1, "dy": 2, "tre": 3, "tri": 3, "katër": 4, "kater": 4,
    "pesë": 5, "pese": 5, "gjashtë": 6, "gjashte": 6, "shtatë": 7, "shtate": 7,
    "tetë": 8, "tete": 8, "nëntë": 9, "nente": 9, "dhjetë": 10, "dhjete": 10,
}

UNIT_DAYS = {"dit": 1, "jav": 7, "muaj": 30, "vit": 365}


@dataclass
class Obligation:
    obligation_id: str
    doc_id: int
    cite: str
    heading: str
    domain: str
    authority: str
    subject: str | None
    duty_span: str
    deadline_text: str | None
    deadline_days: int | None
    deadline_date: str | None
    rates: list[str] = field(default_factory=list)
    amounts: list[str] = field(default_factory=list)
    penalty: str | None = None
    form: str | None = None
    matched_by: list[str] = field(default_factory=list)
    source_url: str = ""


def _context(text: str, match: re.Match, before: int = 160, after: int = 260) -> str:
    """Window around a match, snapped to word boundaries.

    Cutting at a fixed offset leaves fragments like "tueshëm për TVSH-në", which
    read as extraction errors even when the row is correct.
    """
    start = max(0, match.start() - before)
    end = min(len(text), match.end() + after)
    if start > 0:
        space = text.find(" ", start)
        if space != -1 and space < match.start():
            start = space + 1
    if end < len(text):
        space = text.rfind(" ", match.end(), end)
        if space != -1:
            end = space
    return re.sub(r"\s+", " ", text[start:end]).strip()


def find_subject(window: str) -> str | None:
    for label, pattern in SUBJECTS:
        if re.search(pattern, window):
            return label
    return None


def parse_relative_deadline(match: re.Match) -> tuple[str, int | None]:
    raw = re.sub(r"\s+", " ", match.group(0)).strip()
    quantity = match.group(1).lower()
    number = int(quantity) if quantity.isdigit() else WORD_NUMBERS.get(quantity)
    unit = match.group(2).lower()
    days = None
    if number is not None:
        for stem, factor in UNIT_DAYS.items():
            if unit.startswith(stem):
                days = number * factor
                break
    return raw, days


def parse_absolute_deadline(match: re.Match) -> tuple[str, str]:
    raw = re.sub(r"\s+", " ", match.group(0)).strip()
    day = int(match.group(1))
    # Month names are matched diacritic-insensitively, since the corpus writes both
    # `nëntor` and `nentor`.
    month_text = match.group(2).lower().replace("ë", "e")
    index = next(
        (i for i, name in enumerate(MONTHS, start=1)
         if name.replace("ë", "e") == month_text),
        None,
    )
    return raw, (f"--{index:02d}-{day:02d}" if index else raw)


def extract_from_chunk(chunk: dict, meta: dict) -> list[Obligation]:
    text = chunk["text"]
    out: list[Obligation] = []

    for index, duty in enumerate(DUTY.finditer(text)):
        window = _context(text, duty)
        matched = ["duty"]

        subject = find_subject(window)
        if subject:
            matched.append("subject")

        deadline_text = deadline_days = deadline_date = None
        relative = RELATIVE_DEADLINE.search(window)
        absolute = ABSOLUTE_DEADLINE.search(window)
        if relative:
            deadline_text, deadline_days = parse_relative_deadline(relative)
            matched.append("deadline_relative")
        elif absolute:
            deadline_text, deadline_date = parse_absolute_deadline(absolute)
            matched.append("deadline_absolute")

        rates = [m.group(0) for m in RATE.finditer(window)]
        if rates:
            matched.append("rate")
        amounts = [re.sub(r"\s+", " ", m.group(0)) for m in AMOUNT.finditer(window)]
        if amounts:
            matched.append("amount")

        penalty_match = PENALTY.search(window)
        if penalty_match:
            matched.append("penalty")
        form_match = FORM.search(window)
        if form_match:
            matched.append("form")

        # A duty with no subject, no deadline, no rate and no penalty is just the
        # word "duhet" in a sentence. Requiring one concrete attribute is what keeps
        # the table useful rather than exhaustive.
        if len(matched) < 2:
            continue

        out.append(Obligation(
            obligation_id=f"{chunk['doc_id']}:{chunk.get('cite', '')}:{index}".replace(" ", ""),
            doc_id=chunk["doc_id"],
            cite=chunk.get("cite", ""),
            heading=chunk.get("heading", ""),
            domain=meta.get("domain", "tatime"),
            authority=meta.get("authority", ""),
            subject=subject,
            duty_span=window,
            deadline_text=deadline_text,
            deadline_days=deadline_days,
            deadline_date=deadline_date,
            rates=rates[:4],
            amounts=amounts[:4],
            penalty=penalty_match.group(0) if penalty_match else None,
            form=form_match.group(0) if form_match else None,
            matched_by=matched,
            source_url=meta.get("url", ""),
        ))

    return out


def build() -> None:
    documents = {}
    for line in (PROCESSED / "documents.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            documents[row["doc_id"]] = row

    rows: list[Obligation] = []
    for line in (PROCESSED / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        if chunk["strategy"] != "article":
            continue                      # citations must be article-level
        meta = documents.get(chunk["doc_id"], {})
        if meta.get("archival"):
            continue                      # never surface repealed duties
        rows.extend(extract_from_chunk(chunk, meta))

    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    _report(rows)
    print(f"\nshkruar: {OUTPUT}")


def _report(rows: list[Obligation]) -> None:
    from collections import Counter

    print(f"detyrime të nxjerra: {len(rows)}")
    print(f"  me subjekt të identifikuar : {sum(1 for r in rows if r.subject)}")
    print(f"  me afat                    : {sum(1 for r in rows if r.deadline_text)}")
    print(f"    afat i llogaritur në ditë: {sum(1 for r in rows if r.deadline_days)}")
    print(f"  me normë                   : {sum(1 for r in rows if r.rates)}")
    print(f"  me sanksion                : {sum(1 for r in rows if r.penalty)}")
    print(f"  me formular                : {sum(1 for r in rows if r.form)}")

    print("\n  sipas domenit:")
    for domain, count in Counter(r.domain for r in rows).most_common():
        print(f"    {domain:<14} {count:>5}")

    print("\n  subjektet më të shpeshta:")
    for subject, count in Counter(r.subject for r in rows if r.subject).most_common(6):
        print(f"    {subject:<24} {count:>5}")


if __name__ == "__main__":
    build()
