"""Query the extracted obligations as a table.

The retrieval layer answers "what does the law say about X". It cannot answer
"which of my duties fall due in under fifteen days", because that is not a
similarity question — it is a filter over structured fields, and no amount of
embedding quality produces it.

That is what this module is for. The obligations table already exists
(`src/extract/obligations.py`, 2,077 rows, precision measured at 88% in IV.7);
until now nothing in the running system used it.

Coverage is deliberately reported alongside every result. Only 8.1% of rows carry
an extracted deadline and 2.9% a penalty, so a filtered view is a view of what the
extractor *found*, never of what the law contains. Presenting it as the latter
would be the same overclaim the citation contract exists to prevent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SOURCE = Path("data/processed/obligations.jsonl")

# Subjects worth offering as a filter. Taken from the extracted data by frequency
# rather than invented, so the list cannot drift away from what is really there.
SUBJECTS = [
    "tatimpaguesi",
    "njësia ekonomike",
    "personi i tatueshëm",
    "punëdhënësi",
    "punëmarrësi",
    "organi tatimor",
    "i vetëpunësuari",
    "shoqëria tregtare",
]

DOMAINS = {
    "tatime": "Tatime",
    "kontabilitet": "Kontabilitet",
    "pune": "Marrëdhëniet e punës",
}

# A penalty field holding a bare verb ("sanksionon", "dënohet") is the extractor
# catching the grammar of a sanction without its content. Those rows are still
# real obligations, but they must not be offered under a "has a penalty" filter as
# though a penalty had been identified.
BARE_VERBS = {"sanksionon", "sanksionohet", "dënohet", "denohet", "ndëshkohet"}


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    doc_id: int
    cite: str
    heading: str
    domain: str
    authority: str
    subject: str | None
    text: str
    deadline_text: str | None
    deadline_days: int | None
    rates: tuple[str, ...]
    penalty: str | None
    form: str | None

    @property
    def url(self) -> str:
        return f"https://www.tatime.gov.al/shkarko.php?id={self.doc_id}"

    @property
    def has_real_penalty(self) -> bool:
        if not self.penalty:
            return False
        return self.penalty.strip().lower() not in BARE_VERBS


@lru_cache(maxsize=1)
def load() -> tuple[Obligation, ...]:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"nuk ka {SOURCE} — ekzekuto `python -m src.extract.obligations`"
        )
    rows = []
    for line in SOURCE.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(Obligation(
            obligation_id=d["obligation_id"],
            doc_id=d["doc_id"],
            cite=d.get("cite") or "",
            heading=d.get("heading") or "",
            domain=d.get("domain") or "",
            authority=d.get("authority") or "",
            subject=d.get("subject"),
            text=" ".join((d.get("duty_span") or "").split()),
            deadline_text=d.get("deadline_text"),
            deadline_days=d.get("deadline_days"),
            rates=tuple(d.get("rates") or ()),
            penalty=d.get("penalty"),
            form=d.get("form"),
        ))
    return tuple(rows)


def coverage() -> dict[str, tuple[int, int]]:
    """How many rows carry each attribute. Shown in the UI, never hidden."""
    rows = load()
    n = len(rows)
    return {
        "subjekt": (sum(1 for r in rows if r.subject), n),
        "afat": (sum(1 for r in rows if r.deadline_text), n),
        "normë": (sum(1 for r in rows if r.rates), n),
        "sanksion": (sum(1 for r in rows if r.has_real_penalty), n),
        "formular": (sum(1 for r in rows if r.form), n),
    }


def search(
    domain: str | None = None,
    subject: str | None = None,
    with_deadline: bool = False,
    max_days: int | None = None,
    with_penalty: bool = False,
    text: str | None = None,
    limit: int = 50,
) -> list[Obligation]:
    """Filter the obligations table.

    Sorted so the most actionable rows come first: a known deadline, shortest
    first, then everything else. A user asking for duties with deadlines wants the
    urgent ones at the top, not the ones that happen to be alphabetically early.
    """
    rows = list(load())

    if domain:
        rows = [r for r in rows if r.domain == domain]
    if subject:
        rows = [r for r in rows if r.subject == subject]
    if with_deadline or max_days is not None:
        rows = [r for r in rows if r.deadline_text]
    if max_days is not None:
        rows = [r for r in rows if r.deadline_days is not None
                and r.deadline_days <= max_days]
    if with_penalty:
        rows = [r for r in rows if r.has_real_penalty]
    if text:
        # Diacritic-insensitive, because that is how people type Albanian — the
        # same finding that drove the lexical arm of retrieval (F8).
        from src.index.albanian import fold

        needle = fold(text.lower())
        rows = [
            r for r in rows
            # Subject is searched too. It has its own dropdown, but someone typing
            # "punëdhënësi" into a search box means it as a search term, and a box
            # that silently ignores the word is worse than no box.
            if needle in fold(
                f"{r.text} {r.heading} {r.cite} {r.subject or ''}".lower())
        ]

    def sort_key(r: Obligation):
        return (r.deadline_days is None, r.deadline_days or 0, r.cite)

    rows.sort(key=sort_key)

    # One row per article. The extractor emits a row per matched duty phrase, so a
    # single article can appear several times with the same deadline — three copies
    # of Neni 23 filled the top of a fifteen-day query during testing. That is
    # noise, not information, and it crowds out other articles exactly the way
    # duplicate chunks once crowded the retrieval results.
    #
    # The richest row wins, since the parts that carry a penalty or a rate are the
    # ones worth showing.
    def richness(r: Obligation) -> int:
        return sum((bool(r.deadline_text), bool(r.rates),
                    r.has_real_penalty, bool(r.subject), bool(r.form)))

    best: dict[tuple[int, str], Obligation] = {}
    for row in rows:
        key = (row.doc_id, row.cite)
        if key not in best or richness(row) > richness(best[key]):
            best[key] = row

    deduped = sorted(best.values(), key=sort_key)
    return deduped[:limit]


def summarise(rows: list[Obligation]) -> dict[str, int]:
    return {
        "gjithsej": len(rows),
        "me_afat": sum(1 for r in rows if r.deadline_text),
        "me_sanksion": sum(1 for r in rows if r.has_real_penalty),
        "me_norme": sum(1 for r in rows if r.rates),
    }


# Routing: which questions belong to the table rather than to retrieval.
#
# Kept as explicit patterns rather than a classifier. A wrong route sends a
# question to a layer that cannot answer it, and an opaque router would make that
# failure impossible to explain at a defence.
TABLE_PATTERNS = (
    re.compile(r"\bafat(e|et|eve)?\b.*\b(kam|kemi|ka|jan[ëe])\b", re.I),
    re.compile(r"\bcilat\b.*\bdetyrime\b", re.I),
    re.compile(r"\bçfar[ëe]\b.*\bdetyrime\b.*\bkam\b", re.I),
    re.compile(r"\bdetyrime(t|ve)?\b.*\bsi\b\s+(punëdhënës|tatimpagues)", re.I),
    re.compile(r"\bbrenda\b\s*\d+\s*dit", re.I),
    re.compile(r"\bku\b.*\bgjob[ëa]\b", re.I),
)


def looks_like_a_table_question(question: str) -> bool:
    """True when a question asks to enumerate duties rather than explain one."""
    return any(p.search(question or "") for p in TABLE_PATTERNS)
