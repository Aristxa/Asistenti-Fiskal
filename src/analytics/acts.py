"""Structured metadata over the acts themselves, and the questions it answers.

Retrieval finds a passage. It cannot count, group, or order — so it cannot answer
the question every accountant asks each January: *what changed this year, and
which of it affects me?* Those need the corpus described as a table rather than
searched as text.

Albanian legal titles are unusually regular:

    VKM Nr.783 datë 10.11.2011 "Për procedurat e ndarjes ..." i ndryshuar
    ^type   ^number ^date       ^subject                       ^amendment status

so the dimensions come out of the titles themselves — no annotation, no model.
Where a title does not parse, the field is left null rather than guessed, and the
coverage of each field is reported so the analytics never imply more certainty
than the metadata supports.

This is the star-schema idea applied to legislation instead of financial facts:
one row per act, dimensions for type, year, authority and domain, plus an
amendment edge list connecting acts to the acts they modify.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

PROCESSED = Path("data/processed")
OUTPUT = PROCESSED / "acts.jsonl"

# Ordered longest-first: "Akt normativ" must win over "Akt", and the compound
# ministerial forms ("UMF dhe MD") over the bare one.
ACT_TYPES: tuple[tuple[str, str], ...] = (
    ("Akt normativ", r"(?i)\bakt\s+normativ"),
    ("Udhëzim", r"(?i)\b(udh[ëe]zim|umf)\b"),
    ("VKM", r"(?i)\b(vkm|vendim\s+i?\s*k[ëe]shillit|vendim)\b"),
    ("Ligj", r"(?i)\bligj"),
    ("Urdhër", r"(?i)\burdh[ëe]r"),
    ("Marrëveshje", r"(?i)\bmarr[ëe]veshje|konvent"),
    ("Standard", r"(?i)\b(skk|snrf|snk)\b|standard"),
    ("Kod", r"(?i)\bkodi\b"),
)

# Some acts carry no type marker in the title at all. The double-taxation treaties
# are titled with nothing but the counterparty country — "Austri", "Franca",
# "Greqi" — so the type has to come from where the document was found rather than
# from what it is called. Falling back to the crawl category recovers them.
CATEGORY_TYPES: dict[str, str] = {
    "marreveshje-nderkombetare": "Marrëveshje",
    "marreveshje-kombetare": "Marrëveshje",
    "kkk": "Standard",
    "akte-te-dpt": "Akt i DPT",
    "pune": "Legjislacion pune",
}

# `Nr.783`, `Nr. 92/2014`, `nr 25/2018`
ACT_NUMBER = re.compile(r"(?i)\bnr\.?\s*(\d+)(?:\s*/\s*(\d{4}))?")

# `datë 10.11.2011`, `date 4.01.2012`
ACT_DATE = re.compile(r"(?i)\bdat[ëe]\s*(\d{1,2})[./](\d{1,2})[./](\d{4})")

# A year appearing on its own, used only when no full date parses.
BARE_YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")

AMENDED = re.compile(r"(?i)\bi\s+ndryshuar\b|\bt[ëe]\s+ndryshuar\b")
IS_AMENDMENT = re.compile(r"(?i)p[ëe]r\s+disa\s+ndryshime|ndryshime\s+n[ëe]|^ndryshim")
IS_REPEAL = re.compile(r"(?i)^shfuqizim|shfuqizohet")

# "…në ligjin nr.9920 datë 19.5.2008" — the act this one modifies.
#
# Only the accusative and genitive definite forms count. The nominative (`ligji`,
# `VKM`) is how a title *names itself*: in "VKM Nr.783 datë 10.11.2011" the number
# is the act's own. Including nominative forms made every ordinary title look like
# a reference to another act, which erased its own number.
REFERENCE = re.compile(
    r"(?i)\b(?:ligjin|ligjit|vendimin|vendimit|udh[ëe]zimin|udh[ëe]zimit|aktin|aktit)\s+"
    r"nr\.?\s*(\d+)"
)

QUOTED_SUBJECT = re.compile(r"[\"“”«]([^\"“”«»]{10,200})[\"“”»]")


@dataclass
class Act:
    doc_id: int
    act_type: str | None
    number: int | None
    year: int | None
    date: str | None
    subject: str
    domain: str
    authority: str
    is_amended: bool          # carries "i ndryshuar"
    is_amendment: bool        # is itself an amending act
    is_repeal: bool
    amends: list[int]         # act numbers this one references
    n_articles: int
    regime: str
    chars: int
    url: str


def classify(title: str, category: str = "") -> str | None:
    for label, pattern in ACT_TYPES:
        if re.search(pattern, title):
            return label
    return CATEGORY_TYPES.get(category)


def parse_act(doc: dict) -> Act:
    title = doc.get("title") or ""

    # An act's own number and the numbers of acts it references look identical.
    # In an amending title the *only* number is often the target's -- "Për disa
    # ndryshime në ligjin nr.9920" -- so a naive read assigns the target's number
    # to this act and then drops the amendment edge as a self-reference. Spans
    # that belong to a reference are therefore excluded when looking for the
    # act's own number.
    reference_spans = [m.span() for m in REFERENCE.finditer(title)]

    def inside_reference(position: int) -> bool:
        return any(start <= position < end for start, end in reference_spans)

    number = year = None
    for match in ACT_NUMBER.finditer(title):
        if inside_reference(match.start()):
            continue
        number = int(match.group(1))
        if match.group(2):
            year = int(match.group(2))
        break

    date = None
    date_match = ACT_DATE.search(title)
    if date_match:
        day, month, full_year = (int(g) for g in date_match.groups())
        if 1 <= day <= 31 and 1 <= month <= 12:
            date = f"{full_year:04d}-{month:02d}-{day:02d}"
            year = full_year
    if year is None:
        years = [int(y) for y in BARE_YEAR.findall(title)]
        # Titles cite other acts' years too; the earliest is the safer guess for
        # the act's own vintage only when nothing better exists.
        year = min(years) if years else None

    subject_match = QUOTED_SUBJECT.search(title)
    subject = (subject_match.group(1) if subject_match else title).strip()

    # A reference to the act's own number is not an amendment edge.
    references = [int(n) for n in REFERENCE.findall(title)]
    amends = sorted({n for n in references if n != number})

    return Act(
        doc_id=doc["doc_id"],
        act_type=classify(title, doc.get("category", "")),
        number=number,
        year=year,
        date=date,
        subject=subject[:200],
        domain=doc.get("domain", "tatime"),
        authority=doc.get("authority", ""),
        is_amended=bool(AMENDED.search(title)),
        is_amendment=bool(IS_AMENDMENT.search(title)),
        is_repeal=bool(IS_REPEAL.search(title)),
        amends=amends,
        n_articles=doc.get("n_articles", 0),
        regime=doc.get("regime", ""),
        chars=doc.get("chars", 0),
        url=doc.get("url", ""),
    )


def load_documents() -> list[dict]:
    path = PROCESSED / "documents.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build() -> list[Act]:
    acts = [parse_act(doc) for doc in load_documents()]
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for act in acts:
            handle.write(json.dumps(asdict(act), ensure_ascii=False) + "\n")
    return acts


# --------------------------------------------------------------------------
# The questions retrieval cannot answer
# --------------------------------------------------------------------------

def coverage(acts: list[Act]) -> None:
    total = len(acts)
    print(f"akte: {total}")
    for label, predicate in (
        ("lloji i njohur", lambda a: a.act_type),
        ("numri i aktit", lambda a: a.number),
        ("viti", lambda a: a.year),
        ("data e plotë", lambda a: a.date),
    ):
        n = sum(1 for a in acts if predicate(a))
        print(f"  {label:<18} {n:>4}/{total}  ({100 * n // total}%)")


def by_type(acts: list[Act]) -> None:
    print("\nsipas llojit:")
    for label, count in Counter(a.act_type or "(i paklasifikuar)" for a in acts).most_common():
        print(f"  {label:<20} {count:>4}")


def by_year(acts: list[Act], since: int = 2014) -> None:
    print(f"\nakte sipas vitit (nga {since}):")
    counts = Counter(a.year for a in acts if a.year and a.year >= since)
    if not counts:
        return
    peak = max(counts.values())
    for year in sorted(counts):
        bar = "█" * max(1, round(20 * counts[year] / peak))
        print(f"  {year}  {counts[year]:>3}  {bar}")


def by_domain(acts: list[Act]) -> None:
    print("\nsipas domenit dhe llojit:")
    grid: dict[str, Counter] = defaultdict(Counter)
    for act in acts:
        grid[act.domain][act.act_type or "?"] += 1
    for domain, counter in grid.items():
        top = ", ".join(f"{k} {v}" for k, v in counter.most_common(4))
        print(f"  {domain:<14} {sum(counter.values()):>4}  ({top})")


def amendment_activity(acts: list[Act]) -> None:
    print("\naktet më të referuara (objekt ndryshimesh):")
    referenced = Counter()
    for act in acts:
        for number in act.amends:
            referenced[number] += 1
    by_number = {a.number: a for a in acts if a.number}
    for number, count in referenced.most_common(8):
        target = by_number.get(number)
        name = target.subject[:52] if target else "(jashtë korpusit)"
        print(f"  nr.{number:<6} referuar {count:>2} herë  {name}")

    amending = sum(1 for a in acts if a.is_amendment)
    amended = sum(1 for a in acts if a.is_amended)
    repeals = sum(1 for a in acts if a.is_repeal)
    print(f"\n  akte që ndryshojnë të tjera : {amending}")
    print(f"  akte të shënuara 'i ndryshuar': {amended}")
    print(f"  shfuqizime                    : {repeals}")


def whats_new(acts: list[Act], year: int) -> None:
    print(f"\nçfarë ndryshoi në {year}:")
    rows = [a for a in acts if a.year == year]
    if not rows:
        print("  (asnjë akt i datuar në këtë vit)")
        return
    for act in sorted(rows, key=lambda a: (a.act_type or "", a.number or 0))[:12]:
        flag = " [ndryshim]" if act.is_amendment else ""
        print(f"  {(act.act_type or '?'):<12} nr.{act.number or '?':<6} {act.subject[:56]}{flag}")
    print(f"  gjithsej: {len(rows)} akte")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analitika mbi aktet")
    parser.add_argument("--year", type=int, help="çfarë ndryshoi në një vit")
    args = parser.parse_args()

    acts = build()
    coverage(acts)
    by_type(acts)
    by_domain(acts)
    by_year(acts)
    amendment_activity(acts)
    if args.year:
        whats_new(acts, args.year)
    print(f"\nshkruar: {OUTPUT}")


if __name__ == "__main__":
    main()
