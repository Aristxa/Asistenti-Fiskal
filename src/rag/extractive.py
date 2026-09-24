"""Answer from the statute's own words, with no language model at all.

The generated layer writes fluent Albanian and needs a commercial API. This one
selects sentences from the retrieved articles and presents them verbatim. It is
worse prose and a stronger guarantee: **an extractive answer cannot hallucinate,
by construction**, because every word shown was written by the legislator. The
citation contract stops being a rule the model is asked to follow and becomes a
property of the method.

That makes it the right default for a system that must keep working when the API
key is empty, and an interesting comparison in its own right: fluency and
faithfulness are separable, and here they are separated on purpose.

Selection is deliberately simple — overlap between the question's terms and the
sentence's, over the Albanian tokeniser that already folds diacritics and strips
inflection. A learned sentence ranker would be another model to justify, and the
thesis claim is about retrieval, not about summarisation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.index.albanian import tokenise

# Sentence boundaries in Albanian legal text.
#
# A naive split on "." shatters this register: it is dense with "nr.", "datë",
# "neni", "pika", "lek/m2" and numbered lists like "1.4". Splitting only at a
# terminator followed by whitespace and a capital letter or a list marker keeps
# those intact, and legal drafting is regular enough for that to hold.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZËÇ«\"]|\d+[.)]\s)")

# Fragments this short are headings, numbering artefacts or dangling clauses.
MIN_SENTENCE_CHARS = 40

# Deadlines, rates and money are what a taxpayer is actually looking for, and a
# sentence carrying one answers the question more often than a sentence of equal
# term overlap that does not.
CONCRETE = re.compile(
    r"\b\d+\s*(dit[ëe]|muaj|vit|vjet|%|përqind|lek[ëe]?)\b"
    r"|\bbrenda\b|\bderi më\b|\bafat", re.I
)


@dataclass
class Passage:
    cite: str
    heading: str
    doc_id: int
    sentences: list[str]
    rank: int

    @property
    def url(self) -> str:
        return f"https://www.tatime.gov.al/shkarko.php?id={self.doc_id}"


def split_sentences(text: str) -> list[str]:
    parts = [" ".join(p.split()) for p in SENTENCE_SPLIT.split(text or "")]
    return [p for p in parts if len(p) >= MIN_SENTENCE_CHARS]


def score_sentence(sentence: str, query_terms: set[str]) -> float:
    """Share of the question's terms present, with a bonus for concrete facts."""
    if not query_terms:
        return 0.0
    terms = set(tokenise(sentence))
    if not terms:
        return 0.0
    overlap = len(query_terms & terms) / len(query_terms)
    return overlap + (0.15 if CONCRETE.search(sentence) else 0.0)


def select(question: str, hits, per_article: int = 2,
           max_articles: int = 4, min_score: float = 0.08) -> list[Passage]:
    """Pick the sentences of each retrieved article that address the question.

    Sentences are returned in the order the statute states them, not in score
    order: legal provisions carry conditions before consequences, and reordering
    them changes what they say.
    """
    query_terms = set(tokenise(question))
    passages: list[Passage] = []

    for hit in hits[:max_articles]:
        chunk = hit.chunk
        scored = [
            (score_sentence(s, query_terms), i, s)
            for i, s in enumerate(split_sentences(chunk.get("text", "")))
        ]
        scored = [t for t in scored if t[0] >= min_score]
        if not scored:
            continue
        scored.sort(key=lambda t: t[0], reverse=True)
        chosen = sorted(scored[:per_article], key=lambda t: t[1])
        passages.append(Passage(
            cite=chunk.get("cite") or chunk.get("label") or "",
            heading=chunk.get("heading") or "",
            doc_id=chunk["doc_id"],
            sentences=[s for _, _, s in chosen],
            rank=hit.rank,
        ))

    return passages


def format_answer(question: str, hits) -> str:
    """Render an extractive answer in Albanian, quoting the statute verbatim."""
    passages = select(question, hits)
    if not passages:
        return (
            "Nuk gjeta një fjali që t'i përgjigjet drejtpërdrejt kësaj pyetjeje "
            "në nenet e gjetura. Nenet më të afërta janë renditur më poshtë — "
            "lexoji drejtpërdrejt."
        )

    lines = [
        "**Çdo fjali më poshtë është fjalë për fjalë nga neni i cituar**, pa "
        "riformulim — pra e verifikueshme drejtpërdrejt te burimi.",
        "",
    ]
    for p in passages:
        title = f"{p.cite} — {p.heading}" if p.heading else p.cite
        lines.append(f"### {title}  [S{p.rank}]")
        for sentence in p.sentences:
            lines.append(f"> {sentence}")
        lines.append(f"[Shkarko dokumentin zyrtar]({p.url})")
        lines.append("")

    lines.append("---")
    lines.append(
        "_Kjo mënyrë nuk e përmbledh ligjin dhe nuk e shpjegon me fjalë të tjera; "
        "ajo tregon se ku përgjigjja gjendet dhe ta jep të pandryshuar._"
    )
    return "\n".join(lines)
