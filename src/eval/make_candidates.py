"""Propose candidate benchmark questions for the author to review.

The benchmark is the thesis's most durable contribution, and its gold labels are
tax-domain judgements — deciding a question is answered by *Neni 117* and not
*Neni 116* is not an engineering task. So this script never produces gold labels.
It selects articles worth asking about and prepares review slots; the author
supplies or corrects the question and confirms the article.

**The trap this script is designed around.** If candidate questions are written by
paraphrasing the article they come from, they inherit the article's distinctive
vocabulary, BM25 matches them trivially, and every configuration scores near 100%.
The benchmark would then measure nothing at all. Questions must be phrased the way
a taxpayer would actually ask — different words, same meaning. `--draft` instructs
the model accordingly, and `report_overlap` measures how well that worked so the
problem is visible rather than silent.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from src.index.build import tokenise

MODEL = "claude-opus-5"

PROCESSED = Path("data/processed")
OUT = Path("eval/gold/candidates.jsonl")

MIN_CHARS = 400          # too short to hold a real obligation
MAX_CHARS = 3000
PER_CATEGORY = 12        # stratify so one big law cannot dominate

# Articles that only define terms make poor benchmark questions: they are answered
# by a dictionary lookup rather than by locating an obligation.
DEFINITION_ONLY = re.compile(
    r"^(për qëllime të këtij ligji|në kuptim të këtij ligji|përkufizime)", re.I
)

DRAFT_PROMPT = """Je duke ndërtuar një bazë testimi për një sistem pyetje-përgjigje \
mbi legjislacionin tatimor shqiptar.

Më poshtë është një nen nga legjislacioni. Shkruaj NJË pyetje në shqip që një \
tatimpagues i zakonshëm (jo jurist) do ta bënte, dhe që përgjigjet nga ky nen.

Rregulla të detyrueshme:
- Përdor fjalorin e një personi të zakonshëm, JO fjalët e nenit. Mos kopjo terma \
karakteristikë nga teksti.
- Pyetja duhet të jetë konkrete dhe praktike, jo abstrakte.
- Mos përmend numrin e nenit apo të ligjit.
- Kthe vetëm pyetjen, asgjë tjetër.

NENI:
{text}
"""


def load_article_chunks() -> list[dict]:
    path = PROCESSED / "chunks.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["strategy"] == "article":
            rows.append(row)
    return rows


def load_documents() -> dict[int, dict]:
    path = PROCESSED / "documents.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["doc_id"]] = row
    return out


def is_substantive(chunk: dict) -> bool:
    if not MIN_CHARS <= chunk["chars"] <= MAX_CHARS:
        return False
    if "(" in chunk.get("label", ""):   # a fragment of a split article
        return False
    if DEFINITION_ONLY.search(chunk.get("heading", "")):
        return False
    return bool(chunk.get("heading"))


def select(seed: int = 20260827) -> list[dict]:
    documents = load_documents()
    by_category: dict[str, list[dict]] = defaultdict(list)

    for chunk in load_article_chunks():
        if not is_substantive(chunk):
            continue
        meta = documents.get(chunk["doc_id"], {})
        by_category[meta.get("category", "unknown")].append((chunk, meta))

    rng = random.Random(seed)
    candidates = []
    for category, items in sorted(by_category.items()):
        rng.shuffle(items)
        for chunk, meta in items[:PER_CATEGORY]:
            candidates.append({
                "id": f"{category}-{chunk['doc_id']}-{chunk['label'].replace(' ', '')}",
                "question": "",                     # author or --draft fills this
                "gold_doc_id": chunk["doc_id"],
                "gold_article": chunk.get("cite", chunk.get("label", "")),
                "category": category,
                "answerable": True,
                "reviewed": False,
                "source_title": meta.get("title", ""),
                "source_url": meta.get("url", ""),
                "source_heading": chunk.get("heading", ""),
                "source_text": chunk["text"][:1200],
            })
    return candidates


def draft_questions(candidates: list[dict]) -> None:
    """Ask Claude to draft each question. Author review is still required."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    for i, row in enumerate(candidates, start=1):
        if row["question"]:
            continue
        message = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": DRAFT_PROMPT.format(text=row["source_text"]),
            }],
        )
        row["question"] = message.content[0].text.strip()
        row["drafted_by_model"] = True
        print(f"  [{i}/{len(candidates)}] {row['id']}: {row['question'][:70]}")


def report_overlap(candidates: list[dict]) -> None:
    """How much question vocabulary is copied from the source article.

    High overlap means the benchmark is measuring lexical copying rather than
    retrieval. Reported so the problem cannot hide.
    """
    scored = []
    for row in candidates:
        if not row["question"]:
            continue
        q = set(tokenise(row["question"]))
        src = set(tokenise(row["source_text"]))
        if q:
            scored.append(len(q & src) / len(q))
    if not scored:
        print("\noverlap: no drafted questions yet")
        return
    mean = sum(scored) / len(scored)
    high = sum(1 for s in scored if s > 0.6)
    print(f"\nquestion/article vocabulary overlap: mean {mean:.0%} "
          f"| {high}/{len(scored)} above 60%")
    if mean > 0.6:
        print("  WARNING: questions are too close to the source text — BM25 will win")
        print("  trivially and the benchmark will not distinguish configurations.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true",
                        help="draft questions with Claude (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    candidates = select()
    print(f"selected {len(candidates)} candidate articles across "
          f"{len({c['category'] for c in candidates})} categories")

    if args.draft:
        draft_questions(candidates)

    report_overlap(candidates)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwritten: {OUT}")
    print("Next: review each row, fix the question, set \"reviewed\": true,")
    print("then copy the reviewed rows into eval/gold/questions.jsonl")


if __name__ == "__main__":
    main()
