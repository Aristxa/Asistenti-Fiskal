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

# Categories whose documents are compilations of scanned correspondence rather
# than legislation. Their sections are a sentence of substance wrapped in
# letterhead, and even after the letterhead is stripped they read as fragments of
# a reply letter. They stay in the corpus -- they are real DPT positions and worth
# retrieving -- but they make poor benchmark items, and asking a reviewer to judge
# them wastes the scarcest resource in the project.
EXCLUDED_CATEGORIES = {"akte-te-dpt", "vendime-teknike"}

# An article that stops mid-sentence cannot be judged: the reviewer cannot tell
# whether the answer was in the part that got cut. 26% of the first candidate set
# ended without terminal punctuation.
SENTENCE_END = tuple(".!?:;”\")")

# Articles that only define terms make poor benchmark questions: they are answered
# by a dictionary lookup rather than by locating an obligation.
DEFINITION_ONLY = re.compile(
    r"^(për qëllime të këtij ligji|në kuptim të këtij ligji|përkufizime)", re.I
)

# Every Albanian legal act closes with the same formulaic provisions: who is
# charged with implementation, when it enters into force, what it repeals. They
# are articles, but nobody asks a question about them, and including them would
# spend the author's review time on rows that cannot become useful benchmark
# entries.
BOILERPLATE = re.compile(
    r"(?i)(^\s*ngarkoh"
    r"|hyn në fuqi"
    r"|fletoren zyrtare"
    r"|shfuqizoh"
    r"|^\s*ky (vendim|ligj|udhëzim) )"
)

# Who is asking. A benchmark whose questions all sound like one person tests one
# register; real users arrive as several. The persona is rotated deterministically
# so the set is varied without being random between runs.
PERSONAS: tuple[tuple[str, str], ...] = (
    ("pronar biznesi i vogël",
     "Ke një dyqan ose një biznes të vogël. Nuk ke studiuar drejtësi dhe nuk njeh "
     "terminologjinë ligjore."),
    ("i vetëpunësuar",
     "Punon për vete, pa punonjës. Të interesojnë detyrimet e tua personale."),
    ("punëdhënës",
     "Ke disa punonjës dhe të interesojnë pagat, kontributet dhe detyrimet ndaj tyre."),
    ("kontabilist fillestar",
     "Sapo ke filluar punë si kontabilist dhe kërkon rregullin e saktë, por e formulon "
     "pyetjen thjesht."),
    ("themelues i ri biznesi",
     "Po hap biznes dhe nuk di ende asgjë nga procedurat."),
)

# Question shapes offered as options, not assigned. Forcing a shape was a mistake:
# rotating a "what happens if you fail to" form onto an article that contains no
# penalty made the model invent the framing, and 25 of 26 such drafts asked about
# a consequence their article never mentions. The same held for deadline and
# amount forms. The model now picks the shape the article can actually support,
# which is a decision only the article's content can make.
QUESTION_FORMS: tuple[str, ...] = (
    "A ... (po/jo)",
    "Kur / Deri kur ... (afat)",
    "Sa ... (shumë ose përqindje)",
    "Çfarë ndodh nëse ... (pasojë e mospërmbushjes)",
    "Si ... (procedurë)",
    "Kush ... (subjekti i detyrimit)",
)

DRAFT_PROMPT = """Po ndërtojmë një bazë testimi për një sistem pyetje-përgjigje mbi legjislacionin tatimor dhe kontabël shqiptar.

TI JE: {persona_name}. {persona_desc}

Më poshtë është një nen i legjislacionit. Shkruaj pyetjen që do t'i bëje ti një kontabilisti ose Drejtorisë së Tatimeve për situatën që trajton ky nen.

ZGJIDH VETË FORMËN e pyetjes, atë që ky nen mund ta përgjigjet vërtet:
{form}

RREGULL KRYESOR — MOS E SHKEL:
Pyetja duhet t'i përgjigjet VETËM nga ky nen. Mos pyet për një afat nëse neni
nuk jep afat. Mos pyet "çfarë ndodh nëse" nëse neni nuk përmend pasojë apo
sanksion. Mos pyet për shumë ose përqindje nëse neni nuk përmban asnjë.
Nëse neni thjesht përshkruan një rregull, pyet për vetë rregullin.

RREGULLA TË TJERA:
- Përdor gjuhën e përditshme. MOS përdor termat karakteristikë të nenit. Nëse neni thotë "subjekt i tatueshëm", ti thuaj "unë" ose "biznesi im"; nëse thotë "furnizim mallrash", ti thuaj "kur shes diçka".
- Pyetja duhet të jetë konkrete dhe praktike — diçka që dikush e pyet vërtet.
- Mos përmend numrin e nenit, të ligjit apo të udhëzimit.
- Mos e kopjo strukturën e fjalisë së nenit.
- Një pyetje e vetme, maksimumi 20 fjalë.
- Kthe VETËM pyetjen, pa shpjegim dhe pa thonjëza.

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


def is_substantive(chunk: dict, category: str = "") -> bool:
    if category in EXCLUDED_CATEGORIES:
        return False
    if not MIN_CHARS <= chunk["chars"] <= MAX_CHARS:
        return False
    body = chunk["text"].rstrip()
    if not body or not body.endswith(SENTENCE_END):
        return False
    if "(" in chunk.get("label", ""):   # a fragment of a split article
        return False
    if DEFINITION_ONLY.search(chunk.get("heading", "")):
        return False
    if BOILERPLATE.search(chunk["text"][:300]):
        return False
    return bool(chunk.get("heading"))


def select(seed: int = 20260827) -> list[dict]:
    documents = load_documents()
    by_category: dict[str, list[dict]] = defaultdict(list)

    for chunk in load_article_chunks():
        meta = documents.get(chunk["doc_id"], {})
        if not is_substantive(chunk, meta.get("category", "")):
            continue
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
                # Full text, not an excerpt: the author cannot judge whether an article
                # answers a question while seeing only part of it.
                "source_text": chunk["text"],
            })
    return candidates


def draft_questions(candidates: list[dict]) -> None:
    """Draft one question per article, rotating persona and question form.

    The label is not a model judgement: the question is written *from* a known
    article, so that article is the answer by construction. What the author
    verifies is that the draft is a sensible question which that article really
    answers -- review rather than authoring.
    """
    import anthropic

    client = anthropic.Anthropic()
    pending = [row for row in candidates if not row["question"]]

    for i, row in enumerate(pending, start=1):
        persona_name, persona_desc = PERSONAS[i % len(PERSONAS)]
        form = "\n".join(f"  - {f}" for f in QUESTION_FORMS)
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": DRAFT_PROMPT.format(
                        persona_name=persona_name,
                        persona_desc=persona_desc,
                        form=form,
                        text=row["source_text"],
                    ),
                }],
            )
        except anthropic.AuthenticationError as exc:
            raise SystemExit(
                "nuk ka kredenciale — vendos ANTHROPIC_API_KEY dhe provo sërish"
            ) from exc
        except anthropic.RateLimitError as exc:
            retry = exc.response.headers.get("retry-after", "60")
            raise SystemExit(f"kufi shpejtësie; provo sërish pas {retry}s") from exc

        if message.stop_reason == "refusal":
            print(f"  [{i}/{len(pending)}] {row['id']}: u refuzua, u kapërcye")
            continue

        text = "".join(b.text for b in message.content if b.type == "text").strip()
        row["question"] = text.strip('"“”')
        row["drafted_by_model"] = True
        row["persona"] = persona_name
        print(f"  [{i}/{len(pending)}] [{persona_name}] {row['question'][:64]}")


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


# A question's premise must exist in its article. Asking "what happens if you
# fail to" against an article containing no penalty produces a question with no
# answer, and in evaluation that counts as a retrieval failure -- the system is
# blamed for correctly failing to answer the unanswerable. Detected here so the
# drafting run reports it rather than leaving it for a human to notice.
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


def unsupported_form(row: dict) -> str | None:
    """Which question form this article cannot answer, if any."""
    question = row.get("question") or ""
    text = row.get("source_text") or ""
    for name, opener, evidence in FORM_REQUIREMENTS:
        if re.search(opener, question) and not re.search(evidence, text):
            return name
    return None


def report_unsupported(candidates: list[dict]) -> None:
    from collections import Counter

    flagged = [(r, unsupported_form(r)) for r in candidates if r.get("question")]
    flagged = [(r, name) for r, name in flagged if name]
    for row, name in flagged:
        row["unsupported_form"] = name
    if not flagged:
        print("\nforma e pyetjeve: të gjitha mbështeten nga neni përkatës")
        return
    print(f"\n{len(flagged)} pyetje kërkojnë diçka që neni nuk e përmban:")
    for name, count in Counter(n for _, n in flagged).most_common():
        print(f"    {name:<10} {count}")
    print("  Këto do të numëroheshin si dështime kërkimi edhe kur sistemi ka të drejtë.")


def flag_paraphrases(candidates: list[dict], threshold: float = 0.6) -> None:
    """Mark drafts that lean too heavily on the article's own wording.

    These are the rows most worth the author's attention: a question built from
    the article's vocabulary is found by keyword matching alone, so it measures
    string overlap rather than retrieval.
    """
    flagged = 0
    for row in candidates:
        if not row.get("question"):
            continue
        question = set(tokenise(row["question"]))
        source = set(tokenise(row["source_text"]))
        if not question:
            continue
        overlap = len(question & source) / len(question)
        row["overlap"] = round(overlap, 2)
        if overlap > threshold:
            row["needs_rewrite"] = True
            flagged += 1
    if flagged:
        print(f"\n{flagged} pyetje përsërisin fjalorin e nenit (>{threshold:.0%}).")
        print("Janë shënuar 'needs_rewrite' — rishikoji këto të parat.")


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
    flag_paraphrases(candidates)
    report_unsupported(candidates)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwritten: {OUT}")
    print("Next: review each row, fix the question, set \"reviewed\": true,")
    print("then copy the reviewed rows into eval/gold/questions.jsonl")


if __name__ == "__main__":
    main()
