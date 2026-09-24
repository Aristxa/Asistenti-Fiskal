"""Why does retrieval miss? Categorise every failure instead of guessing.

A single accuracy number says the system is wrong 50% of the time but not what to
build next. The fork that matters is depth: if the controlling article is in the
candidate pool but ranked below k, the ranking is at fault and a reranker fixes
it. If it never enters the pool at all, ranking is irrelevant and the problem is
the encoder, the chunking, or the vocabulary gap between how a taxpayer asks and
how a statute is written.

Those two failures look identical in the headline metric and have opposite fixes.

    python -m src.eval.failure_analysis
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.eval.retrieval_eval import load_gold, normalise_cite
from src.index.albanian import fold
from src.index.build import tokenise
from src.rag.retrieve import encode_query, get_index

POOL = 50           # how deep to look before declaring the article unreachable
K = 5               # the pre-registered cutoff
OUT = Path("eval/failure_analysis.json")


def gold_positions(index, gold_doc: int, gold_article: str) -> list[int]:
    """Every index position whose chunk is the gold article of the gold document."""
    return [
        i for i, chunk in enumerate(index.chunks)
        if chunk["doc_id"] == gold_doc
        and normalise_cite(chunk.get("cite", "")) == gold_article
    ]


def lexical_overlap(question: str, text: str) -> float:
    """Share of question tokens that appear in the article, after Albanian folding.

    A low value on a miss points at vocabulary: the taxpayer's words and the
    statute's words do not meet, which is the case query expansion addresses.
    """
    q = set(tokenise(question))
    if not q:
        return 0.0
    t = set(tokenise(text))
    return len(q & t) / len(q)


def main() -> None:
    questions = [row for row in load_gold() if row.get("answerable")]
    index = get_index("article")

    rows: list[dict] = []
    verdicts: Counter[str] = Counter()

    for row in questions:
        gold_doc = row["gold_doc_id"]
        gold_article = normalise_cite(row.get("gold_article", ""))
        targets = set(gold_positions(index, gold_doc, gold_article))

        scored = index.search_dense(encode_query(row["question"]), POOL)

        # Deduplicate exactly as retrieve() does, so a rank here means the same
        # thing as a rank there. Without this a long article stored in several
        # parts occupies several positions, ranks come out pessimistic, and the
        # "hit" row disagrees with the headline number for no real reason.
        ranked, seen = [], set()
        for i, _ in scored:
            chunk = index.chunks[i]
            key = (chunk["doc_id"], chunk.get("cite") or chunk.get("label", ""))
            if key in seen:
                continue
            seen.add(key)
            ranked.append(i)

        # Rank of the gold article within the deep pool, 1-based.
        rank = next((n for n, i in enumerate(ranked, start=1) if i in targets), None)
        doc_rank = next(
            (n for n, i in enumerate(ranked, start=1)
             if index.chunks[i]["doc_id"] == gold_doc), None)

        if not targets:
            verdict = "gold_article_not_in_index"
        elif rank is not None and rank <= K:
            verdict = "hit"
        elif rank is not None:
            verdict = "rankable"          # in the pool, below the cutoff
        elif doc_rank is not None:
            verdict = "wrong_article_right_doc"
        else:
            verdict = "unreachable"       # document never surfaces at all

        verdicts[verdict] += 1
        text = index.chunks[next(iter(targets))]["text"] if targets else ""
        rows.append({
            "id": row.get("id"),
            "question": row["question"],
            "gold_doc_id": gold_doc,
            "gold_article": gold_article,
            "verdict": verdict,
            "rank_in_pool": rank,
            "doc_rank_in_pool": doc_rank,
            "overlap": round(lexical_overlap(row["question"], text), 3),
        })

    total = len(rows)
    print(f"n = {total} answerable questions, pool = {POOL}, cutoff k = {K}\n")
    labels = {
        "hit": "gjendet brenda k (sukses)",
        "rankable": "eshte ne pool, por nen k  -> RENDITJA e ka fajin",
        "wrong_article_right_doc": "ligji i sakte, neni i gabuar",
        "unreachable": "ligji nuk del fare ne pool -> KODUESI/FJALORI",
        "gold_article_not_in_index": "neni i arte mungon ne indeks (defekt te dhenash)",
    }
    for verdict, label in labels.items():
        n = verdicts.get(verdict, 0)
        if n:
            print(f"  {label:<52} {n:>3}  ({n / total:5.1%})")

    misses = [r for r in rows if r["verdict"] != "hit"]
    if misses:
        mean_overlap = sum(r["overlap"] for r in misses) / len(misses)
        hits = [r for r in rows if r["verdict"] == "hit"]
        print(f"\n  mbivendosja leksikore  — sukses: "
              f"{sum(r['overlap'] for r in hits) / max(len(hits), 1):.3f}"
              f"   deshtim: {mean_overlap:.3f}")

    ceiling = verdicts.get("hit", 0) + verdicts.get("rankable", 0)
    print(f"\n  TAVANI i nje riorganizuesi te perkryer: {ceiling}/{total} "
          f"= {ceiling / total:.1%}")
    print(f"  (cdo pyetje ku neni eshte tashme ne pool-in e {POOL})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nshkruar: {OUT}")


if __name__ == "__main__":
    main()
