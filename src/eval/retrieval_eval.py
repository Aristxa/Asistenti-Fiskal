"""Retrieval evaluation — the core result, measured by counting.

Runs the benchmark against six configurations (three retrieval modes x two
chunking strategies) and reports, for each, how often the controlling article
was retrieved. No language model is involved and no API is called, so this can
be re-run freely.

Reporting rules, fixed before any result is seen (docs/PLAN.md §4):

  * every number is printed as `n/N = pp%`, so the sample size is always visible
  * a difference below DECISION_MARGIN points is reported as "no clear difference"

There are no significance tests here. With a benchmark of this size a ten-point
threshold is what can honestly be distinguished, and saying so in plain language
is more defensible than a p-value that invites questions about assumptions no
one checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.rag.retrieve import retrieve

GOLD = Path("eval/gold/questions.jsonl")

MODES = ("dense", "bm25", "hybrid")
STRATEGIES = ("article", "fixed")

DECISION_MARGIN = 10.0  # percentage points; set in advance, never tuned to results
DEFAULT_K = 5


@dataclass
class Result:
    mode: str
    strategy: str
    article_hits: int
    doc_hits: int
    total: int
    chars_retrieved: int

    @property
    def article_rate(self) -> float:
        return 100.0 * self.article_hits / max(self.total, 1)

    @property
    def doc_rate(self) -> float:
        return 100.0 * self.doc_hits / max(self.total, 1)

    @property
    def mean_chars(self) -> int:
        return self.chars_retrieved // max(self.total, 1)


def load_gold() -> list[dict]:
    if not GOLD.exists():
        raise FileNotFoundError(
            f"no benchmark at {GOLD} — build it with `python -m src.eval.make_candidates`"
        )
    rows = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Out-of-scope questions have no gold article; they are scored separately.
    return [r for r in rows if r.get("answerable", True)]


def normalise_cite(value: str) -> str:
    """Compare citations ignoring case, spacing and any part suffix."""
    value = (value or "").split("(")[0]
    return " ".join(value.lower().split())


def evaluate(questions: list[dict], mode: str, strategy: str, k: int) -> Result:
    article_hits = doc_hits = chars = 0

    for row in questions:
        hits = retrieve(row["question"], strategy=strategy, mode=mode, k=k)
        chars += sum(h.chunk["chars"] for h in hits)

        gold_doc = row["gold_doc_id"]
        gold_article = normalise_cite(row.get("gold_article", ""))

        if any(h.chunk["doc_id"] == gold_doc for h in hits):
            doc_hits += 1
        # The fixed-window arm has no article labels, so an article-level hit is
        # credited when the right document's window covering that article is found.
        if strategy == "article":
            found = any(
                h.chunk["doc_id"] == gold_doc
                and normalise_cite(h.chunk.get("cite", "")) == gold_article
                for h in hits
            )
        else:
            found = any(
                h.chunk["doc_id"] == gold_doc
                and gold_article
                and gold_article in normalise_cite(h.chunk.get("text", ""))
                for h in hits
            )
        article_hits += int(found)

    return Result(mode, strategy, article_hits, doc_hits, len(questions), chars)


def compare(label: str, a: Result, b: Result, metric: str = "article_rate") -> str:
    lo, hi = getattr(a, metric), getattr(b, metric)
    gap = hi - lo
    if abs(gap) < DECISION_MARGIN:
        return (f"{label}: no clear difference "
                f"({a.mode}/{a.strategy} {lo:.0f}% vs {b.mode}/{b.strategy} {hi:.0f}%, "
                f"gap {abs(gap):.0f}pp < {DECISION_MARGIN:.0f}pp)")
    winner, loser = (b, a) if gap > 0 else (a, b)
    return (f"{label}: {winner.mode}/{winner.strategy} wins by {abs(gap):.0f}pp "
            f"({max(lo, hi):.0f}% vs {min(lo, hi):.0f}%)")


def main(k: int = DEFAULT_K) -> None:
    questions = load_gold()
    print(f"benchmark: {len(questions)} answerable questions, k={k}\n")

    results: dict[tuple[str, str], Result] = {}
    for strategy in STRATEGIES:
        for mode in MODES:
            res = evaluate(questions, mode, strategy, k)
            results[(mode, strategy)] = res

    header = f"{'mode':<8} {'chunking':<9} {'found article':<20} {'found document':<20} {'chars/query':>11}"
    print(header)
    print("-" * len(header))
    for strategy in STRATEGIES:
        for mode in MODES:
            r = results[(mode, strategy)]
            print(f"{mode:<8} {strategy:<9} "
                  f"{r.article_hits:>3}/{r.total:<3} = {r.article_rate:>3.0f}%      "
                  f"{r.doc_hits:>3}/{r.total:<3} = {r.doc_rate:>3.0f}%      "
                  f"{r.mean_chars:>11}")

    print(f"\nCONFIRMATORY — pre-registered comparisons "
          f"(margin {DECISION_MARGIN:.0f}pp, fixed in advance):")
    best_mode = max(MODES, key=lambda m: results[(m, "article")].article_rate)
    print("  " + compare("chunking (hybrid)",
                         results[("hybrid", "fixed")], results[("hybrid", "article")]))
    print("  " + compare("dense vs hybrid (article)",
                         results[("dense", "article")], results[("hybrid", "article")]))
    print("  " + compare("bm25 vs hybrid (article)",
                         results[("bm25", "article")], results[("hybrid", "article")]))

    # Reported separately, and never promoted into the confirmatory list.
    #
    # The pre-registration fixed the chunking comparison on the hybrid arm, because
    # hybrid was the deployed default when it was written. Dense became the default
    # later, on the strength of these same results — so the dense chunking
    # comparison is chosen with knowledge of the outcome, which is precisely what a
    # pre-registration exists to prevent. It is the largest effect in the study and
    # it is still exploratory. Presenting it as confirmatory would be the exact
    # error the decision rule was adopted to avoid.
    print("\nEXPLORATORY — chosen after seeing results, NOT confirmatory:")
    print("  " + compare("chunking (dense)",
                         results[("dense", "fixed")], results[("dense", "article")]))
    print(f"  best retrieval mode on article chunks: {best_mode}")

    if len(questions) < 60:
        print(f"\n  NOTE: {len(questions)} answerable questions; the pre-declared "
              f"minimum is 60.\n  These numbers are provisional until the benchmark "
              f"is complete.")

    out = Path("eval/results_retrieval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {f"{m}/{s}": vars(results[(m, s)]) for m, s in results}, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    args = parser.parse_args()
    main(args.k)
