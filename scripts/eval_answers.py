"""Measure the generation layer: refusal behaviour and citation validity.

Retrieval accuracy says whether the right article was found. It says nothing about
what the system then does with it, and the failures that matter to a reader live
here: answering when it should refuse, refusing when it should answer, and citing
a source that does not support the claim.

Two of the three are measurable without human judgement:

* **Refusal accuracy** — the out-of-scope set has a known answer ("refuse"), so
  this is fully automatic.
* **Citation validity** — every [Sn] marker in the prose must correspond to a
  source that was actually retrieved. A marker pointing at nothing is a fabricated
  citation, and that is checkable mechanically.

The third — whether the cited article genuinely supports the sentence — is a legal
judgement and stays with the author. This script prepares that sample rather than
pretending to score it.

    python scripts/eval_answers.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Local runs keep the key in .env; the Space injects it from its secret store, so
# loading here rather than in library code keeps the deployed path unchanged.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.rag.answer import ask  # noqa: E402
from src.eval.review import OUT_OF_SCOPE, QUESTIONS  # noqa: E402

OUT = Path("eval/results_answers.json")

# Phrases the system uses when it declines. Kept explicit rather than clever: a
# fuzzy match here would silently reclassify real answers as refusals and inflate
# the number this script exists to measure.
REFUSAL_MARKERS = (
    "nuk e gjej përgjigjen",
    "nuk mund ta trajtoj",
    "jashtë fushës",
    "jashtë temës",
    "nuk lidhet me legjislacionin",
    "nuk përgjigjem",
)

CITE = re.compile(r"\[S(\d+)\]")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def refused(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def main() -> None:
    answerable = [r for r in read_jsonl(QUESTIONS) if r.get("answerable")]
    out_of_scope = read_jsonl(OUT_OF_SCOPE)
    print(f"{len(answerable)} te pergjigjshme + {len(out_of_scope)} jashte teme\n")

    # Resume rather than restart. Every row here is a paid API call, and the
    # benchmark grows as review continues — re-running the whole set to add six
    # questions would pay for sixty. Existing rows are kept and only missing ids
    # are requested.
    rows = []
    if OUT.exists():
        previous = json.loads(OUT.read_text(encoding="utf-8"))
        rows = previous.get("rows", [])
        done = {r.get("id") for r in rows}
        before = (len(answerable), len(out_of_scope))
        answerable = [r for r in answerable if r.get("id") not in done]
        out_of_scope = [r for r in out_of_scope if r.get("id") not in done]
        print(f"u gjeten {len(rows)} rreshta te meparshem; "
              f"mbeten {len(answerable)}/{before[0]} + {len(out_of_scope)}/{before[1]}\n")
    for label, batch, should_refuse in (
        ("jashte_teme", out_of_scope, True),
        ("e_pergjigjshme", answerable, False),
    ):
        for n, row in enumerate(batch, start=1):
            try:
                result = ask(row["question"], strategy="article", mode="dense", k=5)
            except Exception as exc:
                print(f"  GABIM {row.get('id')}: {type(exc).__name__}", flush=True)
                continue

            markers = {int(m) for m in CITE.findall(result.text)}
            # Markers carry the *rank* of the hit they cite, and `ask` returns only
            # the hits actually cited -- so the source list is a subset of ranks
            # 1..k, not a contiguous 1..n. Comparing a marker against the *count*
            # of sources therefore flags [S4]-out-of-two as fabricated when it is
            # a correct citation of the fourth retrieved article. The real test is
            # whether a marker resolved to a hit at all.
            available = {h.rank for h in result.sources}
            dangling = markers - available

            rows.append({
                "group": label,
                "id": row.get("id"),
                "question": row["question"],
                "should_refuse": should_refuse,
                "refused": refused(result.text),
                "n_markers": len(markers),
                "n_sources": len(result.sources),
                "dangling_markers": sorted(dangling),
                "cited_ranks": sorted(available),
                "gold_article": row.get("gold_article", ""),
                "cited": [h.citation for h in result.sources],
                "answer": result.text,
            })
            print(f"  [{label} {n:>2}/{len(batch)}] refuzoi={rows[-1]['refused']} "
                  f"citime={len(markers)}", flush=True)
            time.sleep(1)  # stay well inside the public rate limit

    oos = [r for r in rows if r["group"] == "jashte_teme"]
    ans = [r for r in rows if r["group"] == "e_pergjigjshme"]
    correct_refusals = sum(1 for r in oos if r["refused"])
    false_refusals = sum(1 for r in ans if r["refused"])
    dangling = sum(1 for r in rows if r["dangling_markers"])

    summary = {
        "n_out_of_scope": len(oos),
        "n_answerable": len(ans),
        "refusal_accuracy": round(100 * correct_refusals / max(len(oos), 1), 1),
        "false_refusal_rate": round(100 * false_refusals / max(len(ans), 1), 1),
        "answers_with_dangling_citation": dangling,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nsaktesia e refuzimit  : {correct_refusals}/{len(oos)} "
          f"= {summary['refusal_accuracy']}%")
    print(f"refuzime te gabuara   : {false_refusals}/{len(ans)} "
          f"= {summary['false_refusal_rate']}%")
    print(f"citime qe s'ekzistojne: {dangling}/{len(rows)}")
    print(f"\nshkruar: {OUT}")


if __name__ == "__main__":
    main()
