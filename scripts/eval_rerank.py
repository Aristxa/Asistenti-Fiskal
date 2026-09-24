"""Measure whether cross-encoder reranking is worth its cost.

Failure analysis showed the controlling article sits inside the top 50 but below
the cutoff for 28% of questions, which puts the ceiling for a perfect reranker at
about 76%. This asks what an actual reranker recovers, and at what latency.

Slow by nature: the cross-encoder runs once per candidate, on CPU. Run it in the
background and read the file.

    python scripts/eval_rerank.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

torch.set_num_threads(4)  # leave headroom; this runs alongside other work

from src.eval.retrieval_eval import load_gold, normalise_cite  # noqa: E402
from src.rag.retrieve import retrieve  # noqa: E402

K = 5
OUT = Path("eval/results_rerank.json")


def hit(hits, gold_doc: int, gold_article: str) -> bool:
    return any(
        h.chunk["doc_id"] == gold_doc
        and normalise_cite(h.chunk.get("cite", "")) == gold_article
        for h in hits
    )


def main() -> None:
    questions = [r for r in load_gold() if r.get("answerable")]
    rows, base_hits, rr_hits = [], 0, 0
    started = time.time()

    for n, row in enumerate(questions, start=1):
        gold_doc, gold_article = row["gold_doc_id"], normalise_cite(row["gold_article"])

        t0 = time.time()
        base = retrieve(row["question"], k=K, rerank=False)
        t_base = time.time() - t0

        t0 = time.time()
        rr = retrieve(row["question"], k=K, rerank=True)
        t_rr = time.time() - t0

        b, r = hit(base, gold_doc, gold_article), hit(rr, gold_doc, gold_article)
        base_hits += b
        rr_hits += r
        rows.append({
            "id": row.get("id"), "question": row["question"],
            "gold": f"{gold_doc}/{gold_article}",
            "baseline_hit": b, "rerank_hit": r,
            "seconds_baseline": round(t_base, 2), "seconds_rerank": round(t_rr, 2),
        })
        print(f"[{n:>2}/{len(questions)}] base={base_hits:>2} rerank={rr_hits:>2} "
              f"({t_rr:.0f}s)", flush=True)

    total = len(questions)
    changed_up = sum(1 for r in rows if r["rerank_hit"] and not r["baseline_hit"])
    changed_down = sum(1 for r in rows if r["baseline_hit"] and not r["rerank_hit"])
    mean_latency = sum(r["seconds_rerank"] for r in rows) / total

    summary = {
        "n": total,
        "baseline_hits": base_hits,
        "rerank_hits": rr_hits,
        "baseline_rate": round(100 * base_hits / total, 1),
        "rerank_rate": round(100 * rr_hits / total, 1),
        "recovered": changed_up,
        "broken": changed_down,
        "mean_seconds_per_query": round(mean_latency, 1),
        "elapsed_minutes": round((time.time() - started) / 60, 1),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nbaseline : {base_hits}/{total} = {summary['baseline_rate']}%")
    print(f"rerank   : {rr_hits}/{total} = {summary['rerank_rate']}%")
    print(f"  rikuperoi {changed_up}, prishi {changed_down}")
    print(f"  {mean_latency:.0f}s mesatarisht per pyetje")
    print(f"\nshkruar: {OUT}")


if __name__ == "__main__":
    main()
