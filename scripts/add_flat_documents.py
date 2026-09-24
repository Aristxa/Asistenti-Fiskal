"""Give the article index coverage of documents that have no article structure.

49 of 313 documents are `flat`: scanned or free-form acts where no `Neni` and no
decimal numbering can be detected. Structure-aware chunking has nothing to cut on,
so those documents produced no article chunks at all -- the article index covered
264 documents while the fixed-window index covered all 313.

That gap was invisible and it biased the evaluation. Candidates are sampled from
the article index, so no benchmark question could ever come from a flat document,
and the article strategy was never charged for being blind to 16% of the corpus.
A comparison where one arm silently does not index part of the corpus is not a
comparison.

The fix is a documented fallback rather than a fake structure: where no structure
exists, fall back to windows, and cite the document section honestly ("Pjesa 3/12")
instead of inventing an article number that is not in the text.

No re-embedding is needed. Those windows are already embedded in the fixed index
with the same model and the same normalisation, so the vectors are transplanted
directly -- exact, and free.

    python scripts/add_flat_documents.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# Run as a plain script (`python scripts/...`), so the project root is not on
# sys.path by default and `src` would not import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INDEX = Path("data/processed/index")
DOCUMENTS = Path("data/processed/documents.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def flat_doc_titles() -> dict[int, str]:
    return {
        d["doc_id"]: (d.get("title") or "").strip()
        for d in read_jsonl(DOCUMENTS)
        if d.get("regime") == "flat"
    }


def main(dry_run: bool = False) -> None:
    import faiss
    import numpy as np

    titles = flat_doc_titles()
    print(f"dokumente pa strukture (flat): {len(titles)}")

    article_chunks = read_jsonl(INDEX / "article/chunks.jsonl")
    covered = {c["doc_id"] for c in article_chunks}
    todo = {d: t for d, t in titles.items() if d not in covered}
    print(f"  tashme ne indeksin e neneve : {len(titles) - len(todo)}")
    print(f"  per t'u shtuar              : {len(todo)}")
    if not todo:
        print("\nAsgje per te bere.")
        return

    # Positions matter: chunks.jsonl row i corresponds to FAISS vector i.
    fixed_chunks = read_jsonl(INDEX / "fixed/chunks.jsonl")
    take = [(i, c) for i, c in enumerate(fixed_chunks) if c["doc_id"] in todo]
    print(f"  copeza per transplantim     : {len(take)}")

    per_doc: dict[int, int] = {}
    for _, chunk in take:
        per_doc[chunk["doc_id"]] = per_doc.get(chunk["doc_id"], 0) + 1

    new_chunks: list[dict] = []
    seen: dict[int, int] = {}
    for _, chunk in take:
        doc_id = chunk["doc_id"]
        seen[doc_id] = seen.get(doc_id, 0) + 1
        position, total = seen[doc_id], per_doc[doc_id]
        title = titles[doc_id]
        new_chunks.append({
            "chunk_id": f"{doc_id}:flat:{position}",
            "doc_id": doc_id,
            "strategy": "article",
            # No article number is invented. The citation names the part of the
            # document and the heading names the document, so the reader still
            # gets a reference they can check against the official file.
            "label": f"Pjesa {position}/{total}",
            "cite": f"Pjesa {position}/{total}",
            "heading": title[:120],
            "text": chunk["text"],
            "chars": chunk["chars"],
            "order": position - 1,
            "fallback": True,
        })

    if dry_run:
        print("\n--dry-run: asgje nuk u shkrua. Shembuj:")
        for c in new_chunks[:3]:
            print(f"  {c['cite']:<14} {c['heading'][:60]}")
        return

    dense_fixed = faiss.read_index(str(INDEX / "fixed/dense.faiss"))
    vectors = np.vstack([dense_fixed.reconstruct(i) for i, _ in take]).astype("float32")
    norms = np.linalg.norm(vectors, axis=1)
    assert norms.min() > 0.99 and norms.max() < 1.01, "transplanted vectors are not unit-norm"

    dense_article = faiss.read_index(str(INDEX / "article/dense.faiss"))
    before = dense_article.ntotal
    dense_article.add(vectors)
    faiss.write_index(dense_article, str(INDEX / "article/dense.faiss"))
    print(f"\n  dense.faiss : {before:,} -> {dense_article.ntotal:,}")

    with (INDEX / "article/chunks.jsonl").open("a", encoding="utf-8") as handle:
        for chunk in new_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    total_chunks = len(article_chunks) + len(new_chunks)
    print(f"  chunks.jsonl: {len(article_chunks):,} -> {total_chunks:,}")
    assert dense_article.ntotal == total_chunks, "vector count and chunk count disagree"

    from rank_bm25 import BM25Okapi

    from src.index.build import tokenise

    texts = [c["text"] for c in article_chunks + new_chunks]
    (INDEX / "article/bm25.pkl").write_bytes(pickle.dumps(BM25Okapi([tokenise(t) for t in texts])))
    print(f"  bm25.pkl    : rindertuar mbi {len(texts):,} copeza")

    docs_now = len({c["doc_id"] for c in article_chunks + new_chunks})
    print(f"\n  mbulimi i dokumenteve: {len(covered)} -> {docs_now} nga 313")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    main(**vars(parser.parse_args()))
