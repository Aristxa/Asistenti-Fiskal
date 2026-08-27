"""Build the dense and lexical indexes, offline.

Everything expensive happens here, on a developer machine, and the result is
committed. The hosted Space never embeds a corpus — it loads these files and
encodes one short question per request, which is fast on a free CPU tier.

One index pair is built per chunking strategy ("article", "fixed") so the two
arms of the chunking comparison can be searched independently over identical
source text.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

PROCESSED = Path("data/processed")
INDEX_ROOT = PROCESSED / "index"

# Multilingual encoder with usable Albanian coverage. Albanian is low-resource;
# no monolingual sentence encoder of comparable quality exists.
#
# Override with EMBED_MODEL. Measured on this project's build machine (4-core CPU,
# no GPU), encoding the full 31k-chunk corpus takes ~3 h with MiniLM and ~10 h with
# e5-base, so the full index is built on a GPU (see scripts/build_index_gpu.py) and
# committed. The Space only ever encodes one query at a time, which is fast on CPU.
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH", "16"))
MAX_SEQ_LENGTH = int(os.environ.get("EMBED_SEQ", "512"))

# Albanian keeps ë and ç; everything else non-alphanumeric is a separator.
TOKEN = re.compile(r"[a-z0-9ëç]+")


def tokenise(text: str) -> list[str]:
    """Lowercase word tokens for BM25.

    No stemming: Albanian is morphologically rich (nominal declension, definite
    and indefinite forms) and there is no reliable open Albanian stemmer. Lexical
    matching therefore misses inflected variants, which is precisely the weakness
    the dense arm is expected to cover — and the reason the hybrid comparison is
    worth running rather than assuming.
    """
    return TOKEN.findall(text.lower())


def load_chunks(strategy: str, categories: set[str] | None = None) -> list[dict]:
    path = PROCESSED / "chunks.jsonl"

    keep_docs: set[int] | None = None
    if categories:
        keep_docs = set()
        for line in (PROCESSED / "documents.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                doc = json.loads(line)
                if doc.get("category") in categories:
                    keep_docs.add(doc["doc_id"])

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["strategy"] != strategy:
            continue
        if keep_docs is not None and row["doc_id"] not in keep_docs:
            continue
        rows.append(row)
    return rows


def embed_text(chunk: dict) -> str:
    """Prepend the heading so a chunk carries what it is about, not just its body."""
    heading = chunk.get("heading") or ""
    label = chunk.get("cite") or chunk.get("label") or ""
    prefix = f"{label}. {heading}".strip(" .")
    return f"{prefix}\n{chunk['text']}" if prefix else chunk["text"]


def build_strategy(strategy: str, model, categories: set[str] | None = None) -> None:
    chunks = load_chunks(strategy, categories)
    if not chunks:
        print(f"  {strategy}: no chunks, skipping")
        return

    out = INDEX_ROOT / strategy
    out.mkdir(parents=True, exist_ok=True)

    texts = [embed_text(c) for c in chunks]
    print(f"  {strategy}: encoding {len(texts)} chunks ...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,   # cosine similarity via inner product
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out / "dense.faiss"))

    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenise(t) for t in texts])
    (out / "bm25.pkl").write_bytes(pickle.dumps(bm25))

    # Metadata only — the index rows and this file are positionally aligned.
    with (out / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    np.save(out / "dim.npy", np.array([vectors.shape[1]]))
    print(f"  {strategy}: {len(chunks)} chunks, dim {vectors.shape[1]} -> {out}")


def main(categories: set[str] | None = None, device: str = "cpu",
         threads: int | None = None) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    # torch defaults to half the cores on this machine; using all of them is a
    # free ~25% on CPU encoding.
    torch.set_num_threads(threads or os.cpu_count() or 1)

    print(f"loading {MODEL_NAME} on {device} (first run downloads the model) ...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"  max_seq_length={model.max_seq_length} batch={BATCH_SIZE} "
          f"threads={torch.get_num_threads()}")

    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    for strategy in ("article", "fixed"):
        build_strategy(strategy, model, categories)

    (INDEX_ROOT / "model.txt").write_text(MODEL_NAME, encoding="utf-8")
    print("\nindexes built")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build dense + BM25 indexes")
    parser.add_argument("--categories", nargs="*",
                        help="restrict to these categories (default: whole corpus)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()

    main(set(args.categories) if args.categories else None, args.device, args.threads)
