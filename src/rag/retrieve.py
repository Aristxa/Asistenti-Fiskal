"""Retrieval: dense, lexical, and the two fused.

Three configurations are exposed because deciding between them is a research
question, not an implementation detail. Albanian's rich inflection hurts the
lexical arm (no stemmer exists); the dense arm is a multilingual model with thin
Albanian representation. Either could win.

Fusion is Reciprocal Rank Fusion, which combines rankings rather than scores.
Dense cosine similarities and BM25 scores are on incomparable scales, so
score-level blending would need a weight tuned per corpus — one more free
parameter to justify at a defense. RRF has none.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.index.albanian import load_restoration, restore
from src.index.build import MODEL_NAME, tokenise

INDEX_ROOT = Path("data/processed/index")

RRF_K = 60  # standard constant; damps the influence of low ranks


@dataclass
class Hit:
    chunk: dict
    score: float
    rank: int

    @property
    def citation(self) -> str:
        cite = self.chunk.get("cite") or self.chunk.get("label") or ""
        return f"{cite} — {self.chunk.get('heading', '')}".strip(" —")


class Index:
    """Loaded once per process; holds both arms for one chunking strategy."""

    def __init__(self, strategy: str):
        import faiss

        base = INDEX_ROOT / strategy
        if not base.exists():
            raise FileNotFoundError(
                f"no index at {base} — run `python -m src.index.build` first"
            )
        self.strategy = strategy
        self.dense = faiss.read_index(str(base / "dense.faiss"))
        self.bm25 = pickle.loads((base / "bm25.pkl").read_bytes())
        self.chunks = [
            json.loads(line)
            for line in (base / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def search_dense(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores, ids = self.dense.search(query_vec.reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def search_bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenise(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


@lru_cache(maxsize=2)
def get_index(strategy: str) -> Index:
    return Index(strategy)


def index_model_name() -> str:
    """The model the index was actually built with.

    Queries must be encoded by the same model as the corpus. `model.txt` is written
    at build time and is authoritative — reading it here means switching encoders
    cannot silently produce meaningless similarities. A dimension mismatch would
    raise, but a same-dimension mismatch between two different models would not,
    and that is the failure worth preventing.
    """
    stamp = INDEX_ROOT / "model.txt"
    if stamp.exists():
        return stamp.read_text(encoding="utf-8").strip()
    return MODEL_NAME


@lru_cache(maxsize=1)
def get_encoder():
    from sentence_transformers import SentenceTransformer

    name = index_model_name()
    if name != MODEL_NAME:
        print(f"note: encoding queries with {name} (the model the index was built "
              f"with), not {MODEL_NAME}")
    return SentenceTransformer(name, device="cpu")


@lru_cache(maxsize=1)
def get_restoration() -> dict:
    return load_restoration(INDEX_ROOT / "restoration.json")


def encode_query(query: str) -> np.ndarray:
    # Repair missing diacritics before encoding. The lexical arm folds both sides
    # and is already diacritic-blind; the dense arm needs correct orthography
    # instead, because `per` and `për` do not embed to the same point.
    query = restore(query, get_restoration())
    return get_encoder().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")[0]


def rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over several ranked id lists."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:k]


def retrieve(query: str, strategy: str = "article", mode: str = "hybrid", k: int = 5,
             pool: int = 50) -> list[Hit]:
    """Return the top `k` chunks for `query`.

    `mode` is one of "dense", "bm25", "hybrid" — the three configurations compared
    in the evaluation. `pool` is how deep each arm searches before fusion.
    """
    index = get_index(strategy)

    # Search deeper than k, then deduplicate: a long article is stored as several
    # parts under one citation, and without this a single article can occupy every
    # slot in the result set — crowding out the article that actually answers the
    # question and inflating nothing but redundancy.
    if mode == "dense":
        scored = index.search_dense(encode_query(query), pool)
    elif mode == "bm25":
        scored = index.search_bm25(query, pool)
    elif mode == "hybrid":
        dense_ids = [i for i, _ in index.search_dense(encode_query(query), pool)]
        bm25_ids = [i for i, _ in index.search_bm25(query, pool)]
        scored = rrf([dense_ids, bm25_ids], pool)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    hits: list[Hit] = []
    seen: set[tuple[int, str]] = set()
    for i, score in scored:
        chunk = index.chunks[i]
        key = (chunk["doc_id"], chunk.get("cite") or chunk.get("label", ""))
        if key in seen:
            continue
        seen.add(key)
        hits.append(Hit(chunk=chunk, score=score, rank=len(hits) + 1))
        if len(hits) == k:
            break
    return hits
