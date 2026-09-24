"""Retrieval: dense, lexical, and the two fused.

Three configurations are exposed because deciding between them is a research
question, not an implementation detail. Albanian's rich inflection hurts the
lexical arm (no stemmer ships in standard retrieval toolchains); the dense arm is
a multilingual model with thin Albanian representation. Either could win.

Fusion is Reciprocal Rank Fusion, which combines rankings rather than scores.
Dense cosine similarities and BM25 scores are on incomparable scales, so
score-level blending would need a weight tuned per corpus — one more free
parameter to justify at a defense. RRF has none.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.index.albanian import load_restoration, restore
from src.index.build import MODEL_NAME, tokenise

INDEX_ROOT = Path("data/processed/index")

# Exact commit of BAAI/bge-m3 used to embed the corpus. See get_encoder().
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

RRF_K = 60  # standard constant; damps the influence of low ranks

# Cross-encoder used by the optional reranking stage. Same family as the encoder,
# and multilingual — the English-only rerankers are useless on Albanian.
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_POOL = 25  # candidates the reranker scores; latency grows linearly with it


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
    """Load the query encoder, on GPU when one is actually present.

    The corpus is embedded offline, so this only ever encodes a single short
    question and CPU is sufficient — that is what makes the deployment cheap.
    The device is still chosen dynamically because the hosting tier may attach a
    GPU, and there is no reason to ignore it when it is there.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    name = index_model_name()
    if name != MODEL_NAME:
        print(f"note: encoding queries with {name} (the model the index was built "
              f"with), not {MODEL_NAME}")
    # CPU unless a *real* GPU is attached.
    #
    # On ZeroGPU hardware the `spaces` package patches torch.cuda.is_available()
    # to return True at import, so an app initialises as though a GPU were
    # present. Actual CUDA work only happens inside a @spaces.GPU allocation, and
    # by design nothing here runs in one -- the corpus is embedded offline and a
    # request only encodes one short question. Trusting is_available() therefore
    # put the encoder on a phantom device and every query embedding came back
    # degenerate.
    #
    # It failed silently, which is what made it expensive. With a zero query
    # vector every inner product ties at 0, FAISS returns index positions 0,1,2...
    # in storage order, and the app confidently cited Neni 1-7 of whichever law
    # happened to be stored first. The lexical arm is pure Python and was
    # unaffected, so `hybrid` still half-worked and hid the fault.
    device = "cpu"
    if torch.cuda.is_available() and not os.environ.get("SPACES_ZERO_GPU"):
        device = "cuda"

    # The revision is pinned, not just the model name. BAAI/bge-m3 has more than
    # one revision in circulation, and which one resolves depends on the
    # transformers version doing the resolving. model.txt records the name;
    # MODEL_REVISION nails the bytes, so a query is always encoded by the weights
    # that embedded the corpus.
    return SentenceTransformer(name, device=device, revision=MODEL_REVISION)


@lru_cache(maxsize=1)
def get_restoration() -> dict:
    return load_restoration(INDEX_ROOT / "restoration.json")


def encode_query(query: str) -> np.ndarray:
    # Repair missing diacritics before encoding. The lexical arm folds both sides
    # and is already diacritic-blind; the dense arm needs correct orthography
    # instead, because `per` and `për` do not embed to the same point.
    query = restore(query, get_restoration())
    vector = get_encoder().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")[0]

    # A normalised embedding has unit length. Anything else means the encoder did
    # not really run -- a phantom device, half-loaded weights -- and the vector is
    # not comparable to the corpus. Left unchecked that does not raise: the
    # similarities simply tie and retrieval returns whatever sits first in the
    # index, with citations attached, which reads as a confident wrong answer.
    # Failing loudly here is worth more than a plausible-looking result.
    norm = float(np.linalg.norm(vector))
    if not 0.99 <= norm <= 1.01:
        raise RuntimeError(
            f"query embedding is degenerate (norm={norm:.4f}, expected 1.0) — "
            f"the encoder is not producing usable vectors, so retrieval would be "
            f"meaningless"
        )
    return vector


def expand_query(index: "Index", vector: np.ndarray, feedback: int = 3,
                 alpha: float = 1.0, beta: float = 0.4) -> np.ndarray:
    """Rocchio pseudo-relevance feedback, in embedding space.

    The measured failure is vocabulary: on questions the system misses, only 21% of
    the asker's words appear in the controlling article, against 31% on questions
    it gets right. Someone asks "kur duhet të regjistrohem"; the statute says
    "detyrimi për deklarimin e fillimit të veprimtarisë". The two never meet.

    Classical PRF assumes the top results are relevant, harvests their terms and
    re-queries. Here the same idea runs on vectors instead of terms: pull the top
    `feedback` chunks toward the query and search again. It costs one extra search
    and no model — the vectors are already in the index — which is why this is
    tried before anything that needs a second network.

    beta is deliberately well below alpha: the query must stay in charge. Feedback
    that dominates is how PRF drifts onto a confident wrong topic.
    """
    hits = index.search_dense(vector, feedback)
    if not hits:
        return vector
    centroid = np.mean([index.dense.reconstruct(i) for i, _ in hits], axis=0)
    combined = alpha * vector + beta * centroid
    norm = float(np.linalg.norm(combined))
    return (combined / norm).astype("float32") if norm else vector


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder, on the same device policy as the encoder.

    A bi-encoder embeds the question and the article separately and can only
    compare the two summaries. A cross-encoder reads them together, so it can tell
    that an article about registration deadlines answers a question about when to
    register — a judgement the bi-encoder has no way to make. That is why it is
    worth a second model, and it is also why it is slower: it runs once per
    candidate rather than once per query.
    """
    import torch
    from sentence_transformers import CrossEncoder

    device = "cpu"
    if torch.cuda.is_available() and not os.environ.get("SPACES_ZERO_GPU"):
        device = "cuda"
    return CrossEncoder(RERANKER_NAME, device=device, max_length=512)


def rerank_candidates(query: str, candidates: list[tuple[dict, float]]
                      ) -> list[tuple[dict, float]]:
    """Reorder candidates by cross-encoder relevance, highest first."""
    query = restore(query, get_restoration())
    pairs = [(query, chunk["text"]) for chunk, _ in candidates]
    scores = get_reranker().predict(pairs, show_progress_bar=False)
    order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
    return [(candidates[i][0], float(scores[i])) for i in order]


def rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over several ranked id lists."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:k]


def retrieve(query: str, strategy: str = "article", mode: str = "dense", k: int = 5,
             pool: int = 50, rerank: bool = False,
             expand: bool = False) -> list[Hit]:
    """Return the top `k` chunks for `query`.

    `mode` is one of "dense", "bm25", "hybrid" — the three configurations compared
    in the evaluation. `pool` is how deep each arm searches before fusion.

    `rerank` adds a cross-encoder pass over the surviving candidates. It is off by
    default and reported as its own condition, because it is a second model and a
    second cost, and the thesis claim has to stand or fall on the retriever itself.
    """
    index = get_index(strategy)

    # Search deeper than k, then deduplicate: a long article is stored as several
    # parts under one citation, and without this a single article can occupy every
    # slot in the result set — crowding out the article that actually answers the
    # question and inflating nothing but redundancy.
    def dense_vector() -> np.ndarray:
        vector = encode_query(query)
        return expand_query(index, vector) if expand else vector

    if mode == "dense":
        scored = index.search_dense(dense_vector(), pool)
    elif mode == "bm25":
        scored = index.search_bm25(query, pool)
    elif mode == "hybrid":
        dense_ids = [i for i, _ in index.search_dense(dense_vector(), pool)]
        bm25_ids = [i for i, _ in index.search_bm25(query, pool)]
        scored = rrf([dense_ids, bm25_ids], pool)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Keep more candidates than needed when a reranker will reorder them. Failure
    # analysis put the controlling article inside the top 50 but below the cutoff
    # for 28% of questions, so the depth the reranker sees is exactly what decides
    # how many of those it can recover.
    depth = RERANK_POOL if rerank else k

    candidates: list[tuple[dict, float]] = []
    seen: set[tuple[int, str]] = set()
    for i, score in scored:
        chunk = index.chunks[i]
        key = (chunk["doc_id"], chunk.get("cite") or chunk.get("label", ""))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((chunk, score))
        if len(candidates) == depth:
            break

    if rerank and candidates:
        candidates = rerank_candidates(query, candidates)

    return [
        Hit(chunk=chunk, score=score, rank=n)
        for n, (chunk, score) in enumerate(candidates[:k], start=1)
    ]
