"""Tests for query-time retrieval.

Both tests here lock in the same class of bug: retrieval that returns confident,
citation-bearing nonsense instead of failing. That failure mode cost several
deploys to find, because nothing in the stack raises when the query vector is
meaningless -- the ranking simply becomes storage order.
"""

from __future__ import annotations

import sys
import types

import httpx
import numpy as np
import pytest

from src.rag import retrieve


def test_zerogpu_never_selects_cuda(monkeypatch):
    """On ZeroGPU, torch reports a GPU that this app can never legitimately use.

    Regression: the deployed Space trusted torch.cuda.is_available(), put the
    encoder on a device that only exists inside a @spaces.GPU allocation, and
    returned zero vectors. Every similarity tied, FAISS returned storage order,
    and the app cited Neni 1-7 of an unrelated law as though it were the answer.
    """
    captured = {}

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return True

    def fake_sentence_transformer(name, device=None, revision=None):
        captured["device"] = device
        return object()

    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=fake_sentence_transformer),
    )
    retrieve.get_encoder.cache_clear()
    try:
        retrieve.get_encoder()
    finally:
        retrieve.get_encoder.cache_clear()

    assert captured["device"] == "cpu"


def test_degenerate_query_vector_raises(monkeypatch):
    """A zero vector must fail loudly rather than rank by storage order."""
    monkeypatch.setattr(retrieve, "get_restoration", lambda: {})
    monkeypatch.setattr(
        retrieve,
        "get_encoder",
        lambda: types.SimpleNamespace(
            encode=lambda *a, **k: np.zeros((1, 1024), dtype="float32")
        ),
    )
    with pytest.raises(RuntimeError, match="degenerate"):
        retrieve.encode_query("Kur duhet të regjistrohem si subjekt i TVSH-së?")


def test_expansion_keeps_the_query_in_charge():
    """Rocchio feedback must perturb the query, not replace it.

    beta below alpha is what stops PRF drifting onto a confident wrong topic. A
    regression that let feedback dominate would look like working code and quietly
    change what the system retrieves.
    """
    rng = np.random.default_rng(0)

    class FakeIndex:
        def __init__(self):
            self.vectors = rng.normal(size=(4, 8)).astype("float32")
            self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self.dense = self

        def reconstruct(self, i):
            return self.vectors[i]

        def search_dense(self, vector, k):
            return [(i, 0.5) for i in range(min(k, len(self.vectors)))]

    index = FakeIndex()
    query = index.vectors[0].copy()
    expanded = retrieve.expand_query(index, query, feedback=3, alpha=1.0, beta=0.4)

    assert np.isclose(np.linalg.norm(expanded), 1.0, atol=1e-5)
    # Still recognisably the original question, not the centroid of the feedback.
    assert float(query @ expanded) > 0.8


def test_expansion_survives_an_empty_index():
    class EmptyIndex:
        dense = None

        def search_dense(self, vector, k):
            return []

    query = np.ones(8, dtype="float32") / np.sqrt(8)
    assert np.allclose(retrieve.expand_query(EmptyIndex(), query), query)


def test_exhausted_credit_reports_a_billing_problem(monkeypatch):
    """A 400 for credit balance must name the cause, not surface as a dead button.

    Regression: BadRequestError was uncaught, propagated into Gradio, and Gradio
    swallowed it. The third distinct root cause in this project to produce the
    identical symptom — a Pyet button that appears to do nothing.
    """
    import anthropic

    from src.rag import answer as answer_mod

    def boom(*args, **kwargs):
        raise anthropic.BadRequestError(
            "Error code: 400 - Your credit balance is too low to access the "
            "Anthropic API.",
            response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
            body=None,
        )

    class FakeClient:
        def __init__(self, *a, **k):
            self.messages = types.SimpleNamespace(create=boom)

    # `anthropic` is imported inside ask(), so the module itself is patched.
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    monkeypatch.setattr(
        answer_mod, "retrieve",
        lambda *a, **k: [retrieve.Hit(chunk={"doc_id": 1, "cite": "Neni 1",
                                             "heading": "x", "text": "y",
                                             "chars": 1}, score=1.0, rank=1)])

    with pytest.raises(RuntimeError, match="kredit"):
        answer_mod.ask("a pyetje?")
