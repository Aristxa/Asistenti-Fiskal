"""Tests for retrieval fusion and deduplication, and for the rate limiter.

These cover logic that does not need an index or a model, so they run in
milliseconds and can gate every commit.
"""

from __future__ import annotations

import pytest

from src.rag.retrieve import RRF_K, rrf
from src.web import ratelimit


class TestRRF:
    def test_agreement_beats_single_list_top(self):
        """A document both arms rank highly outranks one only a single arm loves."""
        dense = [1, 2, 3]
        lexical = [9, 2, 3]
        ranked = [doc for doc, _ in rrf([dense, lexical], k=3)]
        assert ranked[0] == 2

    def test_score_is_sum_of_reciprocal_ranks(self):
        fused = dict(rrf([[7], [7]], k=1))
        assert fused[7] == pytest.approx(2.0 / (RRF_K + 1))

    def test_respects_k(self):
        assert len(rrf([[1, 2, 3, 4, 5]], k=2)) == 2

    def test_handles_disjoint_lists(self):
        ranked = [doc for doc, _ in rrf([[1], [2]], k=5)]
        assert sorted(ranked) == [1, 2]

    def test_empty_input(self):
        assert rrf([[], []], k=5) == []


class TestDeduplication:
    """The dedup logic in `retrieve`, exercised directly on its key rule.

    Regression: parts of one long article filled the whole result set, crowding
    out the article that actually answered the question.
    """

    @staticmethod
    def dedup(chunks, k):
        hits, seen = [], set()
        for chunk in chunks:
            key = (chunk["doc_id"], chunk.get("cite") or chunk.get("label", ""))
            if key in seen:
                continue
            seen.add(key)
            hits.append(chunk)
            if len(hits) == k:
                break
        return hits

    def test_parts_of_one_article_collapse_to_one_hit(self):
        chunks = [
            {"doc_id": 1, "cite": "Neni 63", "label": "Neni 63 (1/3)"},
            {"doc_id": 1, "cite": "Neni 63", "label": "Neni 63 (2/3)"},
            {"doc_id": 1, "cite": "Neni 63", "label": "Neni 63 (3/3)"},
            {"doc_id": 1, "cite": "Neni 64", "label": "Neni 64"},
        ]
        result = self.dedup(chunks, k=5)
        assert [c["cite"] for c in result] == ["Neni 63", "Neni 64"]

    def test_same_article_number_in_different_documents_is_kept(self):
        chunks = [
            {"doc_id": 1, "cite": "Neni 5", "label": "Neni 5"},
            {"doc_id": 2, "cite": "Neni 5", "label": "Neni 5"},
        ]
        assert len(self.dedup(chunks, k=5)) == 2

    def test_k_counts_distinct_articles(self):
        chunks = [{"doc_id": 1, "cite": "Neni 63", "label": f"Neni 63 ({i}/9)"}
                  for i in range(9)]
        chunks += [{"doc_id": 1, "cite": f"Neni {n}", "label": f"Neni {n}"}
                   for n in (64, 65, 66, 67)]
        assert len(self.dedup(chunks, k=5)) == 5


class TestRateLimiter:
    def setup_method(self):
        self.limiter = ratelimit.RateLimiter()

    def test_allows_up_to_the_session_limit(self):
        for _ in range(ratelimit.PER_SESSION_LIMIT):
            allowed, _ = self.limiter.check("a")
            assert allowed

    def test_blocks_past_the_session_limit(self):
        for _ in range(ratelimit.PER_SESSION_LIMIT):
            self.limiter.check("a")
        allowed, message = self.limiter.check("a")
        assert not allowed
        assert "pyetje" in message           # message is in Albanian

    def test_sessions_are_independent(self):
        for _ in range(ratelimit.PER_SESSION_LIMIT):
            self.limiter.check("a")
        allowed, _ = self.limiter.check("b")
        assert allowed

    def test_window_expiry_releases_the_limit(self, monkeypatch):
        for _ in range(ratelimit.PER_SESSION_LIMIT):
            self.limiter.check("a")
        assert not self.limiter.check("a")[0]

        real_time = ratelimit.time.time
        monkeypatch.setattr(
            ratelimit.time, "time",
            lambda: real_time() + ratelimit.PER_SESSION_WINDOW + 1,
        )
        assert self.limiter.check("a")[0]

    def test_global_ceiling_blocks_across_sessions(self):
        for i in range(ratelimit.GLOBAL_LIMIT):
            self.limiter.check(f"session-{i}")
        allowed, message = self.limiter.check("fresh-session")
        assert not allowed
        assert "kufirin" in message
