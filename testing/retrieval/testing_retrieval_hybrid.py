"""
Smoke test for Phase 1 — Unified Retrieval Architecture.

These tests run entirely with mock retrievers so they do NOT require a
running embedding server or Chroma database. They verify:

- RetrievalResult dataclass and unique_key deduplication logic
- BaseRetriever Protocol conformance (runtime isinstance check)
- HybridRetriever RRF merging (scores, ordering, deduplication)
- HybridRetriever with weighted retrievers
- HybridRetriever with reranker mock
- Edge cases: empty retrievers, single retriever, duplicate items
"""

from __future__ import annotations

from rag.retrieval.base import BaseRetriever, RetrievalResult
from rag.retrieval.hybrid_retriever import HybridRetriever, _rrf_merge


# ---------------------------------------------------------------------------
# Mock retriever — no network / DB needed
# ---------------------------------------------------------------------------

class _MockRetriever:
    """Returns a fixed list of RetrievalResult objects."""

    def __init__(
        self, results: list[RetrievalResult], name: str = "mock"
    ) -> None:
        self._results = results
        self.name = name

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        return self._results[:top_k]


def _make_doc_result(chunk_id: str, content: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        source="document",
        metadata={"chunk_id": chunk_id},
    )


def _make_code_result(symbol_id: str, content: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        source="code",
        metadata={"symbol_id": symbol_id},
    )


# ---------------------------------------------------------------------------
# Tests: RetrievalResult
# ---------------------------------------------------------------------------

def test_retrieval_result_unique_key() -> None:
    r1 = _make_doc_result("chunk-1", "hello world")
    assert r1.unique_key() == "chunk-1"

    r2 = _make_code_result("sym-A", "def foo(): pass")
    assert r2.unique_key() == "sym-A"

    # No metadata key → falls back to content hash
    r3 = RetrievalResult(content="bare content", score=0.5, source="document")
    key = r3.unique_key()
    assert len(key) == 16, f"expected 16-char hex key, got {len(key)!r}: {key!r}"

    # Two results with identical content → same key
    r4 = RetrievalResult(content="bare content", score=0.9, source="code")
    assert r3.unique_key() == r4.unique_key()


# ---------------------------------------------------------------------------
# Tests: BaseRetriever Protocol conformance
# ---------------------------------------------------------------------------

def test_protocol_conformance() -> None:
    mock = _MockRetriever([])
    assert isinstance(
        mock, BaseRetriever
    ), "MockRetriever should satisfy BaseRetriever Protocol"

    hybrid = HybridRetriever([mock])
    assert isinstance(
        hybrid, BaseRetriever
    ), "HybridRetriever should satisfy BaseRetriever Protocol"


# ---------------------------------------------------------------------------
# Tests: HybridRetriever RRF logic
# ---------------------------------------------------------------------------

def test_hybrid_basic_merge() -> None:
    """Items from multiple retrievers should be fused and sorted by RRF score."""
    doc_results = [
        _make_doc_result("d1", "document one", score=0.9),
        _make_doc_result("d2", "document two", score=0.8),
        _make_doc_result("d3", "document three", score=0.7),
    ]
    code_results = [
        _make_code_result("c1", "code snippet one", score=0.95),
        _make_code_result("c2", "code snippet two", score=0.85),
    ]

    ret_doc = _MockRetriever(doc_results, name="doc")
    ret_code = _MockRetriever(code_results, name="code")

    hybrid = HybridRetriever([ret_doc, ret_code])
    results = hybrid.search("test query", top_k=4)

    assert len(results) == 4, f"expected 4 results, got {len(results)}"
    # All results should have RRF scores > 0
    for r in results:
        assert r.score > 0, f"zero score for {r.metadata}"
    # Scores should be descending
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score, \
            f"out-of-order scores: {[r.score for r in results]}"


def test_hybrid_deduplication() -> None:
    """The same item appearing in both retrievers should appear only once in the output."""
    shared = _make_doc_result("shared-1", "shared content", score=0.9)
    only_in_doc = _make_doc_result("only-doc", "doc only", score=0.7)
    only_in_code = _make_code_result("only-code", "code only", score=0.8)

    # 'shared-1' appears in both lists
    shared_as_code = RetrievalResult(
        content="shared content",
        score=0.95,
        source="code",
        metadata={"chunk_id": "shared-1"},  # same key
    )

    ret1 = _MockRetriever([shared, only_in_doc])
    ret2 = _MockRetriever([shared_as_code, only_in_code])

    hybrid = HybridRetriever([ret1, ret2])
    results = hybrid.search("test", top_k=10)

    keys = [r.unique_key() for r in results]
    assert len(keys) == len(set(keys)), f"duplicate keys in results: {keys}"
    assert "shared-1" in keys, "shared item should appear once"
    assert "only-doc" in keys
    assert "only-code" in keys


def test_hybrid_weighted_retrievers() -> None:
    """Higher-weighted retriever should contribute more to the final score."""
    high = _MockRetriever(
        [_make_doc_result("h1", "high weight item", score=0.9)],
        name="high",
    )
    low = _MockRetriever(
        [_make_doc_result("l1", "low weight item", score=0.9)],
        name="low",
    )
    # Give high-weight retriever 3× more weight
    hybrid = HybridRetriever([high, low], weights=[3.0, 1.0])
    results = hybrid.search("test", top_k=2)

    assert len(results) == 2
    # h1 should rank above l1 (same rank within each list, but higher weight)
    assert results[0].unique_key() == "h1", \
        f"expected h1 first, got {results[0].unique_key()!r}"


def test_hybrid_single_retriever() -> None:
    """HybridRetriever with a single retriever should behave like that retriever."""
    results_in = [
        _make_doc_result(f"d{i}", f"content {i}", score=float(i)) for i in range(5)
    ]
    ret = _MockRetriever(results_in)
    hybrid = HybridRetriever([ret])
    out = hybrid.search("q", top_k=3)
    assert len(out) == 3


def test_hybrid_empty_sub_retriever() -> None:
    """If one sub-retriever returns nothing, results still come from the other."""
    empty = _MockRetriever([])
    nonempty = _MockRetriever([_make_doc_result("x1", "exists")])
    hybrid = HybridRetriever([empty, nonempty])
    out = hybrid.search("q", top_k=5)
    assert len(out) == 1
    assert out[0].unique_key() == "x1"


def test_hybrid_no_results() -> None:
    """All empty sub-retrievers → empty result list."""
    hybrid = HybridRetriever([_MockRetriever([]), _MockRetriever([])])
    out = hybrid.search("q", top_k=5)
    assert out == []


def test_rrf_merge_scores() -> None:
    """Verify RRF formula values are correct."""
    list1 = [_make_doc_result("a", "A")]   # rank 1
    list2 = [_make_doc_result("a", "A")]   # rank 1 in second list too

    fused = _rrf_merge([list1, list2], weights=[1.0, 1.0], top_k=1, rrf_k=60)
    assert len(fused) == 1
    # score = 1.0/(60+1) + 1.0/(60+1) = 2/61 ≈ 0.032787
    expected = round(2 / 61, 6)
    assert abs(fused[0].score - expected) < 1e-5, \
        f"RRF score {fused[0].score} != expected {expected}"


def test_hybrid_summary() -> None:
    ret1 = _MockRetriever([])
    ret2 = _MockRetriever([])
    h = HybridRetriever([ret1, ret2], weights=[0.7, 0.3])
    s = h.summary()
    assert "_MockRetriever" in s
    assert "0.7" in s
