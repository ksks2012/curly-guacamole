"""Tests for HierarchicalCodeRetriever repo routing and filter passthrough."""

from __future__ import annotations

from unittest.mock import MagicMock

from rag.retrieval.base import RetrievalResult
from rag.retrieval.hierarchical_code_retriever import HierarchicalCodeRetriever


def _rr(content: str = "x", score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(content=content, score=score, source="code", metadata={})


def _repo_doc(repo_id: str):
    doc = MagicMock()
    doc.metadata = {"repo_id": repo_id}
    return doc


def test_explicit_repo_ids_passthrough_single_eq_filter():
    repo_index = MagicMock()
    code_ret = MagicMock()
    code_ret.search.return_value = [_rr("ok")]

    h = HierarchicalCodeRetriever(repo_index, code_ret)
    out = h.search("q", top_k=3, repo_ids=["myrepo"])

    repo_index.search.assert_not_called()
    code_ret.search.assert_called_once_with(
        "q",
        top_k=3,
        filters={"repo_id": {"$eq": "myrepo"}},
    )
    assert len(out) == 1 and out[0].content == "ok"


def test_explicit_repo_ids_passthrough_multi_in_filter():
    repo_index = MagicMock()
    code_ret = MagicMock()
    code_ret.search.return_value = []

    h = HierarchicalCodeRetriever(repo_index, code_ret)
    h.search("q", top_k=5, repo_ids=["r1", "r2"])

    code_ret.search.assert_called_once_with(
        "q",
        top_k=5,
        filters={"repo_id": {"$in": ["r1", "r2"]}},
    )


def test_repo_ids_passthrough_merges_existing_filters_with_and():
    repo_index = MagicMock()
    code_ret = MagicMock()
    code_ret.search.return_value = []

    h = HierarchicalCodeRetriever(repo_index, code_ret)
    h.search(
        "q",
        top_k=4,
        filters={"language": {"$eq": "Python"}},
        repo_ids=["myrepo"],
    )

    code_ret.search.assert_called_once_with(
        "q",
        top_k=4,
        filters={
            "$and": [
                {"language": {"$eq": "Python"}},
                {"repo_id": {"$eq": "myrepo"}},
            ]
        },
    )


def test_repo_ids_passthrough_extends_existing_and_filter():
    repo_index = MagicMock()
    code_ret = MagicMock()
    code_ret.search.return_value = []

    h = HierarchicalCodeRetriever(repo_index, code_ret)
    h.search(
        "q",
        filters={"$and": [{"language": {"$eq": "Python"}}]},
        repo_ids=["myrepo"],
    )

    code_ret.search.assert_called_once_with(
        "q",
        top_k=5,
        filters={
            "$and": [
                {"language": {"$eq": "Python"}},
                {"repo_id": {"$eq": "myrepo"}},
            ]
        },
    )


def test_without_repo_ids_uses_repo_routing_then_passthrough():
    repo_index = MagicMock()
    repo_index.search.return_value = [_repo_doc("r1"), _repo_doc("r2")]
    code_ret = MagicMock()
    code_ret.search.return_value = [_rr("a")]

    h = HierarchicalCodeRetriever(repo_index, code_ret, repo_top_k=2)
    out = h.search("auth", top_k=6)

    repo_index.search.assert_called_once_with("auth", top_k=2)
    code_ret.search.assert_called_once_with(
        "auth",
        top_k=6,
        filters={"repo_id": {"$in": ["r1", "r2"]}},
    )
    assert len(out) == 1 and out[0].content == "a"


def test_without_repo_ids_fallback_when_routing_empty():
    repo_index = MagicMock()
    repo_index.search.return_value = []
    code_ret = MagicMock()
    code_ret.search.return_value = [_rr("b")]

    h = HierarchicalCodeRetriever(repo_index, code_ret)
    out = h.search("q", top_k=2, filters={"language": {"$eq": "Python"}})

    code_ret.search.assert_called_once_with(
        "q",
        top_k=2,
        filters={"language": {"$eq": "Python"}},
    )
    assert len(out) == 1 and out[0].content == "b"


def test_repo_ids_normalization_dedup_and_strip():
    repo_index = MagicMock()
    code_ret = MagicMock()
    code_ret.search.return_value = []

    h = HierarchicalCodeRetriever(repo_index, code_ret)
    h.search("q", repo_ids=[" r1 ", "", "r1", "r2"])

    code_ret.search.assert_called_once_with(
        "q",
        top_k=5,
        filters={"repo_id": {"$in": ["r1", "r2"]}},
    )
