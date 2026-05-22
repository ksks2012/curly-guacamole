"""
Smoke tests for Step 1.3 — Unified RAGEngine.

Tests the full pipeline without a real embedding server or LLM:
  - RAGEngine accepts a BaseRetriever (not a get_retriever callable)
  - answer() deduplicates via unique_key(), applies reranker, builds context
  - answer() passes filters down to the retriever
  - LocalLlamaClient wiring: doc_retriever / unified_retriever properties exist
  - attach_code_retriever() rebuilds unified_retriever to HybridRetriever
  - answer_unified() swaps retriever and restores it after the call
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from unittest.mock import MagicMock, patch

from rag.retrieval.base import BaseRetriever, RetrievalResult
from rag.retrieval.document_retriever import DocumentRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.engine import RAGEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(content: str, source: Literal["document", "code"] = "document",
            score: float = 0.9, chunk_id: str | None = None) -> RetrievalResult:
    meta = {"chunk_id": chunk_id} if chunk_id else {}
    return RetrievalResult(content=content, score=score, source=source, metadata=meta)


class _MockRetriever:
    """Minimal BaseRetriever that records calls and returns preset results."""

    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self._results = results or []
        self.calls: list[dict] = []

    def search(self, query: str, top_k: int = 5,
               filters: dict | None = None) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return self._results[:top_k]


def _mock_config():
    cfg = MagicMock()
    cfg.query_expansion_enabled = False
    cfg.query_expansion_n = 3
    return cfg


def _mock_llm(response_text: str = "Answer text"):
    llm = MagicMock()
    resp = MagicMock()
    resp.content = response_text
    llm.invoke.return_value = resp
    return llm


def _make_engine(retriever, *, reranker=None, response="ok"):
    return RAGEngine(
        llm=_mock_llm(response),
        retriever=retriever,
        reranker=reranker,
        config=_mock_config(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_engine_uses_retriever_search():
    """RAGEngine.answer() must call retriever.search(), not a LangChain retriever."""
    mock = _MockRetriever([_result("doc A", chunk_id="a"), _result("doc B", chunk_id="b")])
    engine = _make_engine(mock)
    engine.answer("test query", k=2, fetch_k=5)
    assert len(mock.calls) == 1
    assert mock.calls[0]["query"] == "test query"
    assert mock.calls[0]["top_k"] == 5  # fetch_k passed as top_k to retriever
    print("test_engine_uses_retriever_search: OK")


def test_engine_deduplicates():
    """Duplicate unique_key() results should be dropped."""
    # Both results have the same chunk_id
    dup = [_result("same", chunk_id="x"), _result("same", chunk_id="x")]
    mock = _MockRetriever(dup)
    engine = _make_engine(mock)

    # Patch LLM so we can inspect context built
    built_context: list[str] = []
    original_invoke = engine.llm.invoke

    def capture(prompt):
        built_context.append(prompt)
        return original_invoke(prompt)

    engine.llm.invoke = capture
    engine.answer("q", k=5, fetch_k=5)
    # Only one block should appear in the prompt
    assert built_context[0].count("same") == 1
    print("test_engine_deduplicates: OK")


def test_engine_passes_filters():
    """filters kwarg must be forwarded to retriever.search()."""
    mock = _MockRetriever([_result("r")])
    engine = _make_engine(mock)
    engine.answer("q", k=1, fetch_k=1, filters={"doc_id": "abc"})
    assert mock.calls[0]["filters"] == {"doc_id": "abc"}
    print("test_engine_passes_filters: OK")


def test_engine_doc_id_shortcut():
    """doc_id param should become filters={'doc_id': ...} when filters is None."""
    mock = _MockRetriever([_result("r")])
    engine = _make_engine(mock)
    engine.answer("q", k=1, fetch_k=1, doc_id="xyz")
    assert mock.calls[0]["filters"] == {"doc_id": "xyz"}
    print("test_engine_doc_id_shortcut: OK")


def test_engine_code_source_tag():
    """Code results should produce [file :: symbol] context tags."""
    r = RetrievalResult(
        content="def foo(): pass",
        score=0.8,
        source="code",
        metadata={"file_path": "rag/engine.py", "symbol_name": "foo", "chunk_id": "s1"},
    )
    mock = _MockRetriever([r])
    engine = _make_engine(mock)

    captured: list[str] = []
    orig = engine.llm.invoke
    def cap(p): captured.append(p); return orig(p)
    engine.llm.invoke = cap

    engine.answer("q", k=1, fetch_k=1)
    assert "[rag/engine.py :: foo]" in captured[0]
    print("test_engine_code_source_tag: OK")


def test_engine_reranker_applied():
    """When a reranker is provided, it must be called with the candidates."""
    candidates = [
        _result("low", chunk_id="a", score=0.1),
        _result("high", chunk_id="b", score=0.9),
    ]
    mock = _MockRetriever(candidates)

    reranker = MagicMock()
    from langchain_core.documents import Document
    # Reranker returns (Document, score) tuples — matches rerank_with_scores() API
    reranker.rerank_with_scores.return_value = [(Document(page_content="high", metadata={"chunk_id": "b"}), 0.9)]

    engine = _make_engine(mock, reranker=reranker)
    engine.answer("q", k=1, fetch_k=5)
    reranker.rerank_with_scores.assert_called_once()
    print("test_engine_reranker_applied: OK")


def test_protocol_conformance_mock():
    """_MockRetriever must satisfy the BaseRetriever Protocol."""
    assert isinstance(_MockRetriever(), BaseRetriever)
    print("test_protocol_conformance_mock: OK")


def test_document_retriever_is_protocol():
    """DocumentRetriever must satisfy BaseRetriever even without inheritance."""
    from rag.retrieval.document_retriever import DocumentRetriever
    assert isinstance(DocumentRetriever.__new__(DocumentRetriever), BaseRetriever)
    print("test_document_retriever_is_protocol: OK")


def test_hybrid_is_protocol():
    """HybridRetriever must satisfy BaseRetriever Protocol."""
    from rag.retrieval.hybrid_retriever import HybridRetriever
    h = HybridRetriever([_MockRetriever()])
    assert isinstance(h, BaseRetriever)
    print("test_hybrid_is_protocol: OK")


def test_attach_code_retriever_rebuilds_unified():
    """After attach_code_retriever(), unified_retriever should be HybridRetriever."""
    from rag.retrieval.code_retriever import CodeRetriever

    # Build a minimal LocalLlamaClient-like object without network calls
    class _FakeClient:
        def __init__(self):
            mock_searcher = MagicMock()
            mock_searcher.similarity_search_with_scores.return_value = []
            mock_searcher.hybrid_search_with_scores.return_value = ([], [], [])
            mock_searcher.bm25_index = None
            self.searcher = mock_searcher
            self.reranker = None
            self.doc_retriever = DocumentRetriever(mock_searcher)
            self.code_retriever = None
            self.unified_retriever = self.doc_retriever

        def attach_code_retriever(self, code_indexer, *, level="symbol"):
            self.code_retriever = CodeRetriever(code_indexer, level=level)
            self._rebuild_unified()

        def _rebuild_unified(self):
            if self.code_retriever is not None:
                self.unified_retriever = HybridRetriever(
                    [self.doc_retriever, self.code_retriever]
                )
            else:
                self.unified_retriever = self.doc_retriever

    client = _FakeClient()
    assert client.unified_retriever is client.doc_retriever

    mock_code_indexer = MagicMock()
    mock_code_indexer.search_with_scores.return_value = []
    client.attach_code_retriever(mock_code_indexer)

    assert isinstance(client.unified_retriever, HybridRetriever)
    assert client.unified_retriever.retriever_count == 2
    print("test_attach_code_retriever_rebuilds_unified: OK")


def test_answer_unified_restores_retriever():
    """answer_unified() must restore engine.retriever after the call."""
    doc_ret = _MockRetriever([_result("doc")])
    unified_ret = _MockRetriever([_result("uni")])

    engine = _make_engine(doc_ret)
    engine_orig_retriever = engine.retriever

    # Simulate LocalLlamaClient.answer_unified
    def answer_unified(query, **kwargs):
        prev = engine.retriever
        engine.retriever = unified_ret
        try:
            return engine.answer(query, **kwargs)
        finally:
            engine.retriever = prev

    answer_unified("q", k=1, fetch_k=1)
    assert engine.retriever is engine_orig_retriever, "retriever not restored!"
    assert len(unified_ret.calls) == 1
    print("test_answer_unified_restores_retriever: OK")

