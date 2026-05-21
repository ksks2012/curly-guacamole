"""
Smoke tests for Step 1.4 — Retrieval Pipeline Plugin System.

Tests all pipeline components without a real embedding server or LLM:
  - PipelineContext: initialisation, final property
  - PipelineStep Protocol conformance for all concrete steps
  - QueryExpansionStep: appends queries, handles LLM failure gracefully
  - RetrieveStep: calls retriever for every query, fills result_lists
  - DeduplicateStep: dedup by unique_key, score-sorted
  - RRFStep: RRF formula, correct top-k, cross-list deduplication
  - RerankerStep: wraps/unwraps LangChain Documents, fills ctx.results
  - RetrievalPipeline.run(): threads context through all steps
  - PipelineBuilder fluent API: step_names reflect configuration
  - PipelineBuilder.document_pipeline / code_pipeline / unified_pipeline factories
  - RAGEngine._resolve_pipeline(): bare retriever auto-wraps to pipeline
  - unified_pipeline has RRFStep after attach_code_retriever()
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock

from rag.retrieval.base import RetrievalResult
from rag.retrieval.pipeline.context import PipelineContext
from rag.retrieval.pipeline.step import PipelineStep
from rag.retrieval.pipeline.steps import (
    DeduplicateStep,
    QueryExpansionStep,
    RerankerStep,
    RetrieveStep,
    RRFStep,
)
from rag.retrieval.pipeline.pipeline import RetrievalPipeline, PipelineBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(content: str, score: float = 0.9, source="document",
        chunk_id: str | None = None) -> RetrievalResult:
    meta = {"chunk_id": chunk_id} if chunk_id else {}
    return RetrievalResult(content=content, score=score, source=source, metadata=meta)


class _MockRetriever:
    name = "mock"
    def __init__(self, results=None):
        self._results = results or []
        self.calls = []
    def search(self, query, top_k=5, filters=None):
        self.calls.append((query, top_k, filters))
        return self._results[:top_k]


def _mock_llm(json_response='["alt query"]'):
    llm = MagicMock()
    resp = MagicMock(); resp.content = json_response
    llm.invoke.return_value = resp
    return llm


# ---------------------------------------------------------------------------
# PipelineContext
# ---------------------------------------------------------------------------

def test_context_init():
    ctx = PipelineContext(query="hello", top_k=3)
    assert ctx.queries == ["hello"]
    assert ctx.result_lists == []
    assert ctx.candidates == []
    assert ctx.results == []
    assert ctx.final == []
    print("test_context_init: OK")


def test_context_final_uses_candidates_when_no_results():
    ctx = PipelineContext(query="q", top_k=2)
    ctx.candidates = [_r("a"), _r("b"), _r("c")]
    assert len(ctx.final) == 2
    print("test_context_final_uses_candidates_when_no_results: OK")


def test_context_final_prefers_results():
    ctx = PipelineContext(query="q", top_k=5)
    ctx.candidates = [_r("a"), _r("b")]
    ctx.results    = [_r("x")]
    assert ctx.final == [_r("x")]
    print("test_context_final_prefers_results: OK")


# ---------------------------------------------------------------------------
# PipelineStep Protocol conformance
# ---------------------------------------------------------------------------

def test_step_protocol_conformance():
    for step in [DeduplicateStep(), RRFStep(), RetrieveStep(_MockRetriever())]:
        assert isinstance(step, PipelineStep), f"{type(step).__name__} not PipelineStep"
    print("test_step_protocol_conformance: OK")


# ---------------------------------------------------------------------------
# QueryExpansionStep
# ---------------------------------------------------------------------------

def test_query_expansion_appends_queries():
    llm = _mock_llm('["alt A", "alt B"]')
    ctx = PipelineContext(query="original")
    QueryExpansionStep(llm, n=2).run(ctx)
    assert "alt A" in ctx.queries
    assert "alt B" in ctx.queries
    assert ctx.queries[0] == "original"
    print("test_query_expansion_appends_queries: OK")


def test_query_expansion_skip_flag():
    llm = _mock_llm('["alt"]')
    ctx = PipelineContext(query="q", skip_expansion=True)
    QueryExpansionStep(llm, n=1).run(ctx)
    assert ctx.queries == ["q"]
    llm.invoke.assert_not_called()
    print("test_query_expansion_skip_flag: OK")


def test_query_expansion_graceful_failure():
    llm = _mock_llm("NOT JSON")
    ctx = PipelineContext(query="q")
    QueryExpansionStep(llm, n=2).run(ctx)  # must not raise
    assert ctx.queries == ["q"]
    print("test_query_expansion_graceful_failure: OK")


# ---------------------------------------------------------------------------
# RetrieveStep
# ---------------------------------------------------------------------------

def test_retrieve_step_one_query():
    ret = _MockRetriever([_r("doc1", chunk_id="1"), _r("doc2", chunk_id="2")])
    ctx = PipelineContext(query="q", fetch_k=5)
    RetrieveStep(ret).run(ctx)
    assert len(ctx.result_lists) == 1
    assert len(ctx.result_lists[0]) == 2
    assert ret.calls[0][0] == "q"
    print("test_retrieve_step_one_query: OK")


def test_retrieve_step_multiple_queries():
    ret = _MockRetriever([_r("x", chunk_id="x")])
    ctx = PipelineContext(query="q1")
    ctx.queries = ["q1", "q2", "q3"]
    RetrieveStep(ret).run(ctx)
    assert len(ctx.result_lists) == 3
    print("test_retrieve_step_multiple_queries: OK")


def test_retrieve_step_error_resilience():
    class _BadRetriever:
        name = "bad"
        def search(self, *a, **kw): raise RuntimeError("db down")
    ctx = PipelineContext(query="q")
    RetrieveStep(_BadRetriever()).run(ctx)  # must not raise
    assert ctx.result_lists == [[]]
    print("test_retrieve_step_error_resilience: OK")


# ---------------------------------------------------------------------------
# DeduplicateStep
# ---------------------------------------------------------------------------

def test_deduplicate_step_basic():
    ctx = PipelineContext(query="q")
    ctx.result_lists = [
        [_r("a", score=0.5, chunk_id="a"), _r("b", score=0.9, chunk_id="b")],
        [_r("b", score=0.1, chunk_id="b"), _r("c", score=0.7, chunk_id="c")],
    ]
    DeduplicateStep().run(ctx)
    keys = [r.unique_key() for r in ctx.candidates]
    assert len(keys) == len(set(keys)), "duplicate found"
    assert ctx.candidates[0].score >= ctx.candidates[-1].score
    print("test_deduplicate_step_basic: OK")


# ---------------------------------------------------------------------------
# RRFStep
# ---------------------------------------------------------------------------

def test_rrf_step_formula():
    ctx = PipelineContext(query="q", top_k=5)
    ctx.result_lists = [
        [_r("a", chunk_id="a")],
        [_r("b", chunk_id="b")],
    ]
    RRFStep(weights=[1.0, 1.0], rrf_k=60).run(ctx)
    expected = round(1.0 / 61, 6)
    for r in ctx.candidates:
        assert abs(r.score - expected) < 1e-5, f"score {r.score} != {expected}"
    print("test_rrf_step_formula: OK")


def test_rrf_step_cross_list_dedup():
    ctx = PipelineContext(query="q", top_k=10)
    ctx.result_lists = [
        [_r("shared", chunk_id="s"), _r("only-a", chunk_id="a")],
        [_r("shared", chunk_id="s"), _r("only-b", chunk_id="b")],
    ]
    RRFStep().run(ctx)
    keys = [r.unique_key() for r in ctx.candidates]
    assert keys.count("s") == 1
    assert keys[0] == "s"
    print("test_rrf_step_cross_list_dedup: OK")


def test_rrf_step_weight_imbalance():
    ctx = PipelineContext(query="q", top_k=2)
    ctx.result_lists = [
        [_r("heavy", chunk_id="h")],
        [_r("light", chunk_id="l")],
    ]
    RRFStep(weights=[10.0, 1.0]).run(ctx)
    assert ctx.candidates[0].unique_key() == "h"
    print("test_rrf_step_weight_imbalance: OK")


def test_rrf_step_weight_auto_padding():
    ctx = PipelineContext(query="q", top_k=10)
    ctx.result_lists = [
        [_r("a", chunk_id="a")],
        [_r("b", chunk_id="b")],
        [_r("c", chunk_id="c")],
    ]
    RRFStep(weights=[2.0]).run(ctx)  # only one weight provided
    assert len(ctx.candidates) == 3
    print("test_rrf_step_weight_auto_padding: OK")


# ---------------------------------------------------------------------------
# RerankerStep
# ---------------------------------------------------------------------------

def test_reranker_step():
    from langchain_core.documents import Document as _Doc
    reranker = MagicMock()
    reranker.rerank_with_scores.return_value = [
        (_Doc(page_content="best", metadata={"chunk_id": "b"}), 0.99),
    ]
    ctx = PipelineContext(query="q", top_k=1)
    ctx.candidates = [_r("best", chunk_id="b"), _r("worst", chunk_id="w")]
    RerankerStep(reranker).run(ctx)
    assert len(ctx.results) == 1
    assert ctx.results[0].content == "best"
    assert ctx.results[0].score == 0.99
    print("test_reranker_step: OK")


# ---------------------------------------------------------------------------
# RetrievalPipeline
# ---------------------------------------------------------------------------

def test_pipeline_threads_context():
    ret = _MockRetriever([_r("result", chunk_id="r1")])
    llm = _mock_llm('["alt"]')
    pipeline = (
        PipelineBuilder(ret, name="test")
        .with_expansion(llm, n=1)
        .with_deduplicate()
        .build()
    )
    results = pipeline.run("question", top_k=5, fetch_k=5)
    assert len(results) == 1
    assert results[0].content == "result"
    assert len(ret.calls) == 2   # original + 1 expanded
    print("test_pipeline_threads_context: OK")


def test_pipeline_step_names():
    ret = _MockRetriever()
    p = PipelineBuilder(ret).with_rrf().build()
    assert "retrieve" in p.step_names
    assert "rrf" in p.step_names
    print("test_pipeline_step_names: OK")


def test_pipeline_skip_expansion():
    ret = _MockRetriever([_r("x", chunk_id="x")])
    llm = _mock_llm('["alt"]')
    pipeline = PipelineBuilder(ret).with_expansion(llm, n=2).with_deduplicate().build()
    pipeline.run("q", skip_expansion=True)
    assert len(ret.calls) == 1   # only original
    print("test_pipeline_skip_expansion: OK")


# ---------------------------------------------------------------------------
# PipelineBuilder factories
# ---------------------------------------------------------------------------

def test_document_pipeline_factory():
    ret = _MockRetriever()
    p = PipelineBuilder.document_pipeline(ret, name="doc_test")
    assert "retrieve" in p.step_names
    assert "deduplicate" in p.step_names
    assert p.name == "doc_test"
    print("test_document_pipeline_factory: OK")


def test_document_pipeline_factory_with_rrf():
    ret = _MockRetriever()
    p = PipelineBuilder.document_pipeline(ret, use_rrf=True)
    assert "rrf" in p.step_names
    print("test_document_pipeline_factory_with_rrf: OK")


def test_code_pipeline_factory():
    ret = _MockRetriever()
    p = PipelineBuilder.code_pipeline(ret)
    assert p.name == "code"
    assert "deduplicate" in p.step_names
    print("test_code_pipeline_factory: OK")


def test_unified_pipeline_factory():
    r1 = _MockRetriever([_r("doc", chunk_id="d")])
    r2 = _MockRetriever([_r("code", chunk_id="c", source="code")])
    p = PipelineBuilder.unified_pipeline([r1, r2])
    assert p.name == "unified"
    assert "rrf" in p.step_names
    results = p.run("q", top_k=5, fetch_k=5)
    assert len(results) == 2
    sources = {r.source for r in results}
    assert sources == {"document", "code"}
    print("test_unified_pipeline_factory: OK")


# ---------------------------------------------------------------------------
# RAGEngine auto-pipeline wrapping
# ---------------------------------------------------------------------------

def test_ragengine_auto_wraps_bare_retriever():
    from rag.engine import RAGEngine
    from rag.retrieval.pipeline import RetrievalPipeline

    ret = _MockRetriever([_r("r", chunk_id="r1")])
    cfg = MagicMock()
    cfg.query_expansion_enabled = False
    cfg.query_expansion_n = 3
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="ans")

    engine = RAGEngine(llm=llm, retriever=ret, reranker=None, config=cfg)
    engine.answer("test", k=1, fetch_k=5)

    pipeline = engine._resolve_pipeline()
    assert isinstance(pipeline, RetrievalPipeline)
    assert "retrieve" in pipeline.step_names
    print("test_ragengine_auto_wraps_bare_retriever: OK")


def test_ragengine_uses_pipeline_directly():
    from rag.engine import RAGEngine

    ret = _MockRetriever([_r("r", chunk_id="r1")])
    p = PipelineBuilder.document_pipeline(ret, name="pre_built")
    cfg = MagicMock()
    cfg.query_expansion_enabled = False
    cfg.query_expansion_n = 3
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="ans")

    engine = RAGEngine(llm=llm, retriever=p, reranker=None, config=cfg)
    assert engine._resolve_pipeline() is p
    print("test_ragengine_uses_pipeline_directly: OK")


# ---------------------------------------------------------------------------
# Pipeline attributes (structural — no network calls)
# ---------------------------------------------------------------------------

def test_doc_pipeline_is_retrieval_pipeline():
    from rag.retrieval.pipeline import RetrievalPipeline
    from rag.retrieval.document_retriever import DocumentRetriever

    mock_searcher = MagicMock()
    mock_searcher.similarity_search_with_scores.return_value = []
    doc_ret = DocumentRetriever(mock_searcher)
    p = PipelineBuilder.document_pipeline(doc_ret)
    assert isinstance(p, RetrievalPipeline)
    assert "retrieve" in p.step_names
    print("test_doc_pipeline_is_retrieval_pipeline: OK")


def test_rebuild_unified_uses_rrf():
    from rag.retrieval.pipeline import RetrievalPipeline
    from rag.retrieval.document_retriever import DocumentRetriever
    from rag.retrieval.code_retriever import CodeRetriever

    mock_searcher = MagicMock()
    mock_searcher.similarity_search_with_scores.return_value = []
    doc_ret = DocumentRetriever(mock_searcher)

    mock_ci = MagicMock()
    mock_ci.search_with_scores.return_value = []
    code_ret = CodeRetriever(mock_ci)

    unified = PipelineBuilder.unified_pipeline([doc_ret, code_ret])
    assert isinstance(unified, RetrievalPipeline)
    assert "rrf" in unified.step_names
    print("test_rebuild_unified_uses_rrf: OK")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_context_init,
        test_context_final_uses_candidates_when_no_results,
        test_context_final_prefers_results,
        test_step_protocol_conformance,
        test_query_expansion_appends_queries,
        test_query_expansion_skip_flag,
        test_query_expansion_graceful_failure,
        test_retrieve_step_one_query,
        test_retrieve_step_multiple_queries,
        test_retrieve_step_error_resilience,
        test_deduplicate_step_basic,
        test_rrf_step_formula,
        test_rrf_step_cross_list_dedup,
        test_rrf_step_weight_imbalance,
        test_rrf_step_weight_auto_padding,
        test_reranker_step,
        test_pipeline_threads_context,
        test_pipeline_step_names,
        test_pipeline_skip_expansion,
        test_document_pipeline_factory,
        test_document_pipeline_factory_with_rrf,
        test_code_pipeline_factory,
        test_unified_pipeline_factory,
        test_ragengine_auto_wraps_bare_retriever,
        test_ragengine_uses_pipeline_directly,
        test_doc_pipeline_is_retrieval_pipeline,
        test_rebuild_unified_uses_rrf,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"{t.__name__}: FAIL — {exc}")
            import traceback; traceback.print_exc()
            failed += 1
    print()
    if failed:
        print(f"FAIL ({failed}/{len(tests)} tests failed)")
        sys.exit(1)
    else:
        print("PASS")
