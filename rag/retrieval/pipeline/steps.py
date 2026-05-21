"""Concrete pipeline steps.

All steps satisfy the PipelineStep Protocol: they expose a ``name`` property
and a ``run(ctx) → ctx`` method that mutates the shared PipelineContext.

Catalogue
---------
QueryExpansionStep  : LLM-based query expansion (appends to ctx.queries)
RetrieveStep        : calls a BaseRetriever for every query (fills ctx.result_lists)
DeduplicateStep     : flat-merge + dedup by unique_key() → ctx.candidates
RRFStep             : Reciprocal Rank Fusion across result_lists → ctx.candidates
RerankerStep        : cross-encoder rerank of ctx.candidates → ctx.results
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rag.retrieval.pipeline.context import PipelineContext
from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from rag.reranker import BaseReranker
    from rag.retrieval.base import BaseRetriever

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# QueryExpansionStep
# ---------------------------------------------------------------------------

class QueryExpansionStep:
    """Expand the original query into N alternative phrasings via LLM.

    Appends extra phrasings to ``ctx.queries`` so subsequent RetrieveStep
    calls cover a wider semantic neighbourhood.

    Skipped when ``ctx.skip_expansion`` is True.
    """

    name = "query_expansion"

    def __init__(self, llm: "ChatOpenAI", n: int = 3) -> None:
        self._llm = llm
        self._n   = n

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.skip_expansion:
            return ctx
        try:
            from rag.prompt import QUERY_EXPANSION_PROMPT
            prompt   = QUERY_EXPANSION_PROMPT.format(question=ctx.query, n=self._n)
            response = self._llm.invoke(prompt)
            raw      = response.content if hasattr(response, "content") else str(response)
            expanded = json.loads(raw.strip())
            if isinstance(expanded, list):
                extras = [str(q) for q in expanded if str(q) != ctx.query]
                ctx.queries.extend(extras)
                log.info("QueryExpansionStep: added %d phrasings", len(extras))
        except Exception as exc:
            log.warning("QueryExpansionStep: failed (%s), continuing with original query", exc)
        return ctx


# ---------------------------------------------------------------------------
# RetrieveStep
# ---------------------------------------------------------------------------

class RetrieveStep:
    """Call a BaseRetriever for every query in ctx.queries.

    Appends one ``list[RetrievalResult]`` per query to ``ctx.result_lists``.
    Errors from the retriever are caught and logged so the pipeline continues.
    """

    name = "retrieve"

    def __init__(self, retriever: "BaseRetriever") -> None:
        self._retriever = retriever

    def run(self, ctx: PipelineContext) -> PipelineContext:
        fetch_k = max(ctx.fetch_k, ctx.top_k)
        for q in ctx.queries:
            try:
                results = self._retriever.search(q, top_k=fetch_k, filters=ctx.filters)
            except Exception as exc:
                log.warning(
                    "RetrieveStep: retriever %s failed for query %r: %s",
                    type(self._retriever).__name__, q, exc,
                )
                results = []
            ctx.result_lists.append(results)
            log.debug(
                "RetrieveStep: %s / q=%r → %d results",
                type(self._retriever).__name__, q[:60], len(results),
            )
        return ctx


# ---------------------------------------------------------------------------
# DeduplicateStep
# ---------------------------------------------------------------------------

class DeduplicateStep:
    """Flatten result_lists into ctx.candidates, deduplicating by unique_key().

    The first occurrence of each key is preserved (earlier lists have
    priority).  Candidates are sorted by descending score.

    Use this when you have a single retriever or when you want score-sorted
    (rather than RRF-ranked) fusion.
    """

    name = "deduplicate"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        seen: set[str] = set()
        flat = []
        for result_list in ctx.result_lists:
            for r in result_list:
                k = r.unique_key()
                if k not in seen:
                    seen.add(k)
                    flat.append(r)
        ctx.candidates = sorted(flat, key=lambda r: r.score, reverse=True)
        log.debug("DeduplicateStep: %d candidates after dedup", len(ctx.candidates))
        return ctx


# ---------------------------------------------------------------------------
# RRFStep
# ---------------------------------------------------------------------------

class RRFStep:
    """Reciprocal Rank Fusion across all lists in ctx.result_lists.

    Produces a single fused, rank-sorted list in ctx.candidates.

    Each list can carry a weight (default 1.0).  The formula is:

        score(d) = Σ_i  weight_i / (rrf_k + rank_i(d))

    This is the canonical way to merge heterogeneous result lists (vector,
    BM25, different collections) when scores are not comparable.

    Parameters
    ----------
    weights : Per-list weights.  When None, all lists are weighted equally.
              The list is cycled / truncated to match the number of result_lists
              at call time, so it is safe to supply fewer weights than lists.
    rrf_k   : RRF dampening constant (default 60).
    """

    name = "rrf"

    def __init__(
        self,
        weights: list[float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._weights = weights
        self._rrf_k   = rrf_k

    def run(self, ctx: PipelineContext) -> PipelineContext:
        result_lists = ctx.result_lists
        if not result_lists:
            ctx.candidates = []
            return ctx

        n = len(result_lists)
        weights = list(self._weights) if self._weights else [1.0] * n
        # Pad or trim weights to match actual number of lists
        if len(weights) < n:
            weights += [1.0] * (n - len(weights))
        weights = weights[:n]

        scores: dict[str, float]           = {}
        items:  dict[str, object]          = {}   # key → RetrievalResult

        for result_list, weight in zip(result_lists, weights):
            for rank, result in enumerate(result_list, start=1):
                key = result.unique_key()
                scores[key] = scores.get(key, 0.0) + weight / (self._rrf_k + rank)
                if key not in items:
                    items[key] = result

        from rag.retrieval.base import RetrievalResult
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ctx.candidates = [
            RetrievalResult(
                content=items[key].content,      # type: ignore[union-attr]
                score=round(score, 6),
                source=items[key].source,        # type: ignore[union-attr]
                metadata=items[key].metadata,    # type: ignore[union-attr]
            )
            for key, score in fused
        ]
        log.debug("RRFStep: %d candidates after RRF fusion", len(ctx.candidates))
        return ctx


# ---------------------------------------------------------------------------
# RerankerStep
# ---------------------------------------------------------------------------

class RerankerStep:
    """Apply a cross-encoder reranker to ctx.candidates → ctx.results.

    Wraps ``BaseReranker.rerank_with_scores()`` which operates on LangChain
    ``Document`` objects; this step handles the conversion transparently.
    """

    name = "reranker"

    def __init__(self, reranker: "BaseReranker") -> None:
        self._reranker = reranker

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.candidates:
            return ctx

        from langchain_core.documents import Document as _Doc
        from rag.retrieval.base import RetrievalResult

        lc_docs = [_Doc(page_content=r.content, metadata=r.metadata) for r in ctx.candidates]
        key_to_source = {r.unique_key(): r.source for r in ctx.candidates}

        reranked = self._reranker.rerank_with_scores(ctx.query, lc_docs, top_k=ctx.top_k)

        ctx.results = [
            RetrievalResult(
                content=doc.page_content,
                score=score,
                source=key_to_source.get(
                    RetrievalResult(content=doc.page_content, score=0.0,
                                    source="document", metadata=doc.metadata).unique_key(),
                    "document",
                ),
                metadata=doc.metadata,
            )
            for doc, score in reranked
        ]
        log.debug("RerankerStep: %d results after rerank", len(ctx.results))
        return ctx
