"""RetrievalPipeline and PipelineBuilder.

RetrievalPipeline
-----------------
An ordered sequence of PipelineStep objects.  ``run()`` threads a fresh
PipelineContext through every step and returns the final results.

PipelineBuilder
---------------
Fluent builder for common configurations plus factory class methods for the
canonical document, code, and unified pipelines.

    # Fluent builder
    pipeline = (
        PipelineBuilder(retriever)
        .with_expansion(llm, n=3)
        .with_rrf()
        .with_reranker(cross_encoder)
        .build()
    )

    # Factory helpers
    pipeline = PipelineBuilder.document_pipeline(searcher, reranker=reranker)
    pipeline = PipelineBuilder.code_pipeline(code_indexer)
    pipeline = PipelineBuilder.unified_pipeline(
        [doc_retriever, code_retriever],
        weights=[0.6, 0.4],
        reranker=reranker,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.retrieval.pipeline.context import PipelineContext
from rag.retrieval.pipeline.step import PipelineStep
from rag.retrieval.pipeline.steps import (
    DeduplicateStep,
    QueryExpansionStep,
    RerankerStep,
    RetrieveStep,
    RRFStep,
)
from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from rag.reranker import BaseReranker
    from rag.retrieval.base import BaseRetriever

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# RetrievalPipeline
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """Executes an ordered list of PipelineStep objects on a PipelineContext.

    Parameters
    ----------
    steps : Ordered list of steps.  Each must satisfy PipelineStep Protocol.
    name  : Optional label for logging.
    """

    def __init__(self, steps: list[PipelineStep], *, name: str = "pipeline") -> None:
        self._steps = steps
        self.name   = name

    def run(
        self,
        query: str,
        *,
        top_k: int = 5,
        fetch_k: int = 20,
        filters: dict | None = None,
        skip_expansion: bool = False,
    ) -> list:
        """Execute the pipeline and return the final list[RetrievalResult].

        Parameters
        ----------
        query          : User query string.
        top_k          : Number of results to return.
        fetch_k        : Candidate pool per retriever call.
        filters        : Chroma ``where`` dict forwarded to all RetrieveStep instances.
        skip_expansion : Override to bypass QueryExpansionStep for this call.
        """
        ctx = PipelineContext(
            query=query,
            top_k=top_k,
            fetch_k=fetch_k,
            filters=filters,
            skip_expansion=skip_expansion,
        )
        for step in self._steps:
            log.debug("Pipeline %r → step %r", self.name, step.name)
            ctx = step.run(ctx)
        return ctx.final

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self._steps]

    def __repr__(self) -> str:
        return f"RetrievalPipeline(name={self.name!r}, steps={self.step_names})"


# ---------------------------------------------------------------------------
# PipelineBuilder
# ---------------------------------------------------------------------------

class PipelineBuilder:
    """Fluent builder that assembles a RetrievalPipeline step by step.

    Usage
    -----
    pipeline = (
        PipelineBuilder(retriever, name="doc")
        .with_expansion(llm, n=3)
        .with_rrf()
        .with_reranker(cross_encoder)
        .build()
    )
    """

    def __init__(self, retriever: "BaseRetriever", *, name: str = "pipeline") -> None:
        self._retriever = retriever
        self._name      = name
        self._expansion: QueryExpansionStep | None = None
        self._fusion: DeduplicateStep | RRFStep | None = None
        self._reranker_step: RerankerStep | None = None

    # ── Fluent configuration ──────────────────────────────────────────────

    def with_expansion(self, llm: "ChatOpenAI", n: int = 3) -> "PipelineBuilder":
        """Add LLM-based query expansion before retrieval."""
        self._expansion = QueryExpansionStep(llm, n=n)
        return self

    def with_rrf(self, weights: list[float] | None = None,
                 rrf_k: int = 60) -> "PipelineBuilder":
        """Use Reciprocal Rank Fusion to merge result lists (replaces deduplicate)."""
        self._fusion = RRFStep(weights=weights, rrf_k=rrf_k)
        return self

    def with_deduplicate(self) -> "PipelineBuilder":
        """Use score-sorted deduplication (replaces RRF)."""
        self._fusion = DeduplicateStep()
        return self

    def with_reranker(self, reranker: "BaseReranker") -> "PipelineBuilder":
        """Add a cross-encoder reranker as the final step."""
        self._reranker_step = RerankerStep(reranker)
        return self

    def build(self) -> RetrievalPipeline:
        """Assemble and return the configured RetrievalPipeline."""
        steps: list[PipelineStep] = []
        if self._expansion:
            steps.append(self._expansion)
        steps.append(RetrieveStep(self._retriever))
        steps.append(self._fusion if self._fusion is not None else DeduplicateStep())
        if self._reranker_step:
            steps.append(self._reranker_step)
        return RetrievalPipeline(steps, name=self._name)

    # ── Factory helpers ───────────────────────────────────────────────────

    @classmethod
    def document_pipeline(
        cls,
        retriever: "BaseRetriever",
        *,
        llm: "ChatOpenAI | None" = None,
        expansion_n: int = 3,
        reranker: "BaseReranker | None" = None,
        use_rrf: bool = False,
        name: str = "document",
    ) -> RetrievalPipeline:
        """Build the standard document-only pipeline.

        Parameters
        ----------
        retriever   : DocumentRetriever (or any BaseRetriever).
        llm         : When supplied, adds QueryExpansionStep.
        expansion_n : Number of extra phrasings for expansion.
        reranker    : When supplied, adds RerankerStep.
        use_rrf     : Use RRFStep instead of DeduplicateStep.
        """
        b = cls(retriever, name=name)
        if llm:
            b.with_expansion(llm, n=expansion_n)
        if use_rrf:
            b.with_rrf()
        else:
            b.with_deduplicate()
        if reranker:
            b.with_reranker(reranker)
        return b.build()

    @classmethod
    def code_pipeline(
        cls,
        retriever: "BaseRetriever",
        *,
        llm: "ChatOpenAI | None" = None,
        expansion_n: int = 3,
        reranker: "BaseReranker | None" = None,
        name: str = "code",
    ) -> RetrievalPipeline:
        """Build a code-search pipeline (vector only, optional expansion + rerank)."""
        b = cls(retriever, name=name)
        if llm:
            b.with_expansion(llm, n=expansion_n)
        b.with_deduplicate()
        if reranker:
            b.with_reranker(reranker)
        return b.build()

    @classmethod
    def unified_pipeline(
        cls,
        retrievers: list["BaseRetriever"],
        *,
        weights: list[float] | None = None,
        llm: "ChatOpenAI | None" = None,
        expansion_n: int = 3,
        reranker: "BaseReranker | None" = None,
        rrf_k: int = 60,
        name: str = "unified",
    ) -> RetrievalPipeline:
        """Build a unified pipeline that merges multiple retrievers via RRF.

        The expand → retrieve × N → RRF → rerank structure mirrors the
        previous HybridRetriever behaviour but is fully inspectable and
        replaceable at each step.

        Parameters
        ----------
        retrievers : List of BaseRetriever instances (doc + code + …).
        weights    : Per-retriever RRF weights.
        llm        : When supplied, query expansion is applied once and all
                     expanded queries are passed to every RetrieveStep.
        expansion_n: Number of expansion phrasings.
        reranker   : Final cross-encoder applied after RRF.
        rrf_k      : RRF dampening constant.
        """
        steps: list[PipelineStep] = []
        if llm:
            steps.append(QueryExpansionStep(llm, n=expansion_n))
        for i, ret in enumerate(retrievers):
            label = getattr(ret, "name", None) or type(ret).__name__.lower()
            step = RetrieveStep(ret)
            step.name = f"retrieve_{label}_{i}"   # type: ignore[misc]
            steps.append(step)
        steps.append(RRFStep(weights=weights, rrf_k=rrf_k))
        if reranker:
            steps.append(RerankerStep(reranker))
        return RetrievalPipeline(steps, name=name)
