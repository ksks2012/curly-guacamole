"""PipelineContext — shared mutable state threaded through every pipeline step."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.retrieval.base import RetrievalResult


@dataclass
class PipelineContext:
    """Carries all retrieval state as it flows through the pipeline.

    Attributes
    ----------
    query          : The original user query (never modified by steps).
    top_k          : Final number of results to return.
    fetch_k        : Candidate pool size requested from each retriever.
    filters        : Optional Chroma ``where`` dict forwarded to retrievers.
    skip_expansion : When True, QueryExpansionStep is a no-op.
    queries        : Working list of queries.  Initialised to ``[query]``.
                     QueryExpansionStep appends alternative phrasings here.
    result_lists   : One ``list[RetrievalResult]`` per (retriever x query)
                     pair.  RetrieveStep appends to this list.
    candidates     : Flat, deduplicated pool after DeduplicateStep or RRFStep.
    results        : Final ranked results after RerankerStep (or slice of
                     candidates when no reranker is present).
    """

    query: str
    top_k: int = 5
    fetch_k: int = 20
    filters: dict | None = None
    skip_expansion: bool = False

    queries: list[str] = field(default_factory=list)
    result_lists: list[list[RetrievalResult]] = field(default_factory=list)
    candidates: list[RetrievalResult] = field(default_factory=list)
    results: list[RetrievalResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.queries:
            self.queries = [self.query]

    @property
    def final(self) -> list[RetrievalResult]:
        """Return results if populated by a RerankerStep, else truncate candidates."""
        if self.results:
            return self.results
        return self.candidates[: self.top_k]
