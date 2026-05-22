"""
DocumentRetriever — adapts the existing Searcher to the BaseRetriever interface.

Wraps ``Searcher`` (which owns vector search, BM25, hybrid-RRF, and reranking
for the document collection) and converts its ``(Document, score)`` output to
``RetrievalResult`` objects so it can be used interchangeably with
``CodeRetriever`` or ``HybridRetriever``.

Supported retrieval modes (controlled at construction time):

    vector only (default):
        calls Searcher.similarity_search_with_scores()

    hybrid (vector + BM25 via RRF):
        calls Searcher.hybrid_search_with_scores()

Optional reranking is applied on the candidate pool before returning results.

Usage
-----
>>> retriever = DocumentRetriever(client.searcher, use_hybrid=True)
>>> results   = retriever.search("explain the ingestion pipeline", top_k=5)
>>> for r in results:
...     print(r.score, r.metadata.get("doc_id"), r.content[:60])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.retrieval.base import RetrievalResult
from utils.logger import AppLogger

if TYPE_CHECKING:
    from rag.reranker import BaseReranker
    from rag.retrieval.searcher import Searcher

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# Internal: thin adapter so raw Chroma where-dicts work with Searcher
# ---------------------------------------------------------------------------

class _RawFilter:
    """Satisfies the SearchFilter duck-type expected by Searcher methods."""

    def __init__(self, where: dict) -> None:
        self._where = where

    def is_empty(self) -> bool:
        return not self._where

    def to_chroma(self) -> dict | None:
        return self._where or None

    def summary(self) -> str:
        return repr(self._where)


# ---------------------------------------------------------------------------
# DocumentRetriever
# ---------------------------------------------------------------------------

class DocumentRetriever:
    """Adapts Searcher to the BaseRetriever Protocol.

    Parameters
    ----------
    searcher    : Existing ``Searcher`` instance (from LocalLlamaClient).
    use_hybrid  : When True, vector + BM25 results are merged via RRF before
                  returning.  When False (default), pure vector search is used.
    fetch_k     : Candidate pool size fetched before optional reranking.
                  Must be >= top_k.  Default 20.
    reranker    : Optional reranker applied after retrieval.  When supplied,
                  *fetch_k* candidates are reranked and the top *top_k* returned.
    """

    def __init__(
        self,
        searcher: "Searcher",
        *,
        use_hybrid: bool = False,
        fetch_k: int = 20,
        reranker: "BaseReranker | None" = None,
    ) -> None:
        self._searcher   = searcher
        self._use_hybrid = use_hybrid
        self._fetch_k    = fetch_k
        self._reranker   = reranker

    # ── BaseRetriever interface ───────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search the document collection.

        Parameters
        ----------
        query   : Natural-language query.
        top_k   : Maximum results to return.
        filters : Optional Chroma ``where`` dict.  Use
                  ``SearchFilter(...).to_chroma()`` to build it, or pass a raw
                  dict.  ``None`` disables filtering.
        """
        search_filter = _RawFilter(filters) if filters else None
        fetch_k = max(self._fetch_k, top_k)

        if self._use_hybrid:
            _, _, pairs = self._searcher.hybrid_search_with_scores(
                query, k=top_k, fetch_k=fetch_k, search_filter=search_filter
            )
        else:
            pairs = self._searcher.similarity_search_with_scores(
                query, k=fetch_k, search_filter=search_filter
            )

        if self._reranker is not None and pairs:
            docs     = [doc for doc, _ in pairs]
            reranked = self._reranker.rerank_with_scores(query, docs, top_k=top_k)
            return [
                RetrievalResult(
                    content=doc.page_content,
                    score=score,
                    source="document",
                    metadata=doc.metadata,
                )
                for doc, score in reranked
            ]

        return [
            RetrievalResult(
                content=doc.page_content,
                score=score,
                source="document",
                metadata=doc.metadata,
            )
            for doc, score in pairs[:top_k]
        ]
