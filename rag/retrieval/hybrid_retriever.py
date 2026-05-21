"""
HybridRetriever — merges results from multiple BaseRetriever instances.

Takes any combination of DocumentRetriever, CodeRetriever, or other
HybridRetriever instances and fuses their result lists using Reciprocal
Rank Fusion (RRF).  An optional reranker can be applied to the fused pool
before returning the final top-k results.

Why RRF?
--------
Each retriever produces scores on different scales (vector L2-derived vs BM25
vs reranker logits).  RRF is a parameter-free, rank-based merging strategy
that is robust to these scale differences:

    score(d) = Σ_i  w_i / (rrf_k + rank_i(d))

Documents absent from a list receive no contribution from it.
Deduplication is keyed on ``RetrievalResult.unique_key()``.

Usage
-----
>>> from rag.retrieval.hybrid_retriever import HybridRetriever
>>> from rag.retrieval.document_retriever import DocumentRetriever
>>> from rag.retrieval.code_retriever import CodeRetriever
>>>
>>> hybrid = HybridRetriever(
...     retrievers=[doc_ret, code_ret],
...     weights=[0.6, 0.4],       # optional per-retriever RRF weights
...     reranker=cross_encoder,   # optional final rerank
... )
>>> results = hybrid.search("How are embeddings generated?", top_k=8)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.retrieval.base import BaseRetriever, RetrievalResult
from utils.logger import AppLogger

if TYPE_CHECKING:
    from rag.reranker import BaseReranker

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# Internal: RRF merge
# ---------------------------------------------------------------------------

def _rrf_merge(
    result_lists: list[list[RetrievalResult]],
    weights: list[float],
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Fuse multiple result lists into one using Reciprocal Rank Fusion.

    Formula::

        score(d) = Σ_i  weight_i / (rrf_k + rank_i(d))

    where ``rank_i(d)`` is 1-based position in result list *i*.
    Documents absent from a list get zero contribution from it.

    Parameters
    ----------
    result_lists : One list of RetrievalResult per retriever.
    weights      : Per-list weight (must match length of result_lists).
    top_k        : Number of results to return after fusion.
    rrf_k        : Dampening constant (default 60).
    """
    scores: dict[str, float]          = {}
    items:  dict[str, RetrievalResult] = {}

    for result_list, weight in zip(result_lists, weights):
        for rank, result in enumerate(result_list, start=1):
            key = result.unique_key()
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            if key not in items:
                items[key] = result

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        RetrievalResult(
            content=items[key].content,
            score=round(score, 6),
            source=items[key].source,
            metadata=items[key].metadata,
        )
        for key, score in fused
    ]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Merges results from multiple BaseRetriever instances via RRF.

    Parameters
    ----------
    retrievers : List of retriever instances.  Any object that satisfies
                 the ``BaseRetriever`` Protocol is accepted.
    weights    : Optional per-retriever RRF weights.  When None, all
                 retrievers are weighted equally at ``1.0``.  Values do not
                 need to sum to 1 — only relative magnitudes matter.
    reranker   : Optional cross-encoder reranker applied to the fused pool.
                 When supplied, ``fetch_k`` candidates are fetched per
                 retriever and the reranker selects the final top-k.
    fetch_k    : How many results to request from each sub-retriever before
                 fusion.  A larger value improves recall at the cost of more
                 embedding comparisons.  Default: max(top_k * 3, 20).
                 Set to 0 to use the same value as top_k.
    rrf_k      : RRF dampening constant.  Default 60 (standard literature).
    """

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        *,
        weights:  list[float] | None = None,
        reranker: "BaseReranker | None" = None,
        fetch_k:  int = 0,
        rrf_k:    int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one retriever")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError(
                f"weights length ({len(weights)}) must match "
                f"retrievers length ({len(retrievers)})"
            )
        self._retrievers = retrievers
        self._weights    = weights or [1.0] * len(retrievers)
        self._reranker   = reranker
        self._fetch_k    = fetch_k
        self._rrf_k      = rrf_k

    # ── BaseRetriever interface ───────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search all sub-retrievers and return RRF-fused results.

        Parameters
        ----------
        query   : Natural-language query.
        top_k   : Final number of results to return.
        filters : Passed through to every sub-retriever unchanged.
                  Each retriever interprets the Chroma where-dict for its
                  own collection.
        """
        candidate_k = self._fetch_k if self._fetch_k > 0 else max(top_k * 3, 20)

        result_lists: list[list[RetrievalResult]] = []
        for retriever in self._retrievers:
            try:
                sub_results = retriever.search(query, top_k=candidate_k, filters=filters)
            except Exception as exc:
                log.warning(
                    "HybridRetriever: sub-retriever %s failed: %s",
                    type(retriever).__name__, exc,
                )
                sub_results = []
            result_lists.append(sub_results)
            log.debug(
                "HybridRetriever: %s → %d results",
                type(retriever).__name__, len(sub_results),
            )

        fused = _rrf_merge(result_lists, self._weights, top_k=candidate_k, rrf_k=self._rrf_k)
        log.debug("HybridRetriever: fused=%d  reranker=%s", len(fused), self._reranker is not None)

        if self._reranker is not None and fused:
            from langchain_core.documents import Document as _Doc
            docs = [_Doc(page_content=r.content, metadata=r.metadata) for r in fused]
            reranked = self._reranker.rerank_with_scores(query, docs, top_k=top_k)
            return [
                RetrievalResult(
                    content=doc.page_content,
                    score=score,
                    source=fused[i].source if i < len(fused) else "document",
                    metadata=doc.metadata,
                )
                for i, (doc, score) in enumerate(reranked)
            ]

        return fused[:top_k]

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def retriever_count(self) -> int:
        return len(self._retrievers)

    def summary(self) -> str:
        names = [type(r).__name__ for r in self._retrievers]
        return (
            f"HybridRetriever(retrievers=[{', '.join(names)}]  "
            f"weights={self._weights}  rrf_k={self._rrf_k})"
        )
