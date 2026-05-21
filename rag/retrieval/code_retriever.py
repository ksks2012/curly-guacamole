"""
CodeRetriever — adapts CodeIndexer to the BaseRetriever interface.

Wraps ``CodeIndexer`` (which manages four Chroma collections at repo / file /
symbol / block granularity) and converts its ``Document`` output to
``RetrievalResult`` objects so it can be used alongside ``DocumentRetriever``
or composed inside ``HybridRetriever``.

Default retrieval level is ``"symbol"`` because that collection captures the
*meaning* of a function/class without raw implementation noise — ideal for
"what does X do?" queries.  Use ``"block"`` when you need the exact source
code (e.g. for code completion or detailed reasoning).

Usage
-----
>>> from rag.retrieval.code_retriever import CodeRetriever
>>> from rag.code.indexer import CodeIndexer
>>>
>>> indexer   = CodeIndexer(persist_dir, embed_fn)
>>> retriever = CodeRetriever(indexer, level="symbol")
>>> results   = retriever.search("parse Python AST chunks", top_k=5)
>>> for r in results:
...     print(r.metadata.get("file_path"), r.score, r.content[:60])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.retrieval.base import RetrievalResult
from utils.logger import AppLogger

if TYPE_CHECKING:
    from rag.code.indexer import CodeIndexer
    from rag.reranker import BaseReranker

log = AppLogger.get(__name__)

_VALID_LEVELS = frozenset({"repo", "file", "symbol", "block"})


class CodeRetriever:
    """Adapts CodeIndexer to the BaseRetriever Protocol.

    Parameters
    ----------
    indexer   : Existing ``CodeIndexer`` instance.
    level     : Which Chroma collection to search.  One of:
                ``"repo"`` (architecture overview),
                ``"file"`` (per-file module summary),
                ``"symbol"`` (class/function/method — default),
                ``"block"`` (full code text).
    fetch_k   : Candidate pool size fetched before optional reranking.
    reranker  : Optional reranker applied after vector retrieval.
    """

    def __init__(
        self,
        indexer: "CodeIndexer",
        *,
        level: str = "symbol",
        fetch_k: int = 20,
        reranker: "BaseReranker | None" = None,
    ) -> None:
        if level not in _VALID_LEVELS:
            raise ValueError(
                f"Invalid level {level!r}.  Choose from: {sorted(_VALID_LEVELS)}"
            )
        self._indexer  = indexer
        self._level    = level
        self._fetch_k  = fetch_k
        self._reranker = reranker

    # ── BaseRetriever interface ───────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search the code collection at the configured *level*.

        Parameters
        ----------
        query   : Natural-language or identifier-style query.
        top_k   : Maximum results to return.
        filters : Optional Chroma ``where`` dict, e.g.
                  ``{"repo_id": {"$eq": "my-repo"}}`` or
                  ``{"language": {"$eq": "Python"}}``.
        """
        fetch_k = max(self._fetch_k, top_k)

        pairs = self._indexer.search_with_scores(
            query, level=self._level, k=fetch_k, filter=filters
        )

        if self._reranker is not None and pairs:
            docs     = [doc for doc, _ in pairs]
            reranked = self._reranker.rerank_with_scores(query, docs, top_k=top_k)
            return [
                RetrievalResult(
                    content=doc.page_content,
                    score=score,
                    source="code",
                    metadata=doc.metadata,
                )
                for doc, score in reranked
            ]

        return [
            RetrievalResult(
                content=doc.page_content,
                score=score,
                source="code",
                metadata=doc.metadata,
            )
            for doc, score in pairs[:top_k]
        ]
