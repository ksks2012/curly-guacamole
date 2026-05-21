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

from langchain_core.documents import Document

from rag.retrieval.base import RetrievalResult
from rag.retrieval.bm25 import BM25Index, rrf_fuse
from rag.code.tokenizer import code_tokenize
from utils.logger import AppLogger

if TYPE_CHECKING:
    from rag.code.indexer import CodeIndexer
    from rag.reranker import BaseReranker

log = AppLogger.get(__name__)

_VALID_LEVELS = frozenset({"file", "symbol", "block"})


class CodeRetriever:
    """Adapts CodeIndexer to the BaseRetriever Protocol.

    Parameters
    ----------
    indexer     : Existing ``CodeIndexer`` instance.
    level       : Which Chroma collection to search.  One of:
                  ``"file"`` (per-file module summary, in the ``documents`` collection),
                  ``"symbol"`` (class/function/method — default, in the ``symbols`` collection),
                  ``"block"`` (full code text, in the ``code_block`` collection).
    fetch_k     : Candidate pool size fetched before optional reranking.
    reranker    : Optional reranker applied after vector retrieval.
    use_hybrid  : When True, fuse vector search with symbol-aware BM25 via RRF.
                  Call ``build_bm25()`` once after indexing to populate the index.
    """

    def __init__(
        self,
        indexer: "CodeIndexer",
        *,
        level: str = "symbol",
        fetch_k: int = 20,
        reranker: "BaseReranker | None" = None,
        use_hybrid: bool = False,
    ) -> None:
        if level not in _VALID_LEVELS:
            raise ValueError(
                f"Invalid level {level!r}.  Choose from: {sorted(_VALID_LEVELS)}"
            )
        self._indexer   = indexer
        self._level     = level
        self._fetch_k   = fetch_k
        self._reranker  = reranker
        self._bm25: BM25Index | None = BM25Index(tokenizer=code_tokenize) if use_hybrid else None
        self._bm25_built: bool = False

    # ── BM25 management ───────────────────────────────────────────────────

    def build_bm25(self) -> None:
        """Fetch all documents from the Chroma collection and (re)build the BM25 index.

        Call this once after indexing is complete, and again whenever the
        collection is updated (or call ``invalidate_bm25()`` to trigger a lazy
        rebuild on the next ``search()`` call).
        """
        if self._bm25 is None:
            self._bm25 = BM25Index(tokenizer=code_tokenize)
        try:
            raw = self._indexer._db(self._level).get(include=["documents", "metadatas"])
        except Exception as exc:
            log.warning("CodeRetriever.build_bm25: failed to fetch docs — %s", exc)
            return
        docs = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(
                raw.get("documents", []),
                raw.get("metadatas", []) or [],
            )
        ]
        self._bm25.build(docs)
        self._bm25_built = True
        log.debug("CodeRetriever.build_bm25: %d docs indexed for level=%r", len(docs), self._level)

    def invalidate_bm25(self) -> None:
        """Mark the BM25 index as stale so it is rebuilt on the next search."""
        self._bm25_built = False

    # ── BaseRetriever interface ───────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Search the code collection at the configured *level*.

        When ``use_hybrid=True`` (set at construction time), vector search and
        symbol-aware BM25 are both run and fused via RRF before reranking.
        The BM25 index is built lazily on the first call if ``build_bm25()``
        has not been called explicitly.

        Parameters
        ----------
        query   : Natural-language or identifier-style query.
        top_k   : Maximum results to return.
        filters : Optional Chroma ``where`` dict, e.g.
                  ``{"repo_id": {"$eq": "my-repo"}}`` or
                  ``{"language": {"$eq": "Python"}}``.
        """
        fetch_k = max(self._fetch_k, top_k)

        # ── Hybrid path (vector + BM25 via RRF) ──────────────────────────
        if self._bm25 is not None:
            if not self._bm25_built:
                self.build_bm25()
            vector_pairs = self._indexer.search_with_scores(
                query, level=self._level, k=fetch_k, filter=filters
            )
            bm25_pairs = self._bm25.search(query, k=fetch_k, where=filters)
            pairs = rrf_fuse(vector_pairs, bm25_pairs, top_k=fetch_k)
        else:
            # ── Vector-only path ─────────────────────────────────────────
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
