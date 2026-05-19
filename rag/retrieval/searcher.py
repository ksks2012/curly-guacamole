"""
Searcher — all retrieval operations extracted from LocalLlamaClient.

Owns the BM25 index state and provides every search/listing method.
LocalLlamaClient holds a Searcher instance and delegates to it.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from rag.retrieval.bm25 import BM25Index, rrf_fuse
from rag.retrieval.filters import SearchFilter
from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from rag.reranker import BaseReranker

log = AppLogger.get(__name__)


class Searcher:
    """Retrieval layer: vector search, BM25, hybrid, trace, and metadata listing.

    Args:
        db       : Chroma collection (shared reference from LocalLlamaClient).
        reranker : Optional reranker instance (None = reranking disabled).
    """

    def __init__(self, db: "Chroma", reranker: "BaseReranker | None") -> None:
        self._db       = db
        self._reranker = reranker
        self.bm25_index  = BM25Index()
        self._bm25_dirty = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _where(
        self,
        doc_id: str | None = None,
        search_filter: SearchFilter | None = None,
    ) -> dict | None:
        if search_filter is not None and not search_filter.is_empty():
            return search_filter.to_chroma()
        if doc_id is not None:
            return {"doc_id": {"$eq": doc_id}}
        return None

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 4, doc_id: str | None = None
    ) -> list[Document]:
        kwargs: dict = {"k": k}
        where = self._where(doc_id=doc_id)
        if where:
            kwargs["filter"] = where
        return self._db.similarity_search(query, **kwargs)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        doc_id: str | None = None,
        search_filter: SearchFilter | None = None,
    ) -> list[tuple[Document, float]]:
        """Returns (Document, score) pairs sorted best-first.

        Chroma L2 distance is converted: ``relevance = 1 / (1 + distance)``.
        """
        log.debug(
            "similarity_search_with_scores: query=%r  k=%d  filter=%s",
            query, k, search_filter.summary() if search_filter else doc_id,
        )
        kwargs: dict = {"k": k}
        where = self._where(doc_id=doc_id, search_filter=search_filter)
        if where:
            kwargs["filter"] = where
        raw = self._db.similarity_search_with_score(query, **kwargs)
        log.debug("  raw results: %d  (L2 distances: %s)",
                  len(raw), [round(d, 4) for _, d in raw])
        return [(doc, round(1 / (1 + dist), 4)) for doc, dist in raw]

    # ------------------------------------------------------------------
    # Metadata listing
    # ------------------------------------------------------------------

    def list_doc_ids(self) -> list[str]:
        result = self._db.get(include=["metadatas"])
        return sorted({
            m.get("doc_id")
            for m in (result.get("metadatas") or [])
            if m and m.get("doc_id")
        })

    def list_doc_title_map(self) -> dict[str, str]:
        result  = self._db.get(include=["metadatas"])
        mapping: dict[str, str] = {}
        for m in (result.get("metadatas") or []):
            if not m:
                continue
            doc_id = m.get("doc_id")
            if not doc_id or doc_id in mapping:
                continue
            mapping[doc_id] = (m.get("title") or "").strip() or doc_id
        return dict(sorted(mapping.items()))

    # Fields stored as comma-separated values in Chroma — must be split when listing.
    # ka_topics/ka_keywords/ka_entities written by B.1; topic_id (scalar) by B.3.
    _CSV_FIELDS = {"tags", "topics", "ka_topics", "ka_keywords", "ka_entities"}

    def list_field_values(self, field: str) -> list[str]:
        result = self._db.get(include=["metadatas"])
        values: set[str] = set()
        for m in (result.get("metadatas") or []):
            if not m:
                continue
            raw = m.get(field, "")
            if not raw:
                continue
            if field in self._CSV_FIELDS:
                for t in str(raw).split(","):
                    t = t.strip()
                    if t:
                        values.add(t)
            else:
                values.add(str(raw))
        return sorted(values)

    def list_workspaces(self) -> list[str]:
        return self.list_field_values("workspace")

    def list_document_types(self) -> list[str]:
        return self.list_field_values("document_type")

    def list_tags(self) -> list[str]:
        return self.list_field_values("tags")

    def browse_chunks(
        self,
        doc_id: str | None = None,
        tag: str | None = None,
        topic: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Return up to *limit* chunks with content and full metadata.

        Each dict has keys ``content`` (page text) and ``metadata`` (raw dict).
        Filtering by *tag* and *topic* uses post-fetch matching because Chroma
        does not support contains-in-csv queries natively.
        """
        kwargs: dict = {"include": ["documents", "metadatas"]}
        if doc_id:
            kwargs["where"] = {"doc_id": {"$eq": doc_id}}
        result = self._db.get(**kwargs)
        docs  = result.get("documents") or []
        metas = result.get("metadatas") or []

        chunks: list[dict] = []
        for text, meta in zip(docs, metas):
            if not text:
                continue
            m = meta or {}
            if tag:
                raw_tags = [t.strip() for t in str(m.get("tags", "")).split(",") if t.strip()]
                if tag not in raw_tags:
                    continue
            if topic:
                raw_topics = [t.strip() for t in str(m.get("topics", "")).split(",") if t.strip()]
                if topic not in raw_topics:
                    continue
            chunks.append({"content": text, "metadata": m})
            if limit and len(chunks) >= limit:
                break
        return chunks

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def rebuild_bm25(self) -> None:
        log.info("rebuild_bm25: fetching all documents from Chroma …")
        result = self._db.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(
                result.get("documents") or [], result.get("metadatas") or []
            )
            if text
        ]
        self.bm25_index.build(docs)
        self._bm25_dirty = False
        log.info("rebuild_bm25 done: %d documents", len(docs))

    def invalidate_bm25(self) -> None:
        self._bm25_dirty = True
        log.debug("BM25 index invalidated")

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def hybrid_search_with_scores(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        search_filter: SearchFilter | None = None,
    ) -> tuple[
        list[tuple[Document, float]],
        list[tuple[Document, float]],
        list[tuple[Document, float]],
    ]:
        """Run vector + BM25 and merge via RRF.

        Returns:
            (vector_results, bm25_results, fused_results) — each sorted best-first.
        """
        if self._bm25_dirty:
            self.rebuild_bm25()

        where  = self._where(search_filter=search_filter)
        vector = self.similarity_search_with_scores(query, k=fetch_k, search_filter=search_filter)
        bm25   = self.bm25_index.search(query, k=fetch_k, where=where)
        fused  = rrf_fuse(vector, bm25, top_k=fetch_k)

        log.debug("hybrid_search: vector=%d  bm25=%d  fused=%d",
                  len(vector), len(bm25), len(fused))
        return vector, bm25, fused

    # ------------------------------------------------------------------
    # Debug / trace search
    # ------------------------------------------------------------------

    def search_for_debug(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        use_rerank: bool = False,
        use_hybrid: bool = False,
        search_filter: SearchFilter | None = None,
    ) -> dict:
        """Returns retrieval results for the debug dashboard."""
        log.info(
            "search_for_debug: query=%r  k=%d  fetch_k=%d"
            "  use_rerank=%s  use_hybrid=%s  filter=%s",
            query, k, fetch_k, use_rerank, use_hybrid,
            search_filter.summary() if search_filter else doc_id,
        )
        bm25_results:   list[tuple[Document, float]] | None = None
        hybrid_results: list[tuple[Document, float]] | None = None

        if use_hybrid:
            vector_results, bm25_results, hybrid_results = self.hybrid_search_with_scores(
                query, k=k, fetch_k=fetch_k, search_filter=search_filter
            )
            rerank_pool = [doc for doc, _ in hybrid_results]
        else:
            vector_results = self.similarity_search_with_scores(
                query, k=fetch_k, doc_id=doc_id, search_filter=search_filter,
            )
            rerank_pool = [doc for doc, _ in vector_results]

        log.info("  vector=%d  bm25=%s  hybrid=%s",
                 len(vector_results),
                 len(bm25_results) if bm25_results is not None else "off",
                 len(hybrid_results) if hybrid_results is not None else "off")

        reranked: list[tuple[Document, float]] | None = None
        if use_rerank:
            if self._reranker is not None:
                log.info("  reranking %d candidates → top %d", len(rerank_pool), k)
                reranked = self._reranker.rerank_with_scores(query, rerank_pool, top_k=k)
                log.info("  reranked results: %d", len(reranked))
            else:
                log.warning("  use_rerank=True but no reranker is configured — skipping")

        return {
            "vector":   vector_results,
            "bm25":     bm25_results,
            "hybrid":   hybrid_results,
            "reranked": reranked,
        }

    def search_for_trace(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        use_rerank: bool = False,
        use_hybrid: bool = False,
        search_filter: SearchFilter | None = None,
    ) -> dict:
        """Like search_for_debug but adds per-step timing in ``trace``."""
        trace: list[dict] = []

        if use_hybrid:
            if self._bm25_dirty:
                self.rebuild_bm25()
            where = self._where(search_filter=search_filter)

            t0 = perf_counter()
            vector_results = self.similarity_search_with_scores(
                query, k=fetch_k, search_filter=search_filter
            )
            trace.append({
                "stage":      "Vector Search",
                "elapsed_ms": round((perf_counter() - t0) * 1000, 1),
                "in_count":   0,
                "out_count":  len(vector_results),
                "docs":       vector_results[:5],
                "params":     {"fetch_k": fetch_k},
            })

            t0 = perf_counter()
            bm25_results = self.bm25_index.search(query, k=fetch_k, where=where)
            trace.append({
                "stage":      "BM25 Search",
                "elapsed_ms": round((perf_counter() - t0) * 1000, 1),
                "in_count":   0,
                "out_count":  len(bm25_results),
                "docs":       bm25_results[:5],
                "params":     {"fetch_k": fetch_k},
            })

            vector_ids = {d.metadata.get("chunk_id") for d, _ in vector_results}
            bm25_ids   = {d.metadata.get("chunk_id") for d, _ in bm25_results}

            t0 = perf_counter()
            hybrid_results = rrf_fuse(vector_results, bm25_results, top_k=fetch_k)
            trace.append({
                "stage":      "RRF Merge",
                "elapsed_ms": round((perf_counter() - t0) * 1000, 1),
                "in_count":   len(vector_results) + len(bm25_results),
                "out_count":  len(hybrid_results),
                "docs":       hybrid_results[:5],
                "params":     {
                    "overlap":      len(vector_ids & bm25_ids),
                    "total_unique": len(vector_ids | bm25_ids),
                },
            })
            rerank_pool = [doc for doc, _ in hybrid_results]
        else:
            t0 = perf_counter()
            vector_results = self.similarity_search_with_scores(
                query, k=fetch_k, doc_id=doc_id, search_filter=search_filter,
            )
            trace.append({
                "stage":      "Vector Search",
                "elapsed_ms": round((perf_counter() - t0) * 1000, 1),
                "in_count":   0,
                "out_count":  len(vector_results),
                "docs":       vector_results[:5],
                "params":     {"fetch_k": fetch_k},
            })
            bm25_results   = None
            hybrid_results = None
            rerank_pool    = [doc for doc, _ in vector_results]

        reranked: list[tuple[Document, float]] | None = None
        if use_rerank and self._reranker is not None:
            t0 = perf_counter()
            reranked = self._reranker.rerank_with_scores(query, rerank_pool, top_k=k)
            trace.append({
                "stage":      "Rerank",
                "elapsed_ms": round((perf_counter() - t0) * 1000, 1),
                "in_count":   len(rerank_pool),
                "out_count":  len(reranked),
                "docs":       reranked[:5],
                "params":     {"top_k": k},
            })

        final      = reranked or hybrid_results or vector_results
        final_docs = final[:k]
        trace.append({
            "stage":      "Final Context",
            "elapsed_ms": 0.0,
            "in_count":   len(final),
            "out_count":  len(final_docs),
            "docs":       final_docs,
            "params":     {"top_k": k},
        })

        log.debug("search_for_trace: %d steps  total=%.1fms",
                  len(trace), sum(s["elapsed_ms"] for s in trace))
        return {
            "vector":   vector_results,
            "bm25":     bm25_results,
            "hybrid":   hybrid_results,
            "reranked": reranked,
            "trace":    trace,
        }
