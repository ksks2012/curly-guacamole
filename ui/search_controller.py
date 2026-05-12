"""
Search controller — logic layer for the RAG debug dashboard.

Responsibilities:
  - Hold all mutable session state (vector results, reranked results, selected chunk)
  - Execute searches via the RAG backend (rag.client)
  - Expose pure helper utilities (score_color, rank_change)

Has NO dependency on NiceGUI. The display layer (dashboard.py) calls this class
and decides how to surface state changes in the UI.
"""

from dataclasses import dataclass, field

from langchain_core.documents import Document

from utils.logger import AppLogger
from rag.client import LocalLlamaClient
from rag.retrieval.filters import SearchFilter


@dataclass
class TraceStep:
    """One stage of the retrieval pipeline, populated by search_for_trace()."""

    stage:      str
    elapsed_ms: float
    in_count:   int
    out_count:  int
    docs:       list[tuple[Document, float]] = field(default_factory=list)
    params:     dict = field(default_factory=dict)

log = AppLogger.get(__name__)

# Sentinel returned by run_search when rerank was requested but unavailable.
RERANKER_UNAVAILABLE = "reranker_unavailable"


class SearchController:
    """
    Bridges the RAG backend and the display layer.

    All mutable state lives here. The display layer reads state through
    properties and triggers mutations through action methods.
    """

    def __init__(self, client: LocalLlamaClient) -> None:
        self._client = client
        self._vector: list[tuple[Document, float]] = []
        self._reranked: list[tuple[Document, float]] | None = None
        self._bm25: list[tuple[Document, float]] | None = None
        self._hybrid: list[tuple[Document, float]] | None = None
        self._metadata: dict = {}
        self._filter: SearchFilter = SearchFilter()
        self._trace: list[TraceStep] = []
        self._last_query: str = ""

    # ------------------------------------------------------------------
    # Read-only state accessors
    # ------------------------------------------------------------------

    @property
    def vector_results(self) -> list[tuple[Document, float]]:
        return self._vector

    @property
    def reranked_results(self) -> list[tuple[Document, float]] | None:
        return self._reranked

    @property
    def bm25_results(self) -> list[tuple[Document, float]] | None:
        """Raw BM25 results from the last hybrid search, or None."""
        return self._bm25

    @property
    def hybrid_results(self) -> list[tuple[Document, float]] | None:
        """RRF-fused results from the last hybrid search, or None."""
        return self._hybrid

    @property
    def hybrid_chunk_ids(self) -> set:
        """Set of chunk_ids present in the fused hybrid result list."""
        return {doc.metadata.get("chunk_id") for doc, _ in (self._hybrid or [])}

    @property
    def selected_metadata(self) -> dict:
        return self._metadata

    @property
    def reranked_chunk_ids(self) -> set:
        """Set of chunk_ids present in the reranked result list."""
        return {doc.metadata.get("chunk_id") for doc, _ in (self._reranked or [])}

    @property
    def filter(self) -> SearchFilter:
        """Active SearchFilter (all-None = no filtering)."""
        return self._filter

    @property
    def filter_doc_id(self) -> str | None:
        """Backward-compatible: active source_id constraint, or None."""
        return self._filter.source_id

    @property
    def filter_active(self) -> bool:
        """True when at least one filter dimension is constrained."""
        return not self._filter.is_empty()

    @property
    def filter_summary(self) -> str:
        """Short human-readable description of active constraints."""
        return self._filter.summary()

    @property
    def trace(self) -> list[TraceStep]:
        """Per-step trace from the last search_for_trace call."""
        return self._trace

    @property
    def last_query(self) -> str:
        """The query string from the most recent search."""
        return self._last_query

    # ------------------------------------------------------------------
    # Actions (state mutations)
    # ------------------------------------------------------------------

    def run_search(
        self,
        query: str,
        k: int,
        fetch_k: int,
        use_rerank: bool,
        use_hybrid: bool = False,
    ) -> str | None:
        """Execute a debug search and update internal state.

        Returns:
            None                  — success
            "Query is empty."     — validation failure (soft, no logging needed)
            RERANKER_UNAVAILABLE  — rerank requested but not configured (warning only)
            str (other)           — error message from an exception (hard failure)
        """
        query = query.strip()
        if not query:
            return "Query is empty."

        log.info(
            "run_search: query=%r  k=%d  fetch_k=%d  use_rerank=%s"
            "  use_hybrid=%s  filter=%s",
            query, k, fetch_k, use_rerank, use_hybrid, self._filter.summary(),
        )
        self._vector = []
        self._reranked = None
        self._bm25 = None
        self._hybrid = None
        self._metadata = {}
        self._trace = []
        self._last_query = query

        try:
            result = self._client.search_for_trace(
                query, k=k, fetch_k=fetch_k,
                use_rerank=use_rerank, use_hybrid=use_hybrid,
                search_filter=self._filter if self.filter_active else None,
            )
        except Exception as e:
            log.error("Search failed: %s", e, exc_info=True)
            return str(e)

        self._vector   = result["vector"]
        self._reranked = result["reranked"]
        self._bm25     = result["bm25"]
        self._hybrid   = result["hybrid"]
        self._trace    = [
            TraceStep(
                stage=s["stage"],
                elapsed_ms=s["elapsed_ms"],
                in_count=s["in_count"],
                out_count=s["out_count"],
                docs=s["docs"],
                params=s["params"],
            )
            for s in result.get("trace", [])
        ]
        log.info(
            "run_search done: %d vector  bm25=%s  hybrid=%s  reranked=%s",
            len(self._vector),
            len(self._bm25) if self._bm25 is not None else "off",
            len(self._hybrid) if self._hybrid is not None else "off",
            len(self._reranked) if self._reranked is not None else "off",
        )

        if use_rerank and self._reranked is None:
            log.warning("use_rerank=True but reranker returned None")
            return RERANKER_UNAVAILABLE

        return None

    def select_chunk(self, doc: Document, score: float, score_key: str) -> None:
        """Store the clicked chunk so the detail panel can render it."""
        self._metadata = {
            score_key: score,
            **doc.metadata,
            "_content_len": len(doc.page_content),
            "_content": doc.page_content[:600],
        }
        log.debug(
            "select_chunk: chunk_id=%s  %s=%s",
            doc.metadata.get("chunk_id"), score_key, score,
        )

    def set_filter(self, f: SearchFilter) -> None:
        """Replace the active filter entirely."""
        self._filter = f
        log.info("filter set: %s", f.summary())

    def set_filter_field(self, field: str, value: str | None) -> None:
        """Set a single filter dimension without touching the others.

        *value* is normalised to None when empty so is_empty() stays accurate.
        """
        normalised = (value or "").strip() or None
        setattr(self._filter, field, normalised)
        log.debug("filter field %s = %r  (full: %s)", field, normalised, self._filter.summary())

    def clear_filter(self) -> None:
        """Remove all filter constraints."""
        self._filter = SearchFilter()
        log.info("filter cleared")

    def list_doc_ids(self) -> list[str]:
        """Return all distinct doc_id values from the Chroma collection."""
        try:
            return self._client.list_doc_ids()
        except Exception as e:
            log.error("list_doc_ids failed: %s", e)
            return []

    def list_doc_title_map(self) -> dict[str, str]:
        """Return a {doc_id: display_title} map for all indexed documents."""
        try:
            return self._client.list_doc_title_map()
        except Exception as e:
            log.error("list_doc_title_map failed: %s", e)
            return {}

    def list_workspaces(self) -> list[str]:
        try:
            return self._client.list_workspaces()
        except Exception as e:
            log.error("list_workspaces failed: %s", e)
            return []

    def list_document_types(self) -> list[str]:
        try:
            return self._client.list_document_types()
        except Exception as e:
            log.error("list_document_types failed: %s", e)
            return []

    def list_tags(self) -> list[str]:
        try:
            return self._client.list_tags()
        except Exception as e:
            log.error("list_tags failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Pure helpers (no state, safe to call as static methods)
    # ------------------------------------------------------------------

    @staticmethod
    def score_color(score: float) -> str:
        """Return a Tailwind text-color class for a 0-1 vector relevance score."""
        if score >= 0.75:
            return "text-green-600"
        if score >= 0.50:
            return "text-yellow-600"
        return "text-red-500"

    @staticmethod
    def rank_change(
        chunk_id, vector_results: list[tuple[Document, float]], rerank_pos: int
    ) -> tuple[str, str]:
        """Return (label, css_class) describing rank change after reranking.

        delta > 0  →  moved up   (▲N, green)
        delta < 0  →  moved down (▼N, red)
        delta == 0 →  no change  (—,  gray)
        not found  →  new entry  (★,  blue) — should not occur in practice
        """
        for v_pos, (doc, _) in enumerate(vector_results):
            if doc.metadata.get("chunk_id") == chunk_id:
                delta = v_pos - rerank_pos
                if delta > 0:
                    return f"▲{delta}", "text-green-600 font-bold"
                if delta < 0:
                    return f"▼{abs(delta)}", "text-red-500 font-bold"
                return "—", "text-gray-400"
        return "★", "text-blue-400"
