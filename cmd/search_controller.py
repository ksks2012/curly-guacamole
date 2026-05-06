"""
Search controller — logic layer for the RAG debug dashboard.

Responsibilities:
  - Hold all mutable session state (vector results, reranked results, selected chunk)
  - Execute searches via the RAG backend (rag.client)
  - Expose pure helper utilities (score_color, rank_change)

Has NO dependency on NiceGUI. The display layer (dashboard.py) calls this class
and decides how to surface state changes in the UI.
"""

from langchain_core.documents import Document

from utils.logger import AppLogger
from rag.client import LocalLlamaClient

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
        self._metadata: dict = {}

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
    def selected_metadata(self) -> dict:
        return self._metadata

    @property
    def reranked_chunk_ids(self) -> set:
        """Set of chunk_ids present in the reranked result list."""
        return {doc.metadata.get("chunk_id") for doc, _ in (self._reranked or [])}

    # ------------------------------------------------------------------
    # Actions (state mutations)
    # ------------------------------------------------------------------

    def run_search(
        self,
        query: str,
        k: int,
        fetch_k: int,
        use_rerank: bool,
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
            "run_search: query=%r  k=%d  fetch_k=%d  use_rerank=%s",
            query, k, fetch_k, use_rerank,
        )
        self._vector = []
        self._reranked = None
        self._metadata = {}

        try:
            result = self._client.search_for_debug(
                query, k=k, fetch_k=fetch_k, use_rerank=use_rerank
            )
        except Exception as e:
            log.error("Search failed: %s", e, exc_info=True)
            return str(e)

        self._vector = result["vector"]
        self._reranked = result["reranked"]
        log.info(
            "run_search done: %d vector  reranked=%s",
            len(self._vector),
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
