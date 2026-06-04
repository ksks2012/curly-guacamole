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

from ui.client_protocols import SearchClientProtocol
from utils.logger import AppLogger
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

    def __init__(self, client: SearchClientProtocol) -> None:
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

    @staticmethod
    def _is_code_doc(doc: Document) -> bool:
        """Return True when a retrieval document belongs to code index data."""
        meta = dict(doc.metadata or {})
        if str(meta.get("source_type", "")).strip().lower() == "code":
            return True
        if str(meta.get("repo_id", "")).strip():
            return True
        if str(meta.get("chunk_type", "")).strip():
            return True
        return False

    def _filter_scope(
        self,
        rows: list[tuple[Document, float]] | None,
        *,
        scope: str,
    ) -> list[tuple[Document, float]] | None:
        """Filter retrieval rows by scope: all | document | code."""
        if rows is None or scope == "all":
            return rows
        if scope == "code":
            return [(d, s) for d, s in rows if self._is_code_doc(d)]
        if scope == "document":
            return [(d, s) for d, s in rows if not self._is_code_doc(d)]
        return rows

    def run_search(
        self,
        query: str,
        k: int,
        fetch_k: int,
        use_rerank: bool,
        use_hybrid: bool = False,
        result_scope: str = "all",
        apply_filter: bool = True,
        include_relations: bool = False,
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
            "  use_hybrid=%s  scope=%s  apply_filter=%s  filter=%s",
            query, k, fetch_k, use_rerank, use_hybrid, result_scope, apply_filter, self._filter.summary(),
        )
        self._vector = []
        self._reranked = None
        self._bm25 = None
        self._hybrid = None
        self._metadata = {}
        self._trace = []
        self._last_query = query

        try:
            if result_scope == "code":
                result = self._client.search_code_blocks(
                    query,
                    k=k,
                    fetch_k=fetch_k,
                    use_rerank=use_rerank,
                    include_relations=include_relations,
                )
            else:
                result = self._client.search_for_trace(
                    query, k=k, fetch_k=fetch_k,
                    use_rerank=use_rerank, use_hybrid=use_hybrid,
                    search_filter=(self._filter if apply_filter and self.filter_active else None),
                )
        except Exception as e:
            log.error("Search failed: %s", e, exc_info=True)
            return str(e)

        self._vector   = self._filter_scope(result["vector"], scope=result_scope) or []
        self._reranked = self._filter_scope(result["reranked"], scope=result_scope)
        self._bm25     = self._filter_scope(result["bm25"], scope=result_scope)
        self._hybrid   = self._filter_scope(result["hybrid"], scope=result_scope)
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

    def select_chunk_by_id(self, chunk_id: str) -> bool:
        """Select a chunk by id from current in-memory search results.

        Search order prefers user-facing lists: reranked -> hybrid -> vector -> bm25.
        Returns True when a matching chunk is found.
        """
        chunk_id = str(chunk_id or "").strip()
        if not chunk_id:
            return False

        pools: list[tuple[str, list[tuple[Document, float]] | None]] = [
            ("rscore", self._reranked),
            ("rrf_score", self._hybrid),
            ("vscore", self._vector),
            ("bm25score", self._bm25),
        ]
        for score_key, docs in pools:
            for doc, score in (docs or []):
                if str(doc.metadata.get("chunk_id", "")).strip() == chunk_id:
                    self.select_chunk(doc, score, score_key)
                    return True
        return False

    def select_graph_node(self, node_data: dict) -> None:
        """Select node metadata when graph node has no in-memory content."""
        self._metadata = {
            "_detail_kind": "node",
            "graph_node_id": str(node_data.get("id", "")),
            "label": str(node_data.get("label", "")),
            "file_path": str(node_data.get("file_path", "")),
            "chunk_type": str(node_data.get("chunk_type", "")),
            "score": float(node_data.get("score", 0.0) or 0.0),
            "is_primary": bool(node_data.get("is_primary", False)),
            "_content_len": 0,
            "_content": "Content unavailable in current result set.",
            **dict(node_data.get("metadata", {}) or {}),
        }

    def select_graph_edge(self, edge_data: dict) -> None:
        """Select edge metadata for graph relation inspection."""
        meta = dict(edge_data.get("metadata", {}) or {})
        explain = str(edge_data.get("explain", "") or "")
        edge_types = meta.get("edge_types", [])
        if not isinstance(edge_types, list):
            edge_types = []

        self._metadata = {
            "_detail_kind": "edge",
            "graph_edge_id": str(edge_data.get("id", "")),
            "source_id": str(edge_data.get("source", "")),
            "target_id": str(edge_data.get("target", "")),
            "edge_type": str(edge_data.get("edge_type", "")),
            "direction": str(edge_data.get("direction", "")),
            "score": float(edge_data.get("score", 0.0) or 0.0),
            "explain": explain,
            "evidence_count": int(meta.get("evidence_count", 0) or 0),
            "edge_types": edge_types,
            "_content_len": len(explain),
            "_content": explain or "No explanation available for this edge.",
        }

    def clear_selection(self) -> None:
        """Clear selected detail metadata."""
        self._metadata = {}

    def handle_graph_node_click(self, node_id: str, node_data: dict) -> None:
        """Handle node click by preferring in-memory chunk data and falling back to node payload."""
        node_id = str(node_id or "").strip()
        if node_id and self.select_chunk_by_id(node_id):
            return
        self.select_graph_node(node_data)

    def handle_graph_edge_click(self, edge_data: dict) -> None:
        """Handle edge click to show edge relation details."""
        self.select_graph_edge(edge_data)

    def handle_graph_canvas_click(self) -> None:
        """Handle canvas click by resetting detail panel state."""
        self.clear_selection()

    def select_code_block(self, row: dict) -> None:
        """Select a code block row for the shared detail panel."""
        meta = dict((row or {}).get("metadata") or {})
        content = str((row or {}).get("content") or "")
        ordered_meta = {
            "source_type": str(meta.get("source_type") or "code"),
            "repo_id": str(meta.get("repo_id", "")),
            "file_path": str(meta.get("file_path", "")),
            "chunk_type": str(meta.get("chunk_type", "")),
            "name": str(meta.get("name", "")),
            "language": str(meta.get("language", "")),
            "branch": str(meta.get("branch", "")),
            "start_line": meta.get("start_line", ""),
            "end_line": meta.get("end_line", ""),
            "chunk_id": str(meta.get("chunk_id", "")),
        }
        for key in sorted(meta.keys()):
            if key not in ordered_meta:
                ordered_meta[key] = meta[key]
        ordered_meta["_content_len"] = len(content)
        ordered_meta["_content"] = content[:4000]
        self._metadata = ordered_meta

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

    def list_code_blocks(
        self,
        *,
        repo_id: str = "",
        file_path: str = "",
        text: str = "",
        limit: int = 500,
    ) -> list[dict]:
        """Return code block rows with optional metadata/text filtering."""
        try:
            rows = self._client.browse_code_blocks(
                repo_id=(repo_id or "").strip() or None,
                file_path=(file_path or "").strip() or None,
                limit=limit,
            )
        except Exception as e:
            log.error("list_code_blocks failed: %s", e)
            return []

        needle = (text or "").strip().lower()
        if not needle:
            return rows

        out: list[dict] = []
        for row in rows:
            meta = row.get("metadata") or {}
            content = str(row.get("content") or "")
            if (
                needle in content.lower()
                or needle in str(meta.get("name", "")).lower()
                or needle in str(meta.get("file_path", "")).lower()
                or needle in str(meta.get("chunk_type", "")).lower()
                or needle in str(meta.get("repo_id", "")).lower()
            ):
                out.append(row)
        return out

    def list_code_repo_ids(self) -> list[str]:
        """Return all known code repo ids for UI selection controls."""
        try:
            return self._client.list_code_repo_ids()
        except Exception as e:
            log.error("list_code_repo_ids failed: %s", e)
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
