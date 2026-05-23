"""
RelatedCodeRetriever — relation expansion for code-block retrieval.

This retriever composes an existing code retriever and enriches each returned
code block with explainable related block information.

Phase GCR2.5 scope (MVP):
- 1-hop relation expansion from GraphStore dependency edges
- optional same-file nearby block expansion
- bounded related result count per primary block
- response contract via ``metadata['related_blocks']``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from rag.retrieval.base import RetrievalResult

if TYPE_CHECKING:
    from rag.code.graph_store import GraphStore


_ALLOWED_EDGE_TYPES = frozenset({"CALLS", "EXTENDS", "IMPLEMENTS", "IMPORTS"})


class RelatedCodeRetriever:
    """Compose a base retriever and append related block metadata.

    Parameters
    ----------
    base_retriever      : Existing code retriever (e.g. HierarchicalCodeRetriever).
    graph_store         : GraphStore instance used for dependency-edge lookups.
    max_related         : Maximum graph-derived relations per primary result.
    max_nearby          : Maximum same-file nearby relations per primary result.
    edge_types          : Allowed dependency edge types used for relation expansion.
    block_fetcher       : Optional callback ``(repo_id, target_id) -> metadata or None``.
    file_blocks_fetcher : Optional callback ``(repo_id, file_path) -> list[metadata]``.

    Notes
    -----
    This class uses composition and does not inherit from other retrievers.
    """

    def __init__(
        self,
        base_retriever,
        graph_store: "GraphStore",
        *,
        max_related: int = 5,
        max_nearby: int = 2,
        edge_types: tuple[str, ...] = ("CALLS", "EXTENDS", "IMPLEMENTS", "IMPORTS"),
        block_fetcher: Callable[[str, str], dict | None] | None = None,
        file_blocks_fetcher: Callable[[str, str], list[dict]] | None = None,
    ) -> None:
        self._base = base_retriever
        self._graph = graph_store
        self._max_related = max(0, int(max_related))
        self._max_nearby = max(0, int(max_nearby))
        self._edge_types = tuple(t for t in edge_types if t in _ALLOWED_EDGE_TYPES)
        self._block_fetcher = block_fetcher
        self._file_blocks_fetcher = file_blocks_fetcher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        *,
        repo_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Search and enrich primary code-block results with related blocks."""
        primary = self._call_base(query, top_k=top_k, filters=filters, repo_ids=repo_ids)
        return [self._enrich_result(r) for r in primary]

    # ------------------------------------------------------------------
    # Internal: base retrieval delegation
    # ------------------------------------------------------------------

    def _call_base(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict | None,
        repo_ids: list[str] | None,
    ) -> list[RetrievalResult]:
        if repo_ids is None:
            return self._base.search(query, top_k=top_k, filters=filters)
        try:
            return self._base.search(query, top_k=top_k, filters=filters, repo_ids=repo_ids)
        except TypeError:
            return self._base.search(query, top_k=top_k, filters=filters)

    # ------------------------------------------------------------------
    # Internal: enrichment
    # ------------------------------------------------------------------

    def _enrich_result(self, result: RetrievalResult) -> RetrievalResult:
        meta = dict(result.metadata or {})

        source_id = str(meta.get("chunk_id", "")).strip()
        repo_id = str(meta.get("repo_id", "")).strip()
        file_path = str(meta.get("file_path", "")).strip()
        start_line = int(meta.get("start_line", 0) or 0)

        related: list[dict] = []
        seen_targets: set[str] = set()

        if source_id and repo_id and self._max_related > 0 and self._edge_types:
            related.extend(
                self._expand_graph_relations(
                    source_id=source_id,
                    repo_id=repo_id,
                    seen_targets=seen_targets,
                )
            )

        if source_id and repo_id and file_path and start_line > 0 and self._max_nearby > 0:
            related.extend(
                self._expand_nearby_relations(
                    source_id=source_id,
                    repo_id=repo_id,
                    file_path=file_path,
                    start_line=start_line,
                    seen_targets=seen_targets,
                )
            )

        meta["related_blocks"] = related
        meta["related_count"] = len(related)

        return RetrievalResult(
            content=result.content,
            score=result.score,
            source=result.source,
            metadata=meta,
        )

    def _expand_graph_relations(
        self,
        *,
        source_id: str,
        repo_id: str,
        seen_targets: set[str],
    ) -> list[dict]:
        rows: list[dict] = []

        outgoing = self._graph.get_edges(src_id=source_id, repo_id=repo_id)
        incoming = self._graph.get_edges(dst_id=source_id, repo_id=repo_id)

        candidates: list[tuple[str, object]] = []
        candidates.extend(("outgoing", e) for e in outgoing)
        candidates.extend(("incoming", e) for e in incoming)

        for direction, edge in candidates:
            if len(rows) >= self._max_related:
                break
            if getattr(edge, "edge_type", "") not in self._edge_types:
                continue

            target_id = edge.dst_id if direction == "outgoing" else edge.src_id
            if not target_id or target_id == source_id:
                continue
            if str(target_id).startswith("import::"):
                continue
            if target_id in seen_targets:
                continue

            target_meta = self._fetch_block_metadata(repo_id=repo_id, target_id=target_id)
            if target_meta is None:
                continue

            seen_targets.add(target_id)
            rows.append(
                {
                    "target_id": target_id,
                    "edge_type": edge.edge_type,
                    "direction": direction,
                    "reason": f"{edge.edge_type} relation from dependency graph",
                    "line_no": int(getattr(edge, "line_no", 0) or 0),
                    "target_file_path": target_meta.get("file_path", ""),
                    "target_name": target_meta.get("name", ""),
                    "target_chunk_type": target_meta.get("chunk_type", ""),
                }
            )
        return rows

    def _expand_nearby_relations(
        self,
        *,
        source_id: str,
        repo_id: str,
        file_path: str,
        start_line: int,
        seen_targets: set[str],
    ) -> list[dict]:
        rows: list[dict] = []
        file_blocks = self._fetch_file_blocks(repo_id=repo_id, file_path=file_path)

        scored: list[tuple[int, dict]] = []
        for b in file_blocks:
            target_id = str(b.get("chunk_id", "")).strip()
            if not target_id or target_id == source_id:
                continue
            if target_id in seen_targets:
                continue
            t_line = int(b.get("start_line", 0) or 0)
            if t_line <= 0:
                continue
            scored.append((abs(t_line - start_line), b))

        scored.sort(key=lambda x: x[0])

        for dist, b in scored[: self._max_nearby]:
            target_id = str(b.get("chunk_id", "")).strip()
            if not target_id:
                continue
            seen_targets.add(target_id)
            rows.append(
                {
                    "target_id": target_id,
                    "edge_type": "NEARBY",
                    "direction": "undirected",
                    "reason": "Nearby block in the same file",
                    "distance": int(dist),
                    "target_file_path": b.get("file_path", ""),
                    "target_name": b.get("name", ""),
                    "target_chunk_type": b.get("chunk_type", ""),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Internal: block metadata lookup
    # ------------------------------------------------------------------

    def _fetch_block_metadata(self, *, repo_id: str, target_id: str) -> dict | None:
        if self._block_fetcher is not None:
            return self._block_fetcher(repo_id, target_id)

        db = self._resolve_block_db()
        if db is None:
            return None

        where = {
            "$and": [
                {"repo_id": {"$eq": repo_id}},
                {"chunk_id": {"$eq": target_id}},
            ]
        }
        raw = db.get(where=where, include=["metadatas"])
        metas = raw.get("metadatas", []) or []
        return dict(metas[0]) if metas else None

    def _fetch_file_blocks(self, *, repo_id: str, file_path: str) -> list[dict]:
        if self._file_blocks_fetcher is not None:
            return self._file_blocks_fetcher(repo_id, file_path)

        db = self._resolve_block_db()
        if db is None:
            return []

        where = {
            "$and": [
                {"repo_id": {"$eq": repo_id}},
                {"file_path": {"$eq": file_path}},
            ]
        }
        raw = db.get(where=where, include=["metadatas"])
        metas = raw.get("metadatas", []) or []
        return [dict(m) for m in metas]

    def _resolve_block_db(self):
        # Supports HierarchicalCodeRetriever -> CodeRetriever -> CodeIndexer
        code_retriever = getattr(self._base, "_code_retriever", None)
        indexer = getattr(code_retriever, "_indexer", None)

        # Supports direct CodeRetriever
        if indexer is None:
            indexer = getattr(self._base, "_indexer", None)

        if indexer is None:
            return None

        try:
            return indexer._db("block")
        except Exception:
            return None
