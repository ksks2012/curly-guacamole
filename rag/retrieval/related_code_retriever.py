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
_EDGE_WEIGHTS = {
    "CALLS": 0.95,
    "EXTENDS": 0.90,
    "IMPLEMENTS": 0.88,
    "IMPORTS": 0.75,
    "NEARBY": 0.55,
}
_DIRECTION_BONUS = {
    "outgoing": 0.03,
    "incoming": 0.02,
    "undirected": 0.01,
}


def _parse_symbol_like_id(identifier: str) -> tuple[str, str, str, str] | None:
    """Parse IDs shaped as ``repo_id::file_path::chunk_type::name``.

    Returns
    -------
    Tuple ``(repo_id, file_path, chunk_type, name)`` or None when malformed.
    """
    parts = identifier.split("::", 3)
    if len(parts) != 4:
        return None
    repo_id, file_path, chunk_type, name = parts
    if not repo_id or not file_path or not chunk_type or not name:
        return None
    return repo_id, file_path, chunk_type, name


def _extract_metadatas(raw: object) -> list[dict]:
    if not isinstance(raw, dict):
        return []
    metas = raw.get("metadatas", [])
    if not isinstance(metas, list):
        return []
    out: list[dict] = []
    for item in metas:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


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
        self._repo_blocks_cache: dict[str, list[dict]] = {}

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
        source_name = str(meta.get("name", "")).strip()
        source_chunk_type = str(meta.get("chunk_type", "")).strip()

        candidates: list[dict] = []

        if source_id and repo_id and self._max_related > 0 and self._edge_types:
            candidates.extend(
                self._collect_graph_relations(
                    source_id=source_id,
                    repo_id=repo_id,
                    file_path=file_path,
                    source_name=source_name,
                    source_chunk_type=source_chunk_type,
                )
            )

        if source_id and repo_id and file_path and start_line > 0 and self._max_nearby > 0:
            candidates.extend(
                self._collect_nearby_relations(
                    source_id=source_id,
                    repo_id=repo_id,
                    file_path=file_path,
                    start_line=start_line,
                )
            )

        related = self._rank_and_aggregate(
            candidates,
            source_file_path=file_path,
            source_start_line=start_line,
        )

        total_limit = self._max_related + self._max_nearby
        if total_limit > 0:
            related = related[:total_limit]

        meta["related_blocks"] = related
        meta["related_count"] = len(related)

        return RetrievalResult(
            content=result.content,
            score=result.score,
            source=result.source,
            metadata=meta,
        )

    def _collect_graph_relations(
        self,
        *,
        source_id: str,
        repo_id: str,
        file_path: str,
        source_name: str,
        source_chunk_type: str,
    ) -> list[dict]:
        rows: list[dict] = []

        anchor_ids = self._source_anchor_ids(
            source_id=source_id,
            repo_id=repo_id,
            file_path=file_path,
            source_name=source_name,
            source_chunk_type=source_chunk_type,
        )

        seen_edges: set[tuple[str, str, str, int, str]] = set()

        candidates: list[tuple[str, object]] = []
        for anchor_id in anchor_ids:
            outgoing = self._get_edges_safe(src_id=anchor_id, repo_id=repo_id)
            incoming = self._get_edges_safe(dst_id=anchor_id, repo_id=repo_id)
            for edge in outgoing:
                edge_key = (
                    str(getattr(edge, "src_id", "")),
                    str(getattr(edge, "dst_id", "")),
                    str(getattr(edge, "edge_type", "")),
                    int(getattr(edge, "line_no", 0) or 0),
                    "outgoing",
                )
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                candidates.append(("outgoing", edge))
            for edge in incoming:
                edge_key = (
                    str(getattr(edge, "src_id", "")),
                    str(getattr(edge, "dst_id", "")),
                    str(getattr(edge, "edge_type", "")),
                    int(getattr(edge, "line_no", 0) or 0),
                    "incoming",
                )
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                candidates.append(("incoming", edge))

        for direction, edge in candidates:
            if getattr(edge, "edge_type", "") not in self._edge_types:
                continue

            target_id = edge.dst_id if direction == "outgoing" else edge.src_id
            if not target_id or target_id in anchor_ids:
                continue

            line_no = int(getattr(edge, "line_no", 0) or 0)
            target_meta = self._fetch_block_metadata(
                repo_id=repo_id,
                target_id=target_id,
                preferred_line=line_no,
            )
            if target_meta is None:
                continue

            resolved_target_id = str(target_meta.get("chunk_id", "")).strip() or str(target_id)

            mapping_strategy = target_meta.pop("_mapping_strategy", "exact_chunk_id")

            rows.append(
                {
                    "target_id": resolved_target_id,
                    "edge_type": edge.edge_type,
                    "direction": direction,
                    "reason": f"{edge.edge_type} relation from dependency graph",
                    "explain": (
                        f"{direction} {edge.edge_type.lower()} dependency"
                    ),
                    "line_no": line_no,
                    "hop": 1,
                    "mapping_strategy": mapping_strategy,
                    "source_anchor": self._edge_source_anchor(
                        edge=edge,
                        direction=direction,
                        anchor_ids=anchor_ids,
                    ),
                    "target_file_path": target_meta.get("file_path", ""),
                    "target_name": target_meta.get("name", ""),
                    "target_chunk_type": target_meta.get("chunk_type", ""),
                }
            )
        return rows[: self._max_related] if self._max_related > 0 else []

    @staticmethod
    def _source_anchor_ids(
        *,
        source_id: str,
        repo_id: str,
        file_path: str,
        source_name: str,
        source_chunk_type: str,
    ) -> tuple[str, ...]:
        anchors: list[str] = []

        def _push(anchor_id: str) -> None:
            aid = str(anchor_id or "").strip()
            if aid and aid not in anchors:
                anchors.append(aid)

        _push(source_id)

        normalized_type = str(source_chunk_type or "").strip()
        if repo_id and file_path and normalized_type != "module":
            _push(f"{repo_id}::{file_path}::module::<module>")

        if repo_id and file_path and normalized_type == "method":
            class_name = str(source_name or "").strip().rsplit(".", 1)[0]
            if class_name:
                _push(f"{repo_id}::{file_path}::class::{class_name}")

        return tuple(anchors)

    def _get_edges_safe(self, **kwargs) -> list[object]:
        try:
            return list(self._graph.get_edges(**kwargs) or [])
        except Exception:
            return []

    @staticmethod
    def _edge_source_anchor(*, edge: object, direction: str, anchor_ids: tuple[str, ...]) -> str:
        if direction == "outgoing":
            source = str(getattr(edge, "src_id", "")).strip()
        else:
            source = str(getattr(edge, "dst_id", "")).strip()
        return source if source in anchor_ids else ""

    def _collect_nearby_relations(
        self,
        *,
        source_id: str,
        repo_id: str,
        file_path: str,
        start_line: int,
    ) -> list[dict]:
        rows: list[dict] = []
        file_blocks = self._fetch_file_blocks(repo_id=repo_id, file_path=file_path)

        scored: list[tuple[int, dict]] = []
        for b in file_blocks:
            target_id = str(b.get("chunk_id", "")).strip()
            if not target_id or target_id == source_id:
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
            rows.append(
                {
                    "target_id": target_id,
                    "edge_type": "NEARBY",
                    "direction": "undirected",
                    "reason": "Nearby block in the same file",
                    "explain": "Nearby block in the same file",
                    "distance": int(dist),
                    "hop": 1,
                    "target_file_path": b.get("file_path", ""),
                    "target_name": b.get("name", ""),
                    "target_chunk_type": b.get("chunk_type", ""),
                }
            )
        return rows

    def _rank_and_aggregate(
        self,
        relations: list[dict],
        *,
        source_file_path: str,
        source_start_line: int,
    ) -> list[dict]:
        by_target: dict[str, dict] = {}
        evidence_map: dict[str, list[dict]] = {}

        for rel in relations:
            target_id = str(rel.get("target_id", "")).strip()
            if not target_id:
                continue

            if target_id not in by_target:
                by_target[target_id] = dict(rel)
                evidence_map[target_id] = [
                    {
                        "edge_type": rel.get("edge_type", ""),
                        "direction": rel.get("direction", ""),
                        "line_no": int(rel.get("line_no", 0) or 0),
                        "distance": int(rel.get("distance", 0) or 0),
                    }
                ]
                continue

            # Keep one row per target_id, preserve first payload fields.
            evidence_map[target_id].append(
                {
                    "edge_type": rel.get("edge_type", ""),
                    "direction": rel.get("direction", ""),
                    "line_no": int(rel.get("line_no", 0) or 0),
                    "distance": int(rel.get("distance", 0) or 0),
                }
            )

        ranked: list[dict] = []

        for target_id, row in by_target.items():
            evidence = evidence_map[target_id]
            edge_types = [str(e.get("edge_type", "")) for e in evidence]
            directions = [str(e.get("direction", "")) for e in evidence]

            dominant_edge = self._pick_dominant_edge(edge_types)
            dominant_direction = self._pick_dominant_direction(directions)

            score = self._relation_score(
                dominant_edge=dominant_edge,
                dominant_direction=dominant_direction,
                evidence_count=len(evidence),
                source_file_path=source_file_path,
                target_file_path=str(row.get("target_file_path", "")),
                source_start_line=source_start_line,
                line_no_values=[int(e.get("line_no", 0) or 0) for e in evidence],
                distance_values=[int(e.get("distance", 0) or 0) for e in evidence],
                hop=int(row.get("hop", 1) or 1),
            )

            explain = self._build_explain(
                edge_type=dominant_edge,
                direction=dominant_direction,
                evidence_count=len(evidence),
                source_file_path=source_file_path,
                target_file_path=str(row.get("target_file_path", "")),
                distance=min([d for d in [int(e.get("distance", 0) or 0) for e in evidence] if d > 0], default=0),
            )

            merged = dict(row)
            merged["edge_type"] = dominant_edge
            merged["direction"] = dominant_direction
            merged["score"] = round(score, 4)
            merged["evidence_count"] = len(evidence)
            merged["edge_types"] = sorted({e for e in edge_types if e})
            merged["mapping_strategy"] = str(row.get("mapping_strategy", "exact_chunk_id"))
            merged["explain"] = explain
            # Keep backward-compatible "reason" while moving UI-facing copy to explain.
            merged["reason"] = explain
            ranked.append(merged)

        ranked.sort(
            key=lambda x: (
                -float(x.get("score", 0.0)),
                str(x.get("target_file_path", "")),
                str(x.get("target_id", "")),
            )
        )
        return ranked

    @staticmethod
    def _pick_dominant_edge(edge_types: list[str]) -> str:
        if not edge_types:
            return "NEARBY"
        return max(edge_types, key=lambda t: _EDGE_WEIGHTS.get(t, 0.0))

    @staticmethod
    def _pick_dominant_direction(directions: list[str]) -> str:
        if not directions:
            return "undirected"
        return max(directions, key=lambda d: _DIRECTION_BONUS.get(d, 0.0))

    def _relation_score(
        self,
        *,
        dominant_edge: str,
        dominant_direction: str,
        evidence_count: int,
        source_file_path: str,
        target_file_path: str,
        source_start_line: int,
        line_no_values: list[int],
        distance_values: list[int],
        hop: int,
    ) -> float:
        base = _EDGE_WEIGHTS.get(dominant_edge, 0.40)
        direction_bonus = _DIRECTION_BONUS.get(dominant_direction, 0.0)
        same_file_bonus = 0.05 if source_file_path and source_file_path == target_file_path else 0.0
        evidence_bonus = min(0.08, 0.02 * max(0, evidence_count - 1))

        hop_penalty = 0.15 * max(0, hop - 1)

        nearby_penalty = 0.0
        valid_dist = [d for d in distance_values if d > 0]
        if valid_dist:
            nearby_penalty = min(0.25, min(valid_dist) / 200 * 0.25)

        line_penalty = 0.0
        valid_line = [l for l in line_no_values if l > 0]
        if source_start_line > 0 and valid_line:
            line_penalty = min(0.08, abs(min(valid_line) - source_start_line) / 500 * 0.08)

        return base + direction_bonus + same_file_bonus + evidence_bonus - hop_penalty - nearby_penalty - line_penalty

    @staticmethod
    def _build_explain(
        *,
        edge_type: str,
        direction: str,
        evidence_count: int,
        source_file_path: str,
        target_file_path: str,
        distance: int,
    ) -> str:
        parts = [f"{direction} {edge_type.lower()} relation"]
        if source_file_path and target_file_path and source_file_path == target_file_path:
            parts.append("same file")
        if edge_type == "NEARBY" and distance > 0:
            parts.append(f"line distance={distance}")
        if evidence_count > 1:
            parts.append(f"evidence={evidence_count}")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Internal: block metadata lookup
    # ------------------------------------------------------------------

    def _fetch_block_metadata(
        self,
        *,
        repo_id: str,
        target_id: str,
        preferred_line: int = 0,
    ) -> dict | None:
        if self._block_fetcher is not None:
            meta = self._block_fetcher(repo_id, target_id)
            if meta is not None:
                m = dict(meta)
                m.setdefault("_mapping_strategy", "custom_fetcher")
                return m
            # Exact lookup returned nothing — fall through to symbol-key fallback below.

        db = self._resolve_block_db()
        if db is not None:
            where = {
                "$and": [
                    {"repo_id": {"$eq": repo_id}},
                    {"chunk_id": {"$eq": target_id}},
                ]
            }
            raw = db.get(where=where, include=["metadatas"])
            metas = _extract_metadatas(raw)
            if metas:
                m = dict(metas[0])
                m["_mapping_strategy"] = "exact_chunk_id"
                return m

        import_target = self._parse_import_target(target_id)
        if import_target is not None:
            resolved = self._resolve_import_target_metadata(
                repo_id=repo_id,
                import_target=import_target,
                preferred_line=preferred_line,
            )
            if resolved is not None:
                return resolved
            return None

        parsed = _parse_symbol_like_id(target_id)
        if parsed is None:
            return None

        parsed_repo, file_path, parsed_type, parsed_name = parsed
        if parsed_repo != repo_id:
            return None

        file_blocks = self._fetch_file_blocks(repo_id=repo_id, file_path=file_path)
        if not file_blocks:
            return None

        matched = [
            b for b in file_blocks
            if str(b.get("name", "")).strip() == parsed_name
        ]

        # Fallback for ambiguous or missing names: use chunk_type + file context.
        if not matched:
            matched = [
                b for b in file_blocks
                if str(b.get("chunk_type", "")).strip() == parsed_type
            ]

        if not matched:
            return None

        chosen = self._choose_block_candidate(
            candidates=matched,
            parsed_type=parsed_type,
            preferred_line=preferred_line,
        )
        if chosen is None:
            return None

        chosen = dict(chosen)
        chosen["_mapping_strategy"] = "symbol_key_fallback"
        return chosen

    @staticmethod
    def _parse_import_target(target_id: str) -> str | None:
        if not str(target_id).startswith("import::"):
            return None
        target = str(target_id).split("::", 1)[1].strip()
        return target or None

    @staticmethod
    def _module_path_from_file_path(file_path: str) -> str:
        path = str(file_path or "").strip()
        if not path.endswith(".py"):
            return ""
        mod = path[:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        return mod.strip(".")

    @staticmethod
    def _leaf_name(symbol_name: str) -> str:
        name = str(symbol_name or "").strip()
        if not name:
            return ""
        return name.rsplit(".", 1)[-1]

    def _fetch_repo_blocks(self, *, repo_id: str) -> list[dict]:
        if repo_id in self._repo_blocks_cache:
            return self._repo_blocks_cache[repo_id]

        db = self._resolve_block_db()
        if db is None:
            self._repo_blocks_cache[repo_id] = []
            return []

        raw = db.get(where={"repo_id": {"$eq": repo_id}}, include=["metadatas"])
        rows = _extract_metadatas(raw)
        self._repo_blocks_cache[repo_id] = rows
        return rows

    def _resolve_import_target_metadata(
        self,
        *,
        repo_id: str,
        import_target: str,
        preferred_line: int,
    ) -> dict | None:
        parts = [p for p in str(import_target).split(".") if p]
        if not parts:
            return None

        repo_blocks = self._fetch_repo_blocks(repo_id=repo_id)
        if not repo_blocks:
            return None

        module_to_file: dict[str, str] = {}
        blocks_by_file: dict[str, list[dict]] = {}
        for b in repo_blocks:
            file_path = str(b.get("file_path", "")).strip()
            if not file_path:
                continue
            blocks_by_file.setdefault(file_path, []).append(b)
            mod = self._module_path_from_file_path(file_path)
            if mod and mod not in module_to_file:
                module_to_file[mod] = file_path

        candidates: list[dict] = []

        for i in range(len(parts), 0, -1):
            module_path = ".".join(parts[:i])
            file_path = module_to_file.get(module_path)
            if not file_path:
                continue

            file_blocks = blocks_by_file.get(file_path, [])
            suffix = parts[i:]
            if not suffix:
                candidates = [
                    b for b in file_blocks
                    if str(b.get("chunk_type", "")).strip() == "module"
                ] or list(file_blocks)
                break

            symbol_suffix = ".".join(suffix)
            candidates = [
                b for b in file_blocks
                if str(b.get("name", "")).strip() == symbol_suffix
            ]
            if candidates:
                break

            candidates = [
                b for b in file_blocks
                if str(b.get("name", "")).strip().endswith("." + symbol_suffix)
            ]
            if candidates:
                break

            leaf = suffix[-1]
            candidates = [
                b for b in file_blocks
                if self._leaf_name(str(b.get("name", "")).strip()) == leaf
            ]
            if candidates:
                break

        if not candidates:
            leaf = parts[-1]
            candidates = [
                b for b in repo_blocks
                if self._leaf_name(str(b.get("name", "")).strip()) == leaf
            ]

        if not candidates:
            return None

        chosen = self._choose_import_candidate(candidates=candidates, preferred_line=preferred_line)
        if chosen is None:
            return None

        out = dict(chosen)
        out["_mapping_strategy"] = "import_path_fallback"
        return out

    def _choose_import_candidate(self, *, candidates: list[dict], preferred_line: int) -> dict | None:
        if not candidates:
            return None

        def _span_size(meta: dict) -> int:
            s = int(meta.get("start_line", 0) or 0)
            e = int(meta.get("end_line", 0) or 0)
            if s > 0 and e >= s:
                return e - s + 1
            return 10**9

        def _line_distance(meta: dict) -> int:
            if preferred_line <= 0:
                return 0
            s = int(meta.get("start_line", 0) or 0)
            if s <= 0:
                return 10**9
            return abs(s - preferred_line)

        ranked = sorted(
            candidates,
            key=lambda m: (
                _line_distance(m),
                _span_size(m),
                int(m.get("start_line", 0) or 0),
                str(m.get("chunk_type", "")),
            ),
        )
        return ranked[0]

    def _choose_block_candidate(
        self,
        *,
        candidates: list[dict],
        parsed_type: str,
        preferred_line: int,
    ) -> dict | None:
        if not candidates:
            return None

        def _type_rank(chunk_type: str) -> int:
            if chunk_type == parsed_type:
                return 0
            if {chunk_type, parsed_type} <= {"function", "method"}:
                return 1
            return 2

        def _span_size(meta: dict) -> int:
            s = int(meta.get("start_line", 0) or 0)
            e = int(meta.get("end_line", 0) or 0)
            if s > 0 and e >= s:
                return e - s + 1
            return 10**9

        def _line_distance(meta: dict) -> int:
            if preferred_line <= 0:
                return 0
            s = int(meta.get("start_line", 0) or 0)
            if s <= 0:
                return 10**9
            return abs(s - preferred_line)

        ranked = sorted(
            candidates,
            key=lambda m: (
                _type_rank(str(m.get("chunk_type", "")).strip()),
                _span_size(m),
                _line_distance(m),
                int(m.get("start_line", 0) or 0),
            ),
        )
        return ranked[0]

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
        return _extract_metadatas(raw)

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
