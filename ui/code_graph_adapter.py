"""Adapter: retrieval results -> code graph payload.

Composition-first utility that transforms RetrievalResult objects into
Cytoscape.js-ready graph payloads according to the Phase 0 contract.
"""

from __future__ import annotations

from rag.retrieval.base import RetrievalResult
from ui.code_graph_contract import (
    CodeGraphEdge,
    CodeGraphNode,
    CodeGraphPayload,
    GraphLimits,
    apply_limits,
    validate_payload,
)


class CodeGraphAdapter:
    """Build graph payload from relation-enriched retrieval results."""

    def __init__(self, *, limits: GraphLimits | None = None) -> None:
        self._limits = limits or GraphLimits()

    def build(
        self,
        results: list[RetrievalResult],
        *,
        query: str,
    ) -> CodeGraphPayload:
        nodes_by_id: dict[str, CodeGraphNode] = {}
        edges_by_id: dict[str, CodeGraphEdge] = {}
        primary_with_related = 0
        related_ref_count = 0

        for r in results:
            source_id = str(r.metadata.get("chunk_id", "")).strip()
            if not source_id:
                # Skip malformed records but keep contract stable.
                continue

            related_blocks = list(r.metadata.get("related_blocks", []) or [])
            related_ref_count += len(related_blocks)
            if related_blocks:
                primary_with_related += 1

            source_node = self._make_node(
                node_id=source_id,
                label=str(r.metadata.get("name", "") or source_id.split("::")[-1]),
                file_path=str(r.metadata.get("file_path", "") or ""),
                chunk_type=str(r.metadata.get("chunk_type", "") or ""),
                score=float(r.score),
                is_primary=True,
                metadata={
                    "repo_id": r.metadata.get("repo_id", ""),
                    "start_line": r.metadata.get("start_line", 0),
                    "end_line": r.metadata.get("end_line", 0),
                },
            )
            nodes_by_id[source_id] = source_node

            for rel in related_blocks:
                target_id = str(rel.get("target_id", "")).strip()
                if not target_id:
                    continue

                if target_id not in nodes_by_id:
                    target_node = self._make_node(
                        node_id=target_id,
                        label=str(rel.get("target_name", "") or target_id.split("::")[-1]),
                        file_path=str(rel.get("target_file_path", "") or ""),
                        chunk_type=str(rel.get("target_chunk_type", "") or ""),
                        score=float(rel.get("score", 0.0) or 0.0),
                        is_primary=False,
                        metadata={
                            "mapping_strategy": rel.get("mapping_strategy", ""),
                        },
                    )
                    nodes_by_id[target_id] = target_node

                edge_type = str(rel.get("edge_type", "") or "RELATED")
                direction = str(rel.get("direction", "") or "undirected")
                edge_id = self._edge_id(source_id, target_id, edge_type, direction)
                if edge_id in edges_by_id:
                    continue

                edges_by_id[edge_id] = CodeGraphEdge(
                    id=edge_id,
                    source=source_id,
                    target=target_id,
                    edge_type=edge_type,
                    direction=direction,
                    score=float(rel.get("score", 0.0) or 0.0),
                    explain=str(rel.get("explain", "") or rel.get("reason", "")),
                    metadata={
                        "evidence_count": rel.get("evidence_count", 1),
                        "edge_types": rel.get("edge_types", []),
                    },
                )

        payload = CodeGraphPayload(
            nodes=list(nodes_by_id.values()),
            edges=list(edges_by_id.values()),
            meta={
                "query": query,
                "primary_count": len(results),
                "related_primary_count": primary_with_related,
                "related_count_hit_rate": (
                    round(primary_with_related / len(results), 4) if results else 0.0
                ),
                "related_ref_count_before_limit": related_ref_count,
                "adapter": "CodeGraphAdapter",
                "schema_version": 1,
            },
        )

        payload = apply_limits(payload, self._limits)
        validate_payload(payload)
        return payload

    @staticmethod
    def _make_node(
        *,
        node_id: str,
        label: str,
        file_path: str,
        chunk_type: str,
        score: float,
        is_primary: bool,
        metadata: dict,
    ) -> CodeGraphNode:
        return CodeGraphNode(
            id=node_id,
            label=(label or node_id),
            file_path=file_path,
            chunk_type=chunk_type,
            score=score,
            is_primary=is_primary,
            metadata=dict(metadata),
        )

    @staticmethod
    def _edge_id(source: str, target: str, edge_type: str, direction: str) -> str:
        return f"{source}|{target}|{edge_type}|{direction}"
