"""Code graph payload contract for Cytoscape.js + NiceGUI.

Phase GCR2 visualization contract:
- Stable payload schema: nodes + edges + meta
- Required fields for nodes and edges
- Safety limits for rendered graph size
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphLimits:
    """Safety limits for graph rendering."""

    max_nodes: int = 100
    max_edges: int = 200


@dataclass
class CodeGraphNode:
    """Node contract consumed by Cytoscape.js."""

    id: str
    label: str
    file_path: str
    chunk_type: str
    score: float
    is_primary: bool
    metadata: dict = field(default_factory=dict)

    def to_cytoscape(self) -> dict:
        return {
            "data": {
                "id": self.id,
                "label": self.label,
                "file_path": self.file_path,
                "chunk_type": self.chunk_type,
                "score": float(self.score),
                "is_primary": bool(self.is_primary),
                "metadata": dict(self.metadata),
            }
        }


@dataclass
class CodeGraphEdge:
    """Edge contract consumed by Cytoscape.js."""

    id: str
    source: str
    target: str
    edge_type: str
    direction: str
    score: float
    explain: str
    metadata: dict = field(default_factory=dict)

    def to_cytoscape(self) -> dict:
        return {
            "data": {
                "id": self.id,
                "source": self.source,
                "target": self.target,
                "edge_type": self.edge_type,
                "direction": self.direction,
                "score": float(self.score),
                "explain": self.explain,
                "metadata": dict(self.metadata),
            }
        }


@dataclass
class CodeGraphPayload:
    """Top-level payload sent from Python to Cytoscape.js view layer."""

    nodes: list[CodeGraphNode]
    edges: list[CodeGraphEdge]
    meta: dict = field(default_factory=dict)

    def to_cytoscape(self) -> dict:
        """Return a serializable payload for the front-end graph component."""
        return {
            "nodes": [n.to_cytoscape() for n in self.nodes],
            "edges": [e.to_cytoscape() for e in self.edges],
            "meta": dict(self.meta),
        }


def validate_payload(payload: CodeGraphPayload) -> None:
    """Validate required schema fields and uniqueness constraints.

    Raises
    ------
    ValueError when payload violates contract constraints.
    """
    node_ids = set()
    for n in payload.nodes:
        if not n.id:
            raise ValueError("Graph node id is required")
        if n.id in node_ids:
            raise ValueError(f"Duplicate graph node id: {n.id}")
        node_ids.add(n.id)
        if not n.label:
            raise ValueError(f"Graph node label is required: {n.id}")

    edge_ids = set()
    for e in payload.edges:
        if not e.id:
            raise ValueError("Graph edge id is required")
        if e.id in edge_ids:
            raise ValueError(f"Duplicate graph edge id: {e.id}")
        edge_ids.add(e.id)
        if e.source not in node_ids:
            raise ValueError(f"Graph edge source not found in nodes: {e.source}")
        if e.target not in node_ids:
            raise ValueError(f"Graph edge target not found in nodes: {e.target}")


def apply_limits(payload: CodeGraphPayload, limits: GraphLimits) -> CodeGraphPayload:
    """Apply node/edge limits and annotate truncation metadata.

    Keeps insertion order for deterministic rendering.
    """
    nodes = payload.nodes[: limits.max_nodes]
    allowed_ids = {n.id for n in nodes}

    edges_filtered = [
        e for e in payload.edges
        if e.source in allowed_ids and e.target in allowed_ids
    ]
    edges = edges_filtered[: limits.max_edges]

    meta = dict(payload.meta)
    meta["truncated_nodes"] = len(payload.nodes) > len(nodes)
    meta["truncated_edges"] = len(edges_filtered) > len(edges)
    meta["node_count_before_limit"] = len(payload.nodes)
    meta["edge_count_before_limit"] = len(payload.edges)
    meta["node_count"] = len(nodes)
    meta["edge_count"] = len(edges)
    meta["max_nodes"] = limits.max_nodes
    meta["max_edges"] = limits.max_edges

    return CodeGraphPayload(nodes=nodes, edges=edges, meta=meta)
