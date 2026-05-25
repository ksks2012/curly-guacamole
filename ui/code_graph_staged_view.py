"""Two-stage graph view configuration.

Stage 1 focuses on sparse overview for quick topology understanding.
Stage 2 enables enriched relations for deep inspection.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphViewStage:
    """Configuration profile for one graph view stage."""

    key: str
    label: str
    description: str
    relations_enabled: bool
    edge_multiplier: float
    forced_layout: str | None
    target_overlap_rate: float
    target_readable_nodes: int
    target_latency_p95_ms: int


STAGE_SPARSE = GraphViewStage(
    key="sparse",
    label="Stage 1: Sparse",
    description="Show primary structure first with reduced relation noise.",
    relations_enabled=False,
    edge_multiplier=0.45,
    forced_layout="breadthfirst",
    target_overlap_rate=0.15,
    target_readable_nodes=40,
    target_latency_p95_ms=250,
)

STAGE_DENSE = GraphViewStage(
    key="dense",
    label="Stage 2: Enriched",
    description="Enable relation-rich view for detailed inspection.",
    relations_enabled=True,
    edge_multiplier=1.0,
    forced_layout=None,
    target_overlap_rate=0.25,
    target_readable_nodes=72,
    target_latency_p95_ms=250,
)

STAGES = {
    STAGE_SPARSE.key: STAGE_SPARSE,
    STAGE_DENSE.key: STAGE_DENSE,
}

DEFAULT_STAGE_KEY = STAGE_SPARSE.key


@dataclass(frozen=True)
class ResolvedStageConfig:
    """Resolved control values after applying stage rules."""

    stage_key: str
    max_nodes: int
    max_edges: int
    layout: str
    relations_enabled: bool


def get_stage(stage_key: str) -> GraphViewStage:
    """Return stage definition by key, defaulting to sparse stage."""
    return STAGES.get(str(stage_key or "").strip().lower(), STAGE_SPARSE)


def apply_stage(
    *,
    stage_key: str,
    max_nodes: int,
    dense_max_edges: int,
    preferred_layout: str,
) -> ResolvedStageConfig:
    """Apply stage policy to node/edge/layout/relation controls."""
    stage = get_stage(stage_key)
    bounded_nodes = max(10, int(max_nodes))
    bounded_dense_edges = max(20, int(dense_max_edges))

    if stage.key == STAGE_SPARSE.key:
        sparse_edges = max(20, int(round(bounded_dense_edges * stage.edge_multiplier)))
        return ResolvedStageConfig(
            stage_key=stage.key,
            max_nodes=bounded_nodes,
            max_edges=sparse_edges,
            layout=stage.forced_layout or preferred_layout,
            relations_enabled=stage.relations_enabled,
        )

    return ResolvedStageConfig(
        stage_key=stage.key,
        max_nodes=bounded_nodes,
        max_edges=bounded_dense_edges,
        layout=preferred_layout,
        relations_enabled=stage.relations_enabled,
    )


def next_stage_key(current_stage_key: str) -> str:
    """Return the next stage key in sparse -> dense flow."""
    if get_stage(current_stage_key).key == STAGE_SPARSE.key:
        return STAGE_DENSE.key
    return STAGE_DENSE.key
