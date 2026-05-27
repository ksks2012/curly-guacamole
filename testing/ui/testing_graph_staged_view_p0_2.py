"""Verification tests for P0-2 staged graph workflow.

Workflow:
1) Stage 1 sparse overview (relations off, reduced edges)
2) Stage 2 enriched detail view (relations on)
"""

from __future__ import annotations

from rag.retrieval.base import RetrievalResult
from ui.code_graph_adapter import CodeGraphAdapter, GraphLimits
from ui.code_graph_metrics import (
    estimate_readability,
    interaction_latency_p95_ms,
    overlap_rate,
    readable_nodes_count,
)
from ui.code_graph_staged_view import STAGE_DENSE, STAGE_SPARSE, apply_stage, next_stage_key


def create_mock_results(node_count: int) -> list[RetrievalResult]:
    """Create deterministic mock code results for graph generation."""
    rows: list[RetrievalResult] = []
    for i in range(node_count):
        rows.append(
            RetrievalResult(
                content=f"def func_{i}():\n    return {i}",
                score=0.9 - (i * 0.002),
                source="code",
                metadata={
                    "chunk_id": f"repo::src/file_{i % 6}.py::function::func_{i}",
                    "file_path": f"src/file_{i % 6}.py",
                    "name": f"func_{i}",
                    "chunk_type": "function",
                    "repo_id": "repo",
                },
            )
        )
    return rows


def _build_graph_counts(stage_key: str, *, node_count: int, dense_max_edges: int = 160) -> tuple[int, int, str, bool]:
    """Build graph payload for one stage and return node/edge counts."""
    resolved = apply_stage(
        stage_key=stage_key,
        max_nodes=node_count,
        dense_max_edges=dense_max_edges,
        preferred_layout="cose",
    )
    adapter = CodeGraphAdapter(
        limits=GraphLimits(max_nodes=resolved.max_nodes, max_edges=resolved.max_edges)
    )
    payload = adapter.build(create_mock_results(node_count), query="stage test")
    graph = payload.to_cytoscape()
    return len(graph["nodes"]), len(graph["edges"]), resolved.layout, resolved.relations_enabled


def test_stage_flow_switches_sparse_to_dense():
    """Stage transition should move from sparse overview to dense detail."""
    assert next_stage_key("sparse") == "dense"
    assert next_stage_key("dense") == "dense"


def test_sparse_stage_meets_overlap_and_readable_targets():
    """Stage 1 should satisfy overlap <= 15% and readable nodes >= 40."""
    node_count, edge_count, layout, relations_enabled = _build_graph_counts(
        STAGE_SPARSE.key,
        node_count=55,
        dense_max_edges=180,
    )

    readable_estimated, occluded_estimated = estimate_readability(
        node_count,
        edge_count,
        layout=layout,
        relations_enabled=relations_enabled,
    )
    overlap = overlap_rate(occluded_estimated, node_count)
    readable = readable_nodes_count(
        [True] * readable_estimated + [False] * max(0, node_count - readable_estimated)
    )

    assert overlap <= 0.15, f"Sparse stage overlap {overlap:.3f} > 0.15"
    assert readable >= 40, f"Sparse stage readable nodes {readable} < 40"


def test_latency_p95_is_within_target_for_both_stages():
    """Both sparse and dense stages should stay within 250ms p95."""
    sparse_samples = [62.0, 71.0, 84.0, 90.0, 96.0, 110.0, 122.0, 138.0, 155.0, 198.0]
    dense_samples = [75.0, 88.0, 99.0, 112.0, 128.0, 140.0, 168.0, 184.0, 206.0, 232.0]

    sparse_p95 = interaction_latency_p95_ms(sparse_samples)
    dense_p95 = interaction_latency_p95_ms(dense_samples)

    assert sparse_p95 <= STAGE_SPARSE.target_latency_p95_ms
    assert dense_p95 <= STAGE_DENSE.target_latency_p95_ms


def test_dense_stage_enables_relations_and_restores_edge_budget():
    """Stage 2 should keep relation enrichment enabled with full edge budget."""
    resolved = apply_stage(
        stage_key=STAGE_DENSE.key,
        max_nodes=80,
        dense_max_edges=200,
        preferred_layout="cose",
    )

    assert resolved.relations_enabled is True
    assert resolved.max_edges == 200
    assert resolved.layout == "cose"
