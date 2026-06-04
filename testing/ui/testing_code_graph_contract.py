"""Tests for code graph payload contract and limits."""

from __future__ import annotations

import pytest

from ui.code_graph_contract import (
    CodeGraphEdge,
    CodeGraphNode,
    CodeGraphPayload,
    GraphLimits,
    apply_limits,
    validate_payload,
)


def _node(i: int) -> CodeGraphNode:
    return CodeGraphNode(
        id=f"n{i}",
        label=f"Node {i}",
        file_path="a.py",
        chunk_type="function",
        score=0.9,
        is_primary=(i == 0),
    )


def _edge(i: int, s: str, t: str) -> CodeGraphEdge:
    return CodeGraphEdge(
        id=f"e{i}",
        source=s,
        target=t,
        edge_type="CALLS",
        direction="outgoing",
        score=0.8,
        explain="outgoing calls relation",
    )


def test_validate_payload_ok():
    payload = CodeGraphPayload(
        nodes=[_node(0), _node(1)],
        edges=[_edge(0, "n0", "n1")],
        meta={"query": "q"},
    )
    validate_payload(payload)


def test_validate_payload_duplicate_node_id_raises():
    payload = CodeGraphPayload(
        nodes=[_node(0), _node(0)],
        edges=[],
    )
    with pytest.raises(ValueError):
        validate_payload(payload)


def test_validate_payload_missing_edge_target_raises():
    payload = CodeGraphPayload(
        nodes=[_node(0)],
        edges=[_edge(0, "n0", "nX")],
    )
    with pytest.raises(ValueError):
        validate_payload(payload)


def test_apply_limits_truncates_nodes_and_edges():
    payload = CodeGraphPayload(
        nodes=[_node(i) for i in range(5)],
        edges=[
            _edge(0, "n0", "n1"),
            _edge(1, "n1", "n2"),
            _edge(2, "n2", "n3"),
            _edge(3, "n3", "n4"),
        ],
    )
    out = apply_limits(payload, GraphLimits(max_nodes=3, max_edges=2))

    assert len(out.nodes) == 3
    assert len(out.edges) <= 2
    assert out.meta["truncated_nodes"] is True
    assert out.meta["node_count_before_limit"] == 5
    assert out.meta["node_count"] == 3


def test_to_cytoscape_shape_stable():
    payload = CodeGraphPayload(
        nodes=[_node(0)],
        edges=[],
        meta={"query": "weird query [] {} / unicode 測試"},
    )
    d = payload.to_cytoscape()

    assert "nodes" in d and isinstance(d["nodes"], list)
    assert "edges" in d and isinstance(d["edges"], list)
    assert "meta" in d and isinstance(d["meta"], dict)
    assert d["meta"]["query"].startswith("weird query")
