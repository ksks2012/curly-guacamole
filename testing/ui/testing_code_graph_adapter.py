"""Tests for CodeGraphAdapter payload mapping stability."""

from __future__ import annotations

from rag.retrieval.base import RetrievalResult
from ui.code_graph_adapter import CodeGraphAdapter
from ui.code_graph_contract import GraphLimits


def _res(query_label: str, chunk_id: str, related: list[dict]) -> RetrievalResult:
    return RetrievalResult(
        content=f"content-{query_label}",
        score=0.9,
        source="code",
        metadata={
            "chunk_id": chunk_id,
            "name": "handler",
            "file_path": "src/handler.py",
            "chunk_type": "function",
            "repo_id": "repo1",
            "start_line": 10,
            "related_blocks": related,
        },
    )


def test_adapter_maps_primary_and_related_nodes_edges():
    related = [
        {
            "target_id": "repo1::src/service.py::function::run",
            "target_name": "run",
            "target_file_path": "src/service.py",
            "target_chunk_type": "function",
            "edge_type": "CALLS",
            "direction": "outgoing",
            "score": 0.88,
            "explain": "outgoing calls relation",
        }
    ]
    r = _res("q", "repo1::src/handler.py::function::handle", related)

    payload = CodeGraphAdapter().build([r], query="How auth works?")

    assert len(payload.nodes) == 2
    assert len(payload.edges) == 1
    assert payload.meta["primary_count"] == 1
    assert payload.meta["related_primary_count"] == 1
    assert payload.meta["related_count_hit_rate"] == 1.0
    assert payload.meta["related_ref_count_before_limit"] == 1
    assert payload.meta["query"] == "How auth works?"


def test_adapter_related_hit_rate_with_mixed_results():
    with_rel = [
        {
            "target_id": "repo1::src/service.py::function::run",
            "target_name": "run",
            "target_file_path": "src/service.py",
            "target_chunk_type": "function",
            "edge_type": "CALLS",
            "direction": "outgoing",
            "score": 0.88,
            "explain": "outgoing calls relation",
        }
    ]
    r1 = _res("q1", "repo1::src/handler.py::function::handle", with_rel)
    r2 = _res("q2", "repo1::src/empty.py::function::noop", [])

    payload = CodeGraphAdapter().build([r1, r2], query="q")
    assert payload.meta["primary_count"] == 2
    assert payload.meta["related_primary_count"] == 1
    assert payload.meta["related_count_hit_rate"] == 0.5


def test_adapter_dedups_edges_by_signature():
    related = [
        {
            "target_id": "repo1::a.py::function::x",
            "target_name": "x",
            "target_file_path": "a.py",
            "target_chunk_type": "function",
            "edge_type": "CALLS",
            "direction": "outgoing",
            "score": 0.8,
            "explain": "e1",
        },
        {
            "target_id": "repo1::a.py::function::x",
            "target_name": "x",
            "target_file_path": "a.py",
            "target_chunk_type": "function",
            "edge_type": "CALLS",
            "direction": "outgoing",
            "score": 0.7,
            "explain": "e2",
        },
    ]
    r = _res("q", "repo1::h.py::function::h", related)

    payload = CodeGraphAdapter().build([r], query="q")
    assert len(payload.edges) == 1


def test_adapter_applies_limits_and_keeps_valid_payload():
    results = []
    for i in range(5):
        related = [
            {
                "target_id": f"repo1::x{i}.py::function::t",
                "target_name": "t",
                "target_file_path": f"x{i}.py",
                "target_chunk_type": "function",
                "edge_type": "CALLS",
                "direction": "outgoing",
                "score": 0.8,
                "explain": "rel",
            }
        ]
        results.append(_res("q", f"repo1::p{i}.py::function::s", related))

    adapter = CodeGraphAdapter(limits=GraphLimits(max_nodes=4, max_edges=3))
    payload = adapter.build(results, query="q")

    assert len(payload.nodes) <= 4
    assert len(payload.edges) <= 3
    assert payload.meta["truncated_nodes"] is True


def test_adapter_handles_query_formats_stably():
    weird_query = "auth?[]{}::\\n中文/emoji-like"
    r = _res("weird", "repo1::a.py::function::f", related=[])

    payload = CodeGraphAdapter().build([r], query=weird_query)
    d = payload.to_cytoscape()

    assert d["meta"]["query"] == weird_query
    assert isinstance(d["nodes"], list)
