"""Tests for SearchController graph event handling paths."""

from __future__ import annotations

from langchain_core.documents import Document

from ui.search_controller import SearchController


class _FakeClient:
    pass


def test_handle_graph_node_click_prefers_existing_chunk_selection():
    ctrl = SearchController(_FakeClient())
    doc = Document(
        page_content="def f():\n    return 1",
        metadata={
            "chunk_id": "repo1::a.py::function::f",
            "file_path": "a.py",
            "name": "f",
        },
    )
    ctrl._vector = [(doc, 0.82)]

    ctrl.handle_graph_node_click(
        "repo1::a.py::function::f",
        {
            "id": "repo1::a.py::function::f",
            "label": "fallback",
            "file_path": "fallback.py",
            "chunk_type": "function",
        },
    )

    assert ctrl.selected_metadata["chunk_id"] == "repo1::a.py::function::f"
    assert "graph_node_id" not in ctrl.selected_metadata


def test_handle_graph_edge_click_sets_edge_detail_metadata():
    ctrl = SearchController(_FakeClient())

    ctrl.handle_graph_edge_click(
        {
            "id": "e1",
            "source": "a",
            "target": "b",
            "edge_type": "CALLS",
            "direction": "outgoing",
            "score": 0.77,
            "explain": "outgoing calls relation",
            "metadata": {"evidence_count": 2, "edge_types": ["CALLS", "NEARBY"]},
        }
    )

    meta = ctrl.selected_metadata
    assert meta["_detail_kind"] == "edge"
    assert meta["graph_edge_id"] == "e1"
    assert meta["explain"] == "outgoing calls relation"
    assert meta["evidence_count"] == 2
    assert meta["edge_types"] == ["CALLS", "NEARBY"]


def test_handle_graph_canvas_click_clears_selected_metadata():
    ctrl = SearchController(_FakeClient())
    ctrl.select_graph_node({"id": "n1", "label": "N1"})
    assert ctrl.selected_metadata

    ctrl.handle_graph_canvas_click()
    assert ctrl.selected_metadata == {}
