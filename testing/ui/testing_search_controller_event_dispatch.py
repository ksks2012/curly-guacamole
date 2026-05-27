"""Tests for SearchController event dispatch and handler routing."""

from __future__ import annotations

from langchain_core.documents import Document

from ui.search_controller import SearchController


class _FakeClient:
    pass


def test_event_dispatch_node_click_with_in_memory_chunk():
    """Verify node click prefers in-memory chunk data over graph node payload."""
    ctrl = SearchController(_FakeClient())
    
    # Add in-memory chunk
    doc = Document(
        page_content="def foo(): pass",
        metadata={
            "chunk_id": "repo1::a.py::function::foo",
            "file_path": "a.py",
            "name": "foo",
        },
    )
    ctrl._vector = [(doc, 0.85)]

    # Dispatch node_click with same ID
    ctrl.handle_graph_node_click(
        "repo1::a.py::function::foo",
        {"id": "fallback", "label": "fallback", "file_path": "fallback.py"},
    )

    # Should have selected in-memory chunk, not fallback node
    assert ctrl.selected_metadata["chunk_id"] == "repo1::a.py::function::foo"
    assert ctrl.selected_metadata["file_path"] == "a.py"
    assert "_detail_kind" not in ctrl.selected_metadata  # chunk selection, not node


def test_event_dispatch_node_click_fallback_to_graph_payload():
    """Verify node click falls back to graph payload when chunk not found."""
    ctrl = SearchController(_FakeClient())

    ctrl.handle_graph_node_click(
        "missing::node::id",
        {
            "id": "repo1::b.py::function::bar",
            "label": "bar",
            "file_path": "b.py",
            "chunk_type": "function",
            "score": 0.72,
            "is_primary": False,
            "metadata": {"mapped_by": "symbol_fallback"},
        },
    )

    meta = ctrl.selected_metadata
    assert meta["_detail_kind"] == "node"
    assert meta["graph_node_id"] == "repo1::b.py::function::bar"
    assert meta["label"] == "bar"
    assert meta["file_path"] == "b.py"


def test_event_dispatch_edge_click_sets_edge_metadata():
    """Verify edge click populates edge detail metadata correctly."""
    ctrl = SearchController(_FakeClient())

    ctrl.handle_graph_edge_click(
        {
            "id": "e1|e2|CALLS|outgoing",
            "source": "e1",
            "target": "e2",
            "edge_type": "CALLS",
            "direction": "outgoing",
            "score": 0.88,
            "explain": "function bar calls function baz",
            "metadata": {
                "evidence_count": 3,
                "edge_types": ["CALLS", "NEARBY"],
            },
        },
    )

    meta = ctrl.selected_metadata
    assert meta["_detail_kind"] == "edge"
    assert meta["graph_edge_id"] == "e1|e2|CALLS|outgoing"
    assert meta["source_id"] == "e1"
    assert meta["target_id"] == "e2"
    assert meta["edge_type"] == "CALLS"
    assert meta["explain"] == "function bar calls function baz"
    assert meta["evidence_count"] == 3
    assert meta["edge_types"] == ["CALLS", "NEARBY"]
    assert meta["_content"] == "function bar calls function baz"


def test_event_dispatch_canvas_click_clears_selection():
    """Verify canvas click clears all selected metadata."""
    ctrl = SearchController(_FakeClient())
    
    # Set up edge selection
    ctrl.handle_graph_edge_click(
        {
            "id": "e1",
            "source": "a",
            "target": "b",
            "edge_type": "CALLS",
            "direction": "outgoing",
            "score": 0.9,
            "explain": "test",
            "metadata": {},
        },
    )
    assert ctrl.selected_metadata  # Verify it's set

    # Dispatch canvas click
    ctrl.handle_graph_canvas_click()

    # Should be cleared
    assert ctrl.selected_metadata == {}


def test_event_dispatch_preserves_node_metadata_fields():
    """Verify graph node click preserves all relevant metadata fields."""
    ctrl = SearchController(_FakeClient())

    node_data = {
        "id": "repo1::complex.py::class::Handler",
        "label": "Handler",
        "file_path": "complex.py",
        "chunk_type": "class",
        "score": 0.91,
        "is_primary": False,
        "metadata": {
            "start_line": 42,
            "end_line": 120,
            "branch": "main",
            "repo_id": "repo1",
            "mapping_strategy": "symbol_key_fallback",
        },
    }
    ctrl.select_graph_node(node_data)

    meta = ctrl.selected_metadata
    assert meta["graph_node_id"] == node_data["id"]
    assert meta["file_path"] == node_data["file_path"]
    assert meta["chunk_type"] == node_data["chunk_type"]
    assert meta["score"] == 0.91
    assert meta["is_primary"] is False
    # Merged metadata fields
    assert meta["start_line"] == 42
    assert meta["mapping_strategy"] == "symbol_key_fallback"


def test_event_dispatch_edge_without_metadata_defaults_gracefully():
    """Verify edge click handles missing metadata fields gracefully."""
    ctrl = SearchController(_FakeClient())

    ctrl.handle_graph_edge_click(
        {
            "id": "e1",
            "source": "a",
            "target": "b",
            "edge_type": "CALLS",
            "direction": "outgoing",
            "score": 0.5,
            "explain": "",
        }
        # No metadata field
    )

    meta = ctrl.selected_metadata
    assert meta["evidence_count"] == 0
    assert meta["edge_types"] == []
    assert meta["_content"] == "No explanation available for this edge."


def test_event_dispatch_sequential_clicks_replace_state():
    """Verify sequential event dispatches properly replace state."""
    ctrl = SearchController(_FakeClient())

    # First: edge click
    ctrl.handle_graph_edge_click(
        {
            "id": "e1",
            "source": "a",
            "target": "b",
            "edge_type": "CALLS",
            "direction": "outgoing",
            "score": 0.8,
            "explain": "edge detail",
            "metadata": {},
        },
    )
    assert ctrl.selected_metadata["_detail_kind"] == "edge"

    # Second: canvas click
    ctrl.handle_graph_canvas_click()
    assert ctrl.selected_metadata == {}

    # Third: node click
    ctrl.select_graph_node(
        {"id": "n1", "label": "node1", "file_path": "f.py", "chunk_type": "func"}
    )
    assert ctrl.selected_metadata["_detail_kind"] == "node"
    assert ctrl.selected_metadata["graph_node_id"] == "n1"
