"""Smoke tests for code graph UI rendering and event handling."""

from __future__ import annotations

import pytest


class MockController:
    """Mock SearchController for UI smoke tests."""

    def __init__(self):
        self._vector = []
        self._reranked = None
        self._bm25 = None
        self._hybrid = None
        self._metadata = {}
        self._last_query = ""
        self._rerank_calls = 0

    @property
    def vector_results(self):
        return self._vector

    @property
    def reranked_results(self):
        return self._reranked

    @property
    def bm25_results(self):
        return self._bm25

    @property
    def hybrid_results(self):
        return self._hybrid

    @property
    def selected_metadata(self):
        return self._metadata

    @property
    def last_query(self):
        return self._last_query

    def run_search(
        self,
        query,
        k,
        fetch_k,
        use_rerank,
        use_hybrid=False,
        result_scope="all",
        apply_filter=True,
        include_relations=False,
    ):
        self._last_query = query
        if use_rerank:
            self._rerank_calls += 1
        return None

    def select_chunk(self, doc, score, score_key):
        self._metadata = {
            score_key: score,
            **doc.metadata,
            "_content": doc.page_content[:600],
        }

    def select_chunk_by_id(self, chunk_id):
        return False

    def handle_graph_node_click(self, node_id, node_data):
        self._metadata = {"_detail_kind": "node", **node_data}

    def handle_graph_edge_click(self, edge_data):
        self._metadata = {"_detail_kind": "edge", **edge_data}

    def handle_graph_canvas_click(self):
        self._metadata = {}


def test_mock_controller_initializes():
    """Verify mock controller baseline state."""
    ctrl = MockController()
    assert ctrl.selected_metadata == {}
    assert ctrl.last_query == ""


def test_mock_controller_run_search_updates_query():
    """Verify mock controller tracks search queries."""
    ctrl = MockController()
    error = ctrl.run_search("test query", 5, 20, False, result_scope="code")
    assert error is None
    assert ctrl.last_query == "test query"


def test_mock_controller_handles_rerank_tracking():
    """Verify mock controller tracks rerank invocations."""
    ctrl = MockController()
    ctrl.run_search("q1", 5, 20, use_rerank=True)
    ctrl.run_search("q2", 5, 20, use_rerank=False)
    ctrl.run_search("q3", 5, 20, use_rerank=True)
    assert ctrl._rerank_calls == 2


def test_mock_controller_graph_event_handlers():
    """Verify all three graph event handlers work via mock."""
    ctrl = MockController()

    # Node click
    ctrl.handle_graph_node_click("n1", {"id": "n1", "label": "Node1"})
    assert ctrl.selected_metadata["_detail_kind"] == "node"

    # Canvas click
    ctrl.handle_graph_canvas_click()
    assert ctrl.selected_metadata == {}

    # Edge click
    ctrl.handle_graph_edge_click(
        {"id": "e1", "source": "a", "target": "b", "edge_type": "CALLS"}
    )
    assert ctrl.selected_metadata["_detail_kind"] == "edge"


def test_code_graph_tab_build_requires_controller():
    """Verify code graph tab build function accepts SearchController interface."""
    # This smoke test verifies that the build() signature is compatible
    # with our MockController.
    from ui.code_graph_tab import build

    ctrl = MockController()
    # Just verify the function accepts the controller without crashing
    # (actual NiceGUI rendering would require full environment)
    import inspect

    sig = inspect.signature(build)
    assert "ctrl" in sig.parameters


def test_code_graph_tab_mentions_soft_scope_query_hints():
    """Verify code graph tab exposes repo/path/module query hint guidance."""
    from ui.code_graph_tab import build

    import inspect

    source = inspect.getsource(build)
    assert "repo:<repo-id>" in source
    assert "path:<path-fragment>" in source
    assert "module:<module-prefix>" in source


def test_search_tab_mentions_soft_scope_query_hints_for_code_scope():
    """Verify shared search tab documents soft scope query conventions."""
    from ui.search_tab import build

    import inspect

    source = inspect.getsource(build)
    assert "repo:payments" in source
    assert "path:src/payments" in source
    assert "module:payments.service" in source


def test_search_tab_embedded_graph_panel_fills_result_column_width():
    """Verify embedded graph panel uses full width within the result column."""
    from ui.search_tab import build

    import inspect

    source = inspect.getsource(build)
    assert "width: 100%" in source
    assert "min-width: 0" in source


def test_code_graph_tab_has_repo_dropdown_selection_flow():
    """Verify code graph tab uses repo dropdown options from controller."""
    from ui.code_graph_tab import build

    import inspect

    source = inspect.getsource(build)
    assert "repo_select = ui.select" in source
    assert "ctrl.list_code_repo_ids()" in source
    assert "repo:" in source


def test_code_graph_tab_uses_fixed_graph_and_detail_panel_sizes():
    """Verify graph and detail panel dimensions are fixed for stable rendering."""
    from ui.code_graph_tab import build

    import inspect

    source = inspect.getsource(build)
    assert "height: 40rem" in source
    assert "width: 26rem" in source


def test_code_graph_tab_splits_connected_components_into_multiple_graphs():
    """Verify connected components are rendered as separate graph canvases."""
    from ui.code_graph_tab import build

    import inspect

    source = inspect.getsource(build)
    assert "_split_connected_components" in source
    assert "Graphs:" in source
    assert "for (let i = 0; i < payloads.length; i += 1)" in source
