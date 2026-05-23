"""Tests for code graph event contract normalization."""

from __future__ import annotations

import pytest

from ui.code_graph_events import normalize_event


def test_normalize_node_click_event():
    ev = normalize_event(
        {
            "event": "node_click",
            "node_id": "n1",
            "payload": {"x": 1},
        }
    )
    assert ev.event == "node_click"
    assert ev.node_id == "n1"
    assert ev.payload["x"] == 1


def test_normalize_unsupported_event_raises():
    with pytest.raises(ValueError):
        normalize_event({"event": "drag_start"})


def test_normalize_missing_fields_defaults():
    ev = normalize_event({"event": "canvas_click"})
    assert ev.node_id == ""
    assert ev.edge_id == ""
    assert ev.payload == {}
