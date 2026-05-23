"""Event contract for code graph interactions (NiceGUI <-> Cytoscape.js)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


GraphEventType = Literal[
    "node_click",
    "edge_click",
    "canvas_click",
    "expand_neighbors",
    "focus_node",
]


@dataclass
class CodeGraphEvent:
    """Normalized graph event structure shared across UI layers."""

    event: GraphEventType
    node_id: str = ""
    edge_id: str = ""
    payload: dict = field(default_factory=dict)


def normalize_event(raw: dict) -> CodeGraphEvent:
    """Convert arbitrary raw event dict into a strict CodeGraphEvent."""
    event = str(raw.get("event", "")).strip()
    allowed = {
        "node_click",
        "edge_click",
        "canvas_click",
        "expand_neighbors",
        "focus_node",
    }
    if event not in allowed:
        raise ValueError(f"Unsupported graph event: {event!r}")

    return CodeGraphEvent(
        event=event,  # type: ignore[arg-type]
        node_id=str(raw.get("node_id", "") or ""),
        edge_id=str(raw.get("edge_id", "") or ""),
        payload=dict(raw.get("payload", {}) or {}),
    )
