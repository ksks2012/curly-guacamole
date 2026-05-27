"""UI helpers and contracts for dashboard composition."""

from ui.code_graph_adapter import CodeGraphAdapter
from ui.code_graph_contract import (
	CodeGraphEdge,
	CodeGraphNode,
	CodeGraphPayload,
	GraphLimits,
	apply_limits,
	validate_payload,
)
from ui.code_graph_events import CodeGraphEvent, GraphEventType, normalize_event

__all__ = [
	"CodeGraphAdapter",
	"CodeGraphNode",
	"CodeGraphEdge",
	"CodeGraphPayload",
	"GraphLimits",
	"apply_limits",
	"validate_payload",
	"CodeGraphEvent",
	"GraphEventType",
	"normalize_event",
]
