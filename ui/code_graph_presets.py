"""Predefined graph presets for Small / Medium / Large scenarios.

Optimizes for overlap rate, readable node count, and interaction latency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphPreset:
    """Configuration preset for code graph rendering."""

    name: str
    description: str
    max_nodes: int
    max_edges: int
    layout: str  # breadthfirst or cose
    relations_enabled: bool
    target_overlap_rate: float  # <= this value is target
    target_readable_nodes: int  # >= this count is target
    target_latency_p95_ms: int  # <= this value is target


# Small preset: focus on primary call chain, avoid clutter
PRESET_SMALL = GraphPreset(
    name="Small",
    description="Primary call chain (< 30 nodes) — minimal noise",
    max_nodes=30,
    max_edges=60,
    layout="breadthfirst",
    relations_enabled=False,
    target_overlap_rate=0.15,
    target_readable_nodes=28,
    target_latency_p95_ms=200,
)

# Medium preset: balanced view, includes relations
PRESET_MEDIUM = GraphPreset(
    name="Medium",
    description="Balanced view (30–80 nodes) — key relations visible",
    max_nodes=80,
    max_edges=160,
    layout="cose",
    relations_enabled=True,
    target_overlap_rate=0.20,
    target_readable_nodes=72,
    target_latency_p95_ms=250,
)

# Large preset: comprehensive view, force-directed for flexibility
PRESET_LARGE = GraphPreset(
    name="Large",
    description="Full graph (80–150 nodes) — explore with care",
    max_nodes=150,
    max_edges=300,
    layout="breadthfirst",
    relations_enabled=True,
    target_overlap_rate=0.25,
    target_readable_nodes=130,
    target_latency_p95_ms=300,
)

PRESETS = {
    "small": PRESET_SMALL,
    "medium": PRESET_MEDIUM,
    "large": PRESET_LARGE,
}


def get_preset_by_node_count(node_count: int) -> GraphPreset:
    """Auto-select preset based on node count."""
    if node_count < 35:
        return PRESET_SMALL
    if node_count < 100:
        return PRESET_MEDIUM
    return PRESET_LARGE
