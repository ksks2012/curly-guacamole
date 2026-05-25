"""Code graph metrics utilities.

Definitions are kept explicit so UI and tests use the same formulas.
"""

from __future__ import annotations

from typing import Iterable


def overlap_rate(occluded_nodes: int, visible_nodes: int) -> float:
    """Return node overlap ratio: occluded / visible."""
    if visible_nodes <= 0:
        return 0.0
    clamped_occluded = max(0, min(int(occluded_nodes), int(visible_nodes)))
    return clamped_occluded / float(visible_nodes)


def readable_nodes_count(readability_flags: Iterable[bool]) -> int:
    """Return count of readable nodes from per-node readability flags."""
    return sum(1 for flag in readability_flags if bool(flag))


def interaction_latency_p95_ms(samples_ms: list[float]) -> float:
    """Return p95 latency in milliseconds for a list of latency samples."""
    if not samples_ms:
        return 0.0

    ordered = sorted(max(0.0, float(v)) for v in samples_ms)
    if len(ordered) == 1:
        return ordered[0]

    rank = int((len(ordered) - 1) * 0.95)
    return ordered[rank]


def estimate_readability(
    node_count: int,
    edge_count: int,
    *,
    layout: str,
    relations_enabled: bool,
) -> tuple[int, int]:
    """Estimate readable and occluded node counts for heuristic verification.

    The estimate is deterministic and tuned for comparative checks between
    sparse and enriched graph stages.
    """
    if node_count <= 0:
        return 0, 0

    density = node_count / 85.0
    edge_pressure = edge_count / max(1.0, node_count * 2.0)

    layout_penalty = 0.0
    if layout == "cose":
        layout_penalty = 0.045
    elif layout == "breadthfirst":
        layout_penalty = 0.015
    else:
        layout_penalty = 0.03

    relation_penalty = 0.06 if relations_enabled else 0.015
    occlusion_ratio = max(0.0, (density - 0.35) * 0.14 + edge_pressure * 0.08 + layout_penalty + relation_penalty)
    occlusion_ratio = min(0.45, occlusion_ratio)

    occluded = int(round(node_count * occlusion_ratio))
    readable = max(0, node_count - occluded)
    return readable, occluded
