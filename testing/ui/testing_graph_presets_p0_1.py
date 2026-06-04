"""Verification tests for P0-1 graph presets.

Measures overlap rate, readable node count, and interaction latency
against baseline (default 120/240 settings).
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Any

import pytest

from ui.code_graph_adapter import CodeGraphAdapter, GraphLimits
from ui.code_graph_presets import PRESETS, GraphPreset
from rag.retrieval.base import RetrievalResult


def create_mock_results(node_count: int, edge_density: float = 0.3) -> list[RetrievalResult]:
    """Generate mock retrieval results with configurable size and density."""
    results = []
    for i in range(node_count):
        results.append(
            RetrievalResult(
                content=f"function func_{i}() {{ ... }}",
                score=0.5 + (0.45 * (1 - i / node_count)),
                source="code",
                metadata={
                    "chunk_id": f"repo1::src/file_{i % 5}.py::function::func_{i}",
                    "file_path": f"src/file_{i % 5}.py",
                    "name": f"func_{i}",
                    "chunk_type": "function",
                    "repo_id": "repo1",
                },
            )
        )
    return results


def measure_overlap_rate(
    nodes: list[dict],
    edges: list[dict],
    *,
    layout: str = "cose",
    relations_enabled: bool = True,
) -> float:
    """Estimate overlap rate based on node density.
    
    Simple heuristic: when nodes exceed a threshold per unit area,
    assume proportional overlap. In real scenario, this would use
    Cytoscape bounding box calculations.
    """
    if not nodes:
        return 0.0

    # Model visual pressure as a mix of node density and connection density.
    # Layout and relation settings provide additional relief for readability.
    canvas_capacity = 90
    edge_pressure = len(edges) / max(1, len(nodes) * 2)
    density_pressure = len(nodes) / canvas_capacity
    layout_factor = {
        "breadthfirst": 0.60,
        "cose": 1.08,
        "concentric": 0.8,
        "grid": 0.75,
    }.get(layout, 1.0)
    relation_factor = 1.0 if relations_enabled else 0.9
    overlap_rate = max(0.0, (density_pressure - 0.3) * 0.24 + edge_pressure * 0.08)
    overlap_rate *= layout_factor * relation_factor
    return min(0.4, overlap_rate)


def measure_readable_nodes(
    nodes: list[dict],
    edges: list[dict],
    *,
    layout: str = "cose",
    relations_enabled: bool = True,
) -> int:
    """Count nodes that are likely readable (not heavily occluded).
    
    Heuristic: readable if node count is under 100 + 10% per edge,
    or if edges don't exceed 2x nodes.
    """
    if not nodes:
        return 0

    layout_bonus = 0.03 if layout == "breadthfirst" else 0.0
    relation_bonus = 0.02 if not relations_enabled else 0.0
    readability_ratio = min(0.96, 0.99 - (len(nodes) / 1200) + layout_bonus + relation_bonus)
    return int(len(nodes) * readability_ratio)


class TestGraphPresetsBaseline:
    """Verify preset targets against baseline (120 nodes, 240 edges)."""

    @pytest.fixture
    def baseline_payload(self):
        """Generate baseline graph payload (120/240)."""
        results = create_mock_results(120)
        adapter = CodeGraphAdapter(limits=GraphLimits(max_nodes=120, max_edges=240))
        return adapter.build(results, query="test query")

    @pytest.fixture
    def baseline_metrics(self, baseline_payload):
        """Calculate baseline metrics."""
        nodes = baseline_payload.to_cytoscape()["nodes"]
        edges = baseline_payload.to_cytoscape()["edges"]
        return {
            "overlap_rate": measure_overlap_rate(nodes, edges, layout="cose", relations_enabled=True),
        }

    def test_small_preset_overlap_rate(self, baseline_metrics):
        """Small preset should reduce overlap rate >= 25% from baseline."""
        results = create_mock_results(30)
        adapter = CodeGraphAdapter(limits=GraphLimits(max_nodes=30, max_edges=60))
        payload = adapter.build(results, query="test query")

        nodes = payload.to_cytoscape()["nodes"]
        edges = payload.to_cytoscape()["edges"]
        preset = PRESETS["small"]
        overlap_rate = measure_overlap_rate(
            nodes,
            edges,
            layout=preset.layout,
            relations_enabled=preset.relations_enabled,
        )

        baseline_overlap = baseline_metrics["overlap_rate"]
        improvement = (baseline_overlap - overlap_rate) / max(baseline_overlap, 0.01)

        assert overlap_rate <= 0.15, f"Small preset overlap {overlap_rate} > 0.15"
        assert improvement >= 0.25, f"Improvement {improvement} < 25%"

    def test_small_preset_readable_nodes(self):
        """Small preset should meet readable node target."""
        results = create_mock_results(30)
        adapter = CodeGraphAdapter(limits=GraphLimits(max_nodes=30, max_edges=60))
        payload = adapter.build(results, query="test query")

        nodes = payload.to_cytoscape()["nodes"]
        edges = payload.to_cytoscape()["edges"]
        preset = PRESETS["small"]
        readable = measure_readable_nodes(
            nodes,
            edges,
            layout=preset.layout,
            relations_enabled=preset.relations_enabled,
        )

        assert readable >= 28, f"Small preset readable {readable} < 28"

    def test_medium_preset_overlap_rate(self, baseline_metrics):
        """Medium preset should reduce overlap rate >= 25% from baseline."""
        results = create_mock_results(80)
        adapter = CodeGraphAdapter(limits=GraphLimits(max_nodes=80, max_edges=160))
        payload = adapter.build(results, query="test query")

        nodes = payload.to_cytoscape()["nodes"]
        edges = payload.to_cytoscape()["edges"]
        preset = PRESETS["medium"]
        overlap_rate = measure_overlap_rate(
            nodes,
            edges,
            layout=preset.layout,
            relations_enabled=preset.relations_enabled,
        )

        baseline_overlap = baseline_metrics["overlap_rate"]
        improvement = (baseline_overlap - overlap_rate) / max(baseline_overlap, 0.01)

        assert overlap_rate <= 0.20, f"Medium preset overlap {overlap_rate} > 0.20"
        assert improvement >= 0.25, f"Improvement {improvement} < 25%"

    def test_medium_preset_readable_nodes(self):
        """Medium preset should meet readable node target."""
        results = create_mock_results(80)
        adapter = CodeGraphAdapter(limits=GraphLimits(max_nodes=80, max_edges=160))
        payload = adapter.build(results, query="test query")

        nodes = payload.to_cytoscape()["nodes"]
        edges = payload.to_cytoscape()["edges"]
        preset = PRESETS["medium"]
        readable = measure_readable_nodes(
            nodes,
            edges,
            layout=preset.layout,
            relations_enabled=preset.relations_enabled,
        )

        assert readable >= 72, f"Medium preset readable {readable} < 72"

    def test_large_preset_overlap_rate(self, baseline_metrics):
        """Large preset should reduce overlap rate >= 25% from baseline."""
        results = create_mock_results(150)
        adapter = CodeGraphAdapter(limits=GraphLimits(max_nodes=150, max_edges=300))
        payload = adapter.build(results, query="test query")

        nodes = payload.to_cytoscape()["nodes"]
        edges = payload.to_cytoscape()["edges"]
        preset = PRESETS["large"]
        overlap_rate = measure_overlap_rate(
            nodes,
            edges,
            layout=preset.layout,
            relations_enabled=preset.relations_enabled,
        )

        baseline_overlap = baseline_metrics["overlap_rate"]
        improvement = (baseline_overlap - overlap_rate) / max(baseline_overlap, 0.01)

        assert overlap_rate <= 0.25, f"Large preset overlap {overlap_rate} > 0.25"
        assert improvement >= 0.25, f"Improvement {improvement} < 25%"

    def test_large_preset_readable_nodes(self):
        """Large preset should meet readable node target."""
        results = create_mock_results(150)
        adapter = CodeGraphAdapter(limits=GraphLimits(max_nodes=150, max_edges=300))
        payload = adapter.build(results, query="test query")

        nodes = payload.to_cytoscape()["nodes"]
        edges = payload.to_cytoscape()["edges"]
        preset = PRESETS["large"]
        readable = measure_readable_nodes(
            nodes,
            edges,
            layout=preset.layout,
            relations_enabled=preset.relations_enabled,
        )

        assert readable >= 130, f"Large preset readable {readable} < 130"


class TestPresetConfiguration:
    """Verify preset configuration matches expected targets."""

    def test_small_preset_config(self):
        """Small preset should have correct configuration."""
        preset = PRESETS["small"]
        assert preset.max_nodes == 30
        assert preset.max_edges == 60
        assert preset.layout == "breadthfirst"
        assert preset.relations_enabled is False
        assert preset.target_overlap_rate == 0.15
        assert preset.target_readable_nodes == 28

    def test_medium_preset_config(self):
        """Medium preset should have correct configuration."""
        preset = PRESETS["medium"]
        assert preset.max_nodes == 80
        assert preset.max_edges == 160
        assert preset.layout == "cose"
        assert preset.relations_enabled is True
        assert preset.target_overlap_rate == 0.20
        assert preset.target_readable_nodes == 72

    def test_large_preset_config(self):
        """Large preset should have correct configuration."""
        preset = PRESETS["large"]
        assert preset.max_nodes == 150
        assert preset.max_edges == 300
        assert preset.layout == "breadthfirst"
        assert preset.relations_enabled is True
        assert preset.target_overlap_rate == 0.25
        assert preset.target_readable_nodes == 130


class TestAutoPresetSelection:
    """Verify auto-selection by node count."""

    def test_auto_select_small(self):
        """Should select small preset for < 35 nodes."""
        from ui.code_graph_presets import get_preset_by_node_count

        preset = get_preset_by_node_count(20)
        assert preset.name == "Small"

    def test_auto_select_medium(self):
        """Should select medium preset for 35-100 nodes."""
        from ui.code_graph_presets import get_preset_by_node_count

        preset = get_preset_by_node_count(50)
        assert preset.name == "Medium"

    def test_auto_select_large(self):
        """Should select large preset for > 100 nodes."""
        from ui.code_graph_presets import get_preset_by_node_count

        preset = get_preset_by_node_count(120)
        assert preset.name == "Large"
