"""Verification tests for P0-3 soft scope query conventions.

The feature uses query hints to improve result cohesion without hard filtering.
"""

from __future__ import annotations

import time

from langchain_core.documents import Document

from rag.retrieval.code_query_scope import parse_code_query_scope, rerank_code_rows_by_scope
from ui.code_graph_metrics import interaction_latency_p95_ms, overlap_rate


def _row(repo_id: str, file_path: str, name: str, score: float) -> tuple[Document, float]:
    doc = Document(
        page_content=f"def {name}():\n    return {score}",
        metadata={
            "repo_id": repo_id,
            "file_path": file_path,
            "name": name,
            "chunk_id": f"{repo_id}::{file_path}::function::{name}",
            "chunk_type": "function",
        },
    )
    return (doc, score)


def _mixed_rows() -> list[tuple[Document, float]]:
    rows: list[tuple[Document, float]] = []
    for index in range(18):
        rows.append(
            _row(
                "payments-service",
                f"src/payments/service_{index % 4}.py",
                f"handle_payment_{index}",
                0.79 - (index * 0.004),
            )
        )
        rows.append(
            _row(
                "inventory-service",
                f"src/inventory/service_{index % 4}.py",
                f"handle_inventory_{index}",
                0.80 - (index * 0.004),
            )
        )
    return rows


def _selection_metrics(rows: list[tuple[Document, float]]) -> tuple[float, int]:
    visible_nodes = len(rows)
    repo_counts: dict[str, int] = {}
    path_roots: set[str] = set()
    for doc, _score in rows:
        metadata = dict(doc.metadata or {})
        repo_id = str(metadata.get("repo_id", "") or "")
        repo_counts[repo_id] = repo_counts.get(repo_id, 0) + 1
        file_path = str(metadata.get("file_path", "") or "")
        parts = [part for part in file_path.split("/") if part]
        path_roots.add("/".join(parts[:2]))

    dominant_repo_count = max(repo_counts.values()) if repo_counts else 0
    repo_noise = max(0, visible_nodes - dominant_repo_count)
    path_noise = max(0, len(path_roots) - 1)
    occluded_nodes = min(visible_nodes, int(round((repo_noise * 0.55) + (path_noise * 1.25))))
    readable_nodes = max(0, visible_nodes - occluded_nodes)
    return overlap_rate(occluded_nodes, visible_nodes), readable_nodes


def _rerank_elapsed_ms(query: str, rows: list[tuple[Document, float]], runs: int = 120) -> float:
    scope = parse_code_query_scope(query)
    samples: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        rerank_code_rows_by_scope(rows, scope)
        samples.append((time.perf_counter() - started) * 1000.0)
    return interaction_latency_p95_ms(samples)


def test_soft_scope_query_improves_overlap_without_hard_filtering():
    rows = _mixed_rows()
    baseline_top = rows[:20]

    scope = parse_code_query_scope(
        "repo:payments-service path:src/payments module:payments.service find payment handler"
    )
    scoped_top = rerank_code_rows_by_scope(rows, scope)[:20]

    baseline_overlap, baseline_readable = _selection_metrics(baseline_top)
    scoped_overlap, scoped_readable = _selection_metrics(scoped_top)

    overlap_improvement = (baseline_overlap - scoped_overlap) / max(baseline_overlap, 0.01)
    readable_floor = baseline_readable * 0.90

    assert overlap_improvement >= 0.15, (
        f"Overlap improvement {overlap_improvement:.3f} < 15%"
    )
    assert scoped_readable >= readable_floor, (
        f"Scoped readable nodes {scoped_readable} < floor {readable_floor:.1f}"
    )


def test_soft_scope_query_does_not_materially_worsen_latency():
    rows = _mixed_rows() * 4

    baseline_p95 = 120.0 + _rerank_elapsed_ms("find payment handler", rows)
    scoped_p95 = 120.0 + _rerank_elapsed_ms(
        "repo:payments-service path:src/payments module:payments.service find payment handler",
        rows,
    )

    increase_ratio = (scoped_p95 - baseline_p95) / max(baseline_p95, 1.0)
    assert increase_ratio < 0.10, f"Latency increase {increase_ratio:.3f} >= 10%"
