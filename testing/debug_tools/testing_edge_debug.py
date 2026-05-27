"""Tests for the edge debug utility output formatting."""

from __future__ import annotations

from rag.retrieval.base import RetrievalResult


def test_format_code_search_text_includes_related_blocks():
    """Verify code search output includes related block provenance."""
    from debug_tools.edge_debug import _format_code_search_text

    result = RetrievalResult(
        content="def demo():\n    return 1",
        score=0.9876,
        source="code",
        metadata={
            "chunk_id": "repo::file.py::symbol::demo",
            "file_path": "file.py",
            "chunk_type": "function",
            "name": "demo",
            "start_line": 10,
            "end_line": 20,
            "related_blocks": [
                {
                    "target_id": "repo::file.py::symbol::helper",
                    "edge_type": "CALLS",
                    "mapping_strategy": "exact_chunk_id",
                    "source_anchor": "repo::file.py::class::Demo",
                    "direction": "outgoing",
                }
            ],
        },
    )

    text = _format_code_search_text(
        query="demo",
        repo_id="repo",
        level="symbol",
        rows=[result],
    )

    assert text.splitlines()[0] == "query=demo repo_id=repo level=symbol count=1"
    assert "repo::file.py::symbol::demo" in text
    assert "related\tCALLS\texact_chunk_id\trepo::file.py::class::Demo\toutgoing\trepo::file.py::symbol::helper" in text
