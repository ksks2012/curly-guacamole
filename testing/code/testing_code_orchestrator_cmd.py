"""Unit tests for cmd/code_orchestrator.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

from rag.code.schema import CodeChunk, DependencyEdge, RepoFile, RepoManifest
from rag.indexer import IndexStats


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent.parent
    mod_path = repo_root / "cmd" / "code_orchestrator.py"
    spec = importlib.util.spec_from_file_location("code_orchestrator", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _chunk(chunk_id: str, file_path: str) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        content="def f():\n    return 1\n",
        repo_id="r1",
        file_path=file_path,
        language="python",
        chunk_type="function",
        name="f",
        start_line=1,
        end_line=2,
        content_hash="h1",
    )


def _edge(edge_id: str, src_id: str, dst_id: str, file_path: str) -> DependencyEdge:
    return DependencyEdge(
        edge_id=edge_id,
        src_id=src_id,
        dst_id=dst_id,
        edge_type="IMPORTS",
        repo_id="r1",
        file_path=file_path,
        line_no=1,
    )


def test_run_ingest_aggregates_parse_stats_and_calls_kb(tmp_path):
    mod = _load_module()

    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "d.py").write_text("def d():\n    return 1\n", encoding="utf-8")

    manifest = RepoManifest(
        repo_id="r1",
        repo_root=str(tmp_path),
        branch="main",
        files={
            "a.py": RepoFile("r1", "main", "a.py", "Python", 1, False, False, "h", "t"),
            "c.py": RepoFile("r1", "main", "c.py", "Python", 1, False, False, "h", "t"),
            "d.py": RepoFile("r1", "main", "d.py", "Python", 1, False, False, "h", "t"),
            "notes.md": RepoFile("r1", "main", "notes.md", "Markdown", 1, False, False, "h", "t"),
        },
    )

    scanner = MagicMock()
    scanner.scan.return_value = manifest

    parser = MagicMock()

    def _parse_side_effect(source, file_path, repo_id):
        if file_path == "a.py":
            return [_chunk("r1::a.py::function::f", "a.py")]
        if file_path == "d.py":
            raise ValueError("parse failed")
        return []

    parser.parse.side_effect = _parse_side_effect
    parser.parse_edges.side_effect = lambda source, file_path, repo_id: [
        _edge("e1", "r1::a.py::module::<module>", "import::os", "a.py")
    ] if file_path == "a.py" else []

    kb = MagicMock()
    kb.ingest.return_value = IndexStats(added=2, updated=1, skipped=3, deleted=0)
    kb.collection_stats.return_value = {"repo": 1, "file": 2, "symbol": 3, "block": 4}

    graph_store = MagicMock()
    graph_store.upsert_edges.return_value = 1
    graph_store.delete_repo_edges.return_value = 0

    orch = mod.CodeRepoOrchestrator(
        scanner=scanner,
        parser=parser,
        knowledge_base=kb,
        graph_store=graph_store,
    )
    out = orch.run(repo_path=str(tmp_path), repo_id="r1", mode="ingest")

    assert out.status == "ok"
    assert out.mode == "ingest"
    assert out.parse.source_files_total == 4
    assert out.parse.python_candidates == 3
    assert out.parse.parsed_files == 1
    assert out.parse.parsed_chunks == 1
    assert out.parse.skipped_non_python == 1
    assert out.parse.skipped_missing_files == 1
    assert out.parse.parse_errors == 1
    assert out.edge.parsed_edges == 1
    assert out.edge.inserted_edges == 1
    assert out.edge.deleted_edges == 0
    assert out.edge.edge_errors == 0

    kb.ingest.assert_called_once()
    graph_store.upsert_edges.assert_called_once()
    assert out.index["added"] == 2
    assert out.collections["block"] == 4


def test_run_reindex_calls_reindex(tmp_path):
    mod = _load_module()

    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    manifest = RepoManifest(
        repo_id="r2",
        repo_root=str(tmp_path),
        branch="main",
        files={
            "a.py": RepoFile("r2", "main", "a.py", "Python", 1, False, False, "h", "t"),
        },
    )

    scanner = MagicMock()
    scanner.scan.return_value = manifest

    parser = MagicMock()
    parser.parse.return_value = [_chunk("r2::a.py::function::f", "a.py")]
    parser.parse_edges.return_value = [
        _edge("e2", "r2::a.py::module::<module>", "import::os", "a.py")
    ]

    kb = MagicMock()
    kb.reindex.return_value = IndexStats(added=1, updated=0, skipped=0, deleted=1)
    kb.collection_stats.return_value = {"repo": 1, "file": 1, "symbol": 1, "block": 1}

    graph_store = MagicMock()
    graph_store.delete_repo_edges.return_value = 3
    graph_store.upsert_edges.return_value = 1

    orch = mod.CodeRepoOrchestrator(
        scanner=scanner,
        parser=parser,
        knowledge_base=kb,
        graph_store=graph_store,
    )
    out = orch.run(repo_path=str(tmp_path), repo_id="r2", mode="reindex")

    kb.reindex.assert_called_once()
    graph_store.delete_repo_edges.assert_called_once_with("r2")
    graph_store.upsert_edges.assert_called_once()
    assert out.mode == "reindex"
    assert out.index["deleted"] == 1
    assert out.edge.deleted_edges == 3
    assert out.edge.inserted_edges == 1


def test_run_invalid_mode_raises(tmp_path):
    mod = _load_module()

    manifest = RepoManifest(repo_id="r3", repo_root=str(tmp_path), branch="main", files={})
    scanner = MagicMock()
    scanner.scan.return_value = manifest

    kb = MagicMock()
    graph_store = MagicMock()
    orch = mod.CodeRepoOrchestrator(
        scanner=scanner,
        parser=MagicMock(),
        knowledge_base=kb,
        graph_store=graph_store,
    )

    try:
        orch.run(repo_path=str(tmp_path), repo_id="r3", mode="update")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Unsupported mode" in str(e)
