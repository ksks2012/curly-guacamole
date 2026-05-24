"""Unit tests for rag/code/orchestration_service.py."""

from __future__ import annotations

from unittest.mock import MagicMock

from rag.code.orchestration_service import CodeOrchestrationService
from rag.code.schema import CodeChunk, DependencyEdge, RepoFile, RepoManifest
from rag.indexer import IndexStats


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

    svc = CodeOrchestrationService(
        scanner=scanner,
        parser=parser,
        knowledge_base=kb,
        graph_store=graph_store,
    )
    out = svc.run(repo_path=str(tmp_path), repo_id="r1", mode="ingest")

    assert out.status == "ok"
    assert out.operation_id.startswith("r1-")
    assert out.run_state_path.endswith(".json")
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

    svc = CodeOrchestrationService(
        scanner=scanner,
        parser=parser,
        knowledge_base=kb,
        graph_store=graph_store,
    )
    out = svc.run(repo_path=str(tmp_path), repo_id="r2", mode="reindex")

    kb.reindex.assert_called_once()
    graph_store.delete_repo_edges.assert_called_once_with("r2")
    graph_store.upsert_edges.assert_called_once()
    assert out.mode == "reindex"
    assert out.index["deleted"] == 1
    assert out.edge.deleted_edges == 3
    assert out.edge.inserted_edges == 1


def test_run_partial_ok_when_vector_fails_but_edge_succeeds(tmp_path):
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    manifest = RepoManifest(
        repo_id="r4",
        repo_root=str(tmp_path),
        branch="main",
        files={
            "a.py": RepoFile("r4", "main", "a.py", "Python", 1, False, False, "h", "t"),
        },
    )

    scanner = MagicMock()
    scanner.scan.return_value = manifest

    parser = MagicMock()
    parser.parse.return_value = [_chunk("r4::a.py::function::f", "a.py")]
    parser.parse_edges.return_value = [
        _edge("e4", "r4::a.py::module::<module>", "import::os", "a.py")
    ]

    kb = MagicMock()
    kb.ingest.side_effect = RuntimeError("vector failed")
    kb.collection_stats.return_value = {"repo": 0, "file": 0, "symbol": 0, "block": 0}

    graph_store = MagicMock()
    graph_store.delete_repo_edges.return_value = 0
    graph_store.upsert_edges.return_value = 1

    svc = CodeOrchestrationService(
        scanner=scanner,
        parser=parser,
        knowledge_base=kb,
        graph_store=graph_store,
    )
    out = svc.run(repo_path=str(tmp_path), repo_id="r4", mode="ingest")

    assert out.status == "partial_ok"
    assert out.edge.inserted_edges == 1
    assert out.index["added"] == 0


def test_run_error_when_vector_and_edge_both_fail(tmp_path):
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    manifest = RepoManifest(
        repo_id="r5",
        repo_root=str(tmp_path),
        branch="main",
        files={
            "a.py": RepoFile("r5", "main", "a.py", "Python", 1, False, False, "h", "t"),
        },
    )

    scanner = MagicMock()
    scanner.scan.return_value = manifest

    parser = MagicMock()
    parser.parse.return_value = [_chunk("r5::a.py::function::f", "a.py")]
    parser.parse_edges.return_value = [
        _edge("e5", "r5::a.py::module::<module>", "import::os", "a.py")
    ]

    kb = MagicMock()
    kb.ingest.side_effect = RuntimeError("vector failed")
    kb.collection_stats.return_value = {"repo": 0, "file": 0, "symbol": 0, "block": 0}

    graph_store = MagicMock()
    graph_store.delete_repo_edges.return_value = 0
    graph_store.upsert_edges.side_effect = RuntimeError("edge failed")

    svc = CodeOrchestrationService(
        scanner=scanner,
        parser=parser,
        knowledge_base=kb,
        graph_store=graph_store,
    )
    out = svc.run(repo_path=str(tmp_path), repo_id="r5", mode="ingest")

    assert out.status == "error"
    assert out.edge.edge_errors >= 1


def test_run_invalid_mode_raises(tmp_path):
    manifest = RepoManifest(repo_id="r3", repo_root=str(tmp_path), branch="main", files={})
    scanner = MagicMock()
    scanner.scan.return_value = manifest

    kb = MagicMock()
    graph_store = MagicMock()
    svc = CodeOrchestrationService(
        scanner=scanner,
        parser=MagicMock(),
        knowledge_base=kb,
        graph_store=graph_store,
    )

    try:
        svc.run(repo_path=str(tmp_path), repo_id="r3", mode="update")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Unsupported mode" in str(e)


def test_reindex_skips_edge_update_on_same_commit_without_force(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    manifest = RepoManifest(
        repo_id="r6",
        repo_root=str(repo_root),
        branch="main",
        files={
            "a.py": RepoFile("r6", "main", "a.py", "Python", 1, False, False, "h", "t"),
        },
    )

    scanner = MagicMock()
    scanner.scan.return_value = manifest

    parser = MagicMock()
    parser.parse.return_value = [_chunk("r6::a.py::function::f", "a.py")]
    parser.parse_edges.return_value = [
        _edge("e6", "r6::a.py::module::<module>", "import::os", "a.py")
    ]

    kb = MagicMock()
    kb.reindex.return_value = IndexStats(added=1, updated=0, skipped=0, deleted=0)
    kb.collection_stats.return_value = {"repo": 1, "file": 1, "symbol": 1, "block": 1}

    graph_store = MagicMock()
    graph_store.delete_repo_edges.return_value = 2
    graph_store.upsert_edges.return_value = 1

    svc = CodeOrchestrationService(
        code_rag_root=str(tmp_path / "db"),
        scanner=scanner,
        parser=parser,
        knowledge_base=kb,
        graph_store=graph_store,
    )
    svc._git_head_commit = MagicMock(return_value="commit-abc")

    first = svc.run(repo_path=str(repo_root), repo_id="r6", mode="reindex")
    assert first.status == "ok"
    assert first.edge.skipped_same_commit == 0
    assert graph_store.delete_repo_edges.call_count == 1
    assert graph_store.upsert_edges.call_count == 1

    second = svc.run(repo_path=str(repo_root), repo_id="r6", mode="reindex")
    assert second.status == "ok"
    assert second.edge.skipped_same_commit == 1
    assert graph_store.delete_repo_edges.call_count == 1
    assert graph_store.upsert_edges.call_count == 1


def test_reindex_force_edge_update_on_same_commit(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")

    manifest = RepoManifest(
        repo_id="r7",
        repo_root=str(repo_root),
        branch="main",
        files={
            "a.py": RepoFile("r7", "main", "a.py", "Python", 1, False, False, "h", "t"),
        },
    )

    scanner = MagicMock()
    scanner.scan.return_value = manifest

    parser = MagicMock()
    parser.parse.return_value = [_chunk("r7::a.py::function::f", "a.py")]
    parser.parse_edges.return_value = [
        _edge("e7", "r7::a.py::module::<module>", "import::os", "a.py")
    ]

    kb = MagicMock()
    kb.reindex.return_value = IndexStats(added=1, updated=0, skipped=0, deleted=0)
    kb.collection_stats.return_value = {"repo": 1, "file": 1, "symbol": 1, "block": 1}

    graph_store = MagicMock()
    graph_store.delete_repo_edges.return_value = 2
    graph_store.upsert_edges.return_value = 1

    svc = CodeOrchestrationService(
        code_rag_root=str(tmp_path / "db"),
        scanner=scanner,
        parser=parser,
        knowledge_base=kb,
        graph_store=graph_store,
    )
    svc._git_head_commit = MagicMock(return_value="commit-xyz")

    first = svc.run(repo_path=str(repo_root), repo_id="r7", mode="reindex")
    assert first.edge.skipped_same_commit == 0

    second = svc.run(
        repo_path=str(repo_root),
        repo_id="r7",
        mode="reindex",
        force_edge_reindex=True,
    )
    assert second.edge.skipped_same_commit == 0
    assert graph_store.delete_repo_edges.call_count == 2
    assert graph_store.upsert_edges.call_count == 2
