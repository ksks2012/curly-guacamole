"""Smoke tests for cmd/code_orchestrator.py CLI wiring."""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

from rag.code.orchestration_service import EdgeStats, OrchestrationResult, ParseStats


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent.parent
    mod_path = repo_root / "cmd" / "code_orchestrator.py"
    spec = importlib.util.spec_from_file_location("code_orchestrator", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_force_flag(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "code_orchestrator.py",
            "--repo-path",
            "/tmp/repo",
            "--mode",
            "reindex",
            "--force-edge-reindex",
        ],
    )
    args = mod._parse_args()
    assert args.repo_path == "/tmp/repo"
    assert args.mode == "reindex"
    assert args.force_edge_reindex is True


def test_main_success_json_output(monkeypatch, capsys):
    mod = _load_module()

    class _FakeConfig:
        log_level = "INFO"
        log_format = "%(message)s"
        log_datefmt = "%Y-%m-%d %H:%M:%S"

    parse_stats = ParseStats()
    edge_stats = EdgeStats()
    fake_result = OrchestrationResult(
        status="ok",
        operation_id="op-1",
        run_state_path="/tmp/op-1.json",
        head_commit="abc123",
        mode="ingest",
        repo_id="r1",
        repo_path="/tmp/repo",
        parse=parse_stats,
        edge=edge_stats,
        index={"added": 1, "updated": 0, "skipped": 0, "deleted": 0},
        collections={"repo": 1},
    )

    class _FakeService:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            return fake_result

    monkeypatch.setattr(mod, "AppConfig", lambda path=None: _FakeConfig())
    monkeypatch.setattr(mod.AppLogger, "setup", lambda **kwargs: None)
    monkeypatch.setattr(mod, "CodeOrchestrationService", _FakeService)
    monkeypatch.setattr(
        mod,
        "_parse_args",
        lambda: Namespace(
            repo_path="/tmp/repo",
            repo_id="r1",
            mode="ingest",
            branch=None,
            include_repo=True,
            config=None,
            code_rag_root=None,
            persist_directory=None,
            graph_db_path=None,
            force_edge_reindex=False,
            output="json",
        ),
    )

    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert '"status": "ok"' in out


def test_main_error_returns_nonzero(monkeypatch, capsys):
    mod = _load_module()

    class _FakeConfig:
        log_level = "INFO"
        log_format = "%(message)s"
        log_datefmt = "%Y-%m-%d %H:%M:%S"

    class _FakeService:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(mod, "AppConfig", lambda path=None: _FakeConfig())
    monkeypatch.setattr(mod.AppLogger, "setup", lambda **kwargs: None)
    monkeypatch.setattr(mod, "CodeOrchestrationService", _FakeService)
    monkeypatch.setattr(
        mod,
        "_parse_args",
        lambda: Namespace(
            repo_path="/tmp/repo",
            repo_id="r1",
            mode="ingest",
            branch=None,
            include_repo=True,
            config=None,
            code_rag_root=None,
            persist_directory=None,
            graph_db_path=None,
            force_edge_reindex=False,
            output="json",
        ),
    )

    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert '"status": "error"' in out
