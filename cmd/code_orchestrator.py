"""CLI entrypoint for code orchestration service.

Run:
    python cmd/code_orchestrator.py --repo-path /path/to/repo --repo-id my-repo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from rag.code.orchestration_service import OrchestrationResult, CodeOrchestrationService
from utils.config import AppConfig
from utils.logger import AppLogger

log = AppLogger.get(__name__)


@dataclass
class _RunError:
    status: str
    repo_id: str
    repo_path: str
    mode: str
    error: str


def _default_repo_id(repo_path: str) -> str:
    return Path(repo_path).resolve().name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import one code repository into the code knowledge system.",
    )
    parser.add_argument("--repo-path", required=True, help="Path to repository root")
    parser.add_argument("--repo-id", default="", help="Logical repo id (default: folder name)")
    parser.add_argument(
        "--mode",
        choices=["ingest", "reindex"],
        default="ingest",
        help="Lifecycle mode for CodeKnowledgeBase",
    )
    parser.add_argument("--branch", default=None, help="Override branch name in manifest")
    parser.add_argument(
        "--include-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to update repo-level collection (code_repo)",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--code-rag-root",
        default=None,
        help="Root directory for code RAG artifacts (vector store + GraphStore)",
    )
    parser.add_argument(
        "--persist-directory",
        default=None,
        help="Backward-compatible alias for --code-rag-root",
    )
    parser.add_argument(
        "--graph-db-path",
        default=None,
        help="Override the SQLite path used for dependency edges",
    )
    parser.add_argument(
        "--force-edge-reindex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force edge updates in reindex mode even if the current commit was "
            "already reindexed"
        ),
    )
    parser.add_argument(
        "--output",
        choices=["json", "text"],
        default="json",
        help="Output format",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    config = AppConfig(path=args.config) if args.config else AppConfig()
    AppLogger.setup(
        level=config.log_level,
        fmt=config.log_format,
        datefmt=config.log_datefmt,
    )

    repo_id = args.repo_id or _default_repo_id(args.repo_path)

    service = CodeOrchestrationService(
        config=config,
        code_rag_root=args.code_rag_root or args.persist_directory,
        graph_db_path=args.graph_db_path,
    )

    try:
        result = service.run(
            repo_path=args.repo_path,
            repo_id=repo_id,
            mode=args.mode,
            branch=args.branch,
            include_repo=bool(args.include_repo),
            force_edge_reindex=bool(args.force_edge_reindex),
        )
    except Exception as e:
        err = _RunError(
            status="error",
            repo_id=repo_id,
            repo_path=str(Path(args.repo_path).resolve()),
            mode=args.mode,
            error=str(e),
        )
        print(json.dumps(asdict(err), ensure_ascii=True))
        return 1

    if args.output == "json":
        print(json.dumps(asdict(result), ensure_ascii=True))
    else:
        p = result.parse
        idx = result.index
        print(
            f"status={result.status} mode={result.mode} repo_id={result.repo_id} "
            f"operation_id={result.operation_id}"
        )
        print(f"head_commit={result.head_commit or '<none>'}")
        print(f"run_state_path={result.run_state_path}")
        print(
            "parse: "
            f"source={p.source_files_total} python={p.python_candidates} "
            f"parsed_files={p.parsed_files} chunks={p.parsed_chunks} "
            f"empty={p.empty_parsed_files} missing={p.skipped_missing_files} "
            f"non_python={p.skipped_non_python} errors={p.parse_errors}"
        )
        print(
            "index: "
            f"added={idx['added']} updated={idx['updated']} "
            f"skipped={idx['skipped']} deleted={idx['deleted']}"
        )
        edge = result.edge
        print(
            "edge: "
            f"edge_files={edge.edge_files} parsed_edges={edge.parsed_edges} "
            f"inserted={edge.inserted_edges} deleted={edge.deleted_edges} "
            f"skipped_same_commit={edge.skipped_same_commit} "
            f"errors={edge.edge_errors}"
        )
        print(f"collections={result.collections}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
