"""Command/service orchestration for repo code ingestion.

This module wires together:
- RepoScanner (repo_path -> RepoManifest)
- PythonASTParser (Python files -> CodeChunk list)
- CodeKnowledgeBase (ingest/reindex lifecycle)

Run:
    python cmd/code_orchestrator.py --repo-path /path/to/repo --repo-id my-repo
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langchain_openai import OpenAIEmbeddings

from rag.code.ast_parser import PythonASTParser
from rag.code.graph_store import GraphStore
from rag.code.knowledge_base import CodeKnowledgeBase
from rag.code.scanner import RepoScanner
from rag.code.schema import CodeChunk, DependencyEdge, RepoManifest
from rag.code.symbol_store import SymbolStore
from rag.embeddings import OpenRouterEmbeddings
from rag.indexer import IndexStats
from utils.config import AppConfig
from utils.logger import AppLogger

log = AppLogger.get(__name__)


@dataclass
class ParseStats:
    source_files_total: int = 0
    python_candidates: int = 0
    parsed_files: int = 0
    parsed_chunks: int = 0
    empty_parsed_files: int = 0
    skipped_non_python: int = 0
    skipped_missing_files: int = 0
    parse_errors: int = 0


@dataclass
class EdgeStats:
    source_files_total: int = 0
    edge_files: int = 0
    parsed_edges: int = 0
    inserted_edges: int = 0
    deleted_edges: int = 0
    skipped_same_commit: int = 0
    edge_errors: int = 0


@dataclass
class OrchestrationResult:
    status: str
    operation_id: str
    run_state_path: str
    head_commit: str
    mode: str
    repo_id: str
    repo_path: str
    parse: ParseStats
    edge: EdgeStats
    index: dict
    collections: dict


class CodeRepoOrchestrator:
    """Composition-based orchestrator for code repo ingestion."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        persist_directory: str | None = None,
        code_rag_root: str | None = None,
        graph_db_path: str | None = None,
        scanner: RepoScanner | None = None,
        parser: PythonASTParser | None = None,
        knowledge_base: CodeKnowledgeBase | None = None,
        graph_store: GraphStore | None = None,
        embedding_function=None,
    ) -> None:
        self._config = config or AppConfig()
        self._scanner = scanner or RepoScanner()
        self._parser = parser or PythonASTParser()

        effective_code_root = (
            code_rag_root
            or persist_directory
            or self._config.code_rag_root
        )
        self._code_rag_root = str(Path(effective_code_root).resolve())
        effective_graph_db = graph_db_path or os.path.join(self._code_rag_root, "graph.db")
        self._graph_db_path = str(Path(effective_graph_db).resolve())

        if knowledge_base is not None:
            self._kb = knowledge_base
        else:
            embed = embedding_function or self._build_embedding(self._config)
            self._kb = CodeKnowledgeBase(
                self._code_rag_root,
                embed,
            )

        self._graph = graph_store or GraphStore(self._graph_db_path)

    @staticmethod
    def _build_embedding(config: AppConfig):
        if config.model_provider == "openrouter":
            return OpenRouterEmbeddings(
                model=config.embed_model,
                api_key=config.embed_api_key,
                base_url=config.embed_base,
                requests_per_minute=config.requests_rate_limit,
            )
        return OpenAIEmbeddings(
            openai_api_key=config.embed_api_key,
            openai_api_base=config.embed_base,
            model=config.embed_model,
        )

    @staticmethod
    def _stats_to_dict(stats: IndexStats) -> dict:
        return {
            "added": int(stats.added),
            "updated": int(stats.updated),
            "skipped": int(stats.skipped),
            "deleted": int(stats.deleted),
        }

    def _collect_chunks_and_edges(
        self,
        manifest: RepoManifest,
    ) -> tuple[list[CodeChunk], list[DependencyEdge], ParseStats, EdgeStats]:
        repo_root = Path(manifest.repo_root)
        chunks: list[CodeChunk] = []
        edges: list[DependencyEdge] = []
        stats = ParseStats(source_files_total=len(manifest.source_files()))
        edge_stats = EdgeStats(source_files_total=len(manifest.source_files()))

        for rf in manifest.source_files():
            if str(rf.language).lower() != "python":
                stats.skipped_non_python += 1
                continue

            path = repo_root / rf.file_path
            stats.python_candidates += 1

            if not path.exists():
                stats.skipped_missing_files += 1
                log.warning("skip missing file: repo=%s path=%s", manifest.repo_id, path)
                continue

            try:
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                stats.parse_errors += 1
                log.warning("read error: repo=%s file=%s error=%s", manifest.repo_id, rf.file_path, e)
                continue

            try:
                file_chunks = self._parser.parse(source, rf.file_path, manifest.repo_id)
            except Exception as e:
                stats.parse_errors += 1
                log.warning("parse error: repo=%s file=%s error=%s", manifest.repo_id, rf.file_path, e)
                continue

            try:
                file_edges = self._parser.parse_edges(source, rf.file_path, manifest.repo_id)
            except Exception as e:
                edge_stats.edge_errors += 1
                file_edges = []
                log.warning("edge parse error: repo=%s file=%s error=%s", manifest.repo_id, rf.file_path, e)

            if not file_chunks:
                stats.empty_parsed_files += 1
                continue

            # Filter out chunks with empty or whitespace-only content to avoid embedding API errors
            valid_chunks = [c for c in file_chunks if c.content and c.content.strip()]
            if not valid_chunks:
                stats.empty_parsed_files += 1
                continue

            stats.parsed_files += 1
            stats.parsed_chunks += len(valid_chunks)
            chunks.extend(valid_chunks)

            edge_stats.edge_files += 1
            edge_stats.parsed_edges += len(file_edges)
            edges.extend(file_edges)

        return chunks, edges, stats, edge_stats

    def _write_edges(
        self,
        *,
        repo_id: str,
        edges: list[DependencyEdge],
        mode: str,
    ) -> tuple[int, int]:
        deleted = 0
        inserted = 0
        if mode == "reindex":
            deleted = self._graph.delete_repo_edges(repo_id)
        inserted = self._graph.upsert_edges(edges)
        return int(deleted or 0), int(inserted or 0)

    @staticmethod
    def _new_operation_id(repo_id: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{repo_id}-{stamp}-{uuid4().hex[:8]}"

    def _run_state_path(self, operation_id: str) -> Path:
        return Path(self._code_rag_root) / "ops" / f"{operation_id}.json"

    def _write_run_state(self, operation_id: str, payload: dict) -> str:
        path = self._run_state_path(operation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _git_head_commit(repo_path: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", repo_path, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _edge_reindex_state_path(self) -> Path:
        return Path(self._code_rag_root) / "ops" / "edge_reindex_state.json"

    def _load_edge_reindex_state(self) -> dict[str, str]:
        path = self._edge_reindex_state_path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        state: dict[str, str] = {}
        for key, val in raw.items():
            if isinstance(key, str) and isinstance(val, str):
                state[key] = val
        return state

    def _save_edge_reindex_state(self, state: dict[str, str]) -> None:
        path = self._edge_reindex_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    def run(
        self,
        *,
        repo_path: str,
        repo_id: str,
        mode: str = "ingest",
        branch: str | None = None,
        include_repo: bool = True,
        force_edge_reindex: bool = False,
    ) -> OrchestrationResult:
        if mode not in {"ingest", "reindex"}:
            raise ValueError(f"Unsupported mode: {mode!r}")

        repo_path = str(Path(repo_path).resolve())
        operation_id = self._new_operation_id(repo_id)
        started_at = datetime.now(UTC).isoformat()
        head_commit = self._git_head_commit(repo_path)

        manifest = self._scanner.scan(repo_path=repo_path, repo_id=repo_id, branch=branch)
        chunks, edges, parse_stats, edge_stats = self._collect_chunks_and_edges(manifest)
        store = SymbolStore.from_chunks(chunks, repo_id=repo_id)

        index_stats = IndexStats()
        vector_ok = False
        edge_ok = False
        vector_error = ""
        edge_error = ""

        run_state_path = self._write_run_state(
            operation_id,
            {
                "operation_id": operation_id,
                "status": "staging",
                "phase": "staging",
                "started_at": started_at,
                "repo_id": repo_id,
                "repo_path": repo_path,
                "head_commit": head_commit,
                "mode": mode,
                "force_edge_reindex": bool(force_edge_reindex),
                "parse": asdict(parse_stats),
                "edge": asdict(edge_stats),
                "index": self._stats_to_dict(index_stats),
                "vector_ok": False,
                "edge_ok": False,
                "vector_error": "",
                "edge_error": "",
            },
        )

        try:
            source = (manifest, chunks)
            if mode == "ingest":
                index_stats = self._kb.ingest(source, store=store, include_repo=include_repo)
            else:
                index_stats = self._kb.reindex(source, store=store, include_repo=include_repo)
            vector_ok = True
        except Exception as e:
            vector_error = str(e)
            log.warning(
                "vector write failed: repo=%s mode=%s operation_id=%s error=%s",
                repo_id,
                mode,
                operation_id,
                e,
                exc_info=True,
            )

        try:
            should_skip_same_commit = False
            if mode == "reindex" and not force_edge_reindex and head_commit:
                prev = self._load_edge_reindex_state().get(repo_id, "")
                should_skip_same_commit = prev == head_commit

            if should_skip_same_commit:
                edge_stats.skipped_same_commit = 1
                edge_ok = True
                log.info(
                    "edge reindex skipped (same commit): repo=%s commit=%s operation_id=%s",
                    repo_id,
                    head_commit,
                    operation_id,
                )
            else:
                deleted_edges, inserted_edges = self._write_edges(
                    repo_id=repo_id,
                    edges=edges,
                    mode=mode,
                )
                edge_stats.deleted_edges = deleted_edges
                edge_stats.inserted_edges = inserted_edges
                edge_ok = edge_stats.edge_errors == 0

                if mode == "reindex" and edge_ok and head_commit:
                    state = self._load_edge_reindex_state()
                    state[repo_id] = head_commit
                    self._save_edge_reindex_state(state)
        except Exception as e:
            edge_stats.edge_errors += 1
            edge_error = str(e)
            log.warning(
                "edge write failed: repo=%s mode=%s operation_id=%s error=%s",
                repo_id,
                mode,
                operation_id,
                e,
                exc_info=True,
            )

        if vector_ok and edge_ok:
            status = "ok"
        elif vector_ok or edge_ok:
            status = "partial_ok"
        else:
            status = "error"

        self._write_run_state(
            operation_id,
            {
                "operation_id": operation_id,
                "status": status,
                "phase": "completed",
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "repo_id": repo_id,
                "repo_path": repo_path,
                "head_commit": head_commit,
                "mode": mode,
                "force_edge_reindex": bool(force_edge_reindex),
                "parse": asdict(parse_stats),
                "edge": asdict(edge_stats),
                "index": self._stats_to_dict(index_stats),
                "vector_ok": vector_ok,
                "edge_ok": edge_ok,
                "vector_error": vector_error,
                "edge_error": edge_error,
            },
        )

        return OrchestrationResult(
            status=status,
            operation_id=operation_id,
            run_state_path=run_state_path,
            head_commit=head_commit,
            mode=mode,
            repo_id=repo_id,
            repo_path=repo_path,
            parse=parse_stats,
            edge=edge_stats,
            index=self._stats_to_dict(index_stats),
            collections=self._kb.collection_stats(),
        )


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

    orchestrator = CodeRepoOrchestrator(
        config=config,
        code_rag_root=args.code_rag_root or args.persist_directory,
        graph_db_path=args.graph_db_path,
    )

    try:
        result = orchestrator.run(
            repo_path=args.repo_path,
            repo_id=repo_id,
            mode=args.mode,
            branch=args.branch,
            include_repo=bool(args.include_repo),
            force_edge_reindex=bool(args.force_edge_reindex),
        )
    except Exception as e:
        err = {
            "status": "error",
            "repo_id": repo_id,
            "repo_path": str(Path(args.repo_path).resolve()),
            "mode": args.mode,
            "error": str(e),
        }
        print(json.dumps(err, ensure_ascii=True))
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
