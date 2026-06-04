"""Service layer for repository code ingestion orchestration."""

from __future__ import annotations

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
from rag.code.state_repository import CodeOrchestrationStateRepository
from rag.code.symbol_store import SymbolStore
from rag.embeddings import OpenRouterEmbeddings
from rag.indexer import IndexStats
from rag.retrieval.code_result_filter import CodeResultFilter
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


class CodeOrchestrationService:
    """Composition-based service for code repo ingestion lifecycle."""

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
        state_repository: CodeOrchestrationStateRepository | None = None,
        embedding_function=None,
        code_result_filter: CodeResultFilter | None = None,
    ) -> None:
        self._config = config or AppConfig()
        self._code_result_filter = code_result_filter or CodeResultFilter()
        self._scanner = scanner or RepoScanner(code_result_filter=self._code_result_filter)
        self._parser = parser or PythonASTParser()

        effective_code_root = code_rag_root or persist_directory or self._config.code_rag_root
        self._code_rag_root = str(Path(effective_code_root).resolve())
        effective_graph_db = graph_db_path or os.path.join(self._code_rag_root, "graph.db")
        self._graph_db_path = str(Path(effective_graph_db).resolve())

        if knowledge_base is not None:
            self._kb = knowledge_base
        else:
            embed = embedding_function or self._build_embedding(self._config)
            self._kb = CodeKnowledgeBase(self._code_rag_root, embed)

        self._graph = graph_store or GraphStore(self._graph_db_path)
        self._state_repo = state_repository or CodeOrchestrationStateRepository(self._code_rag_root)

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

        run_state_path = self._state_repo.write_operation_state(
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
                prev = self._state_repo.load_edge_reindex_state().get(repo_id, "")
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
                    state = self._state_repo.load_edge_reindex_state()
                    state[repo_id] = head_commit
                    self._state_repo.save_edge_reindex_state(state)
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

        self._state_repo.write_operation_state(
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
