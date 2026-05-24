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
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

from rag.code.ast_parser import PythonASTParser
from rag.code.knowledge_base import CodeKnowledgeBase
from rag.code.scanner import RepoScanner
from rag.code.schema import CodeChunk, RepoManifest
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
class OrchestrationResult:
    status: str
    mode: str
    repo_id: str
    repo_path: str
    parse: ParseStats
    index: dict
    collections: dict


class CodeRepoOrchestrator:
    """Composition-based orchestrator for code repo ingestion."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        persist_directory: str | None = None,
        scanner: RepoScanner | None = None,
        parser: PythonASTParser | None = None,
        knowledge_base: CodeKnowledgeBase | None = None,
        embedding_function=None,
    ) -> None:
        self._config = config or AppConfig()
        self._scanner = scanner or RepoScanner()
        self._parser = parser or PythonASTParser()

        if knowledge_base is not None:
            self._kb = knowledge_base
        else:
            embed = embedding_function or self._build_embedding(self._config)
            self._kb = CodeKnowledgeBase(
                persist_directory or self._config.persist_directory,
                embed,
            )

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

    def _collect_chunks(self, manifest: RepoManifest) -> tuple[list[CodeChunk], ParseStats]:
        repo_root = Path(manifest.repo_root)
        chunks: list[CodeChunk] = []
        stats = ParseStats(source_files_total=len(manifest.source_files()))

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
                file_chunks = self._parser.parse_file(
                    path,
                    repo_root=repo_root,
                    repo_id=manifest.repo_id,
                )
            except Exception as e:
                stats.parse_errors += 1
                log.warning("parse error: repo=%s file=%s error=%s", manifest.repo_id, rf.file_path, e)
                continue

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

        return chunks, stats

    def run(
        self,
        *,
        repo_path: str,
        repo_id: str,
        mode: str = "ingest",
        branch: str | None = None,
        include_repo: bool = True,
    ) -> OrchestrationResult:
        repo_path = str(Path(repo_path).resolve())
        manifest = self._scanner.scan(repo_path=repo_path, repo_id=repo_id, branch=branch)
        chunks, parse_stats = self._collect_chunks(manifest)
        store = SymbolStore.from_chunks(chunks, repo_id=repo_id)

        source = (manifest, chunks)
        if mode == "ingest":
            index_stats = self._kb.ingest(source, store=store, include_repo=include_repo)
        elif mode == "reindex":
            index_stats = self._kb.reindex(source, store=store, include_repo=include_repo)
        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

        return OrchestrationResult(
            status="ok",
            mode=mode,
            repo_id=repo_id,
            repo_path=repo_path,
            parse=parse_stats,
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
        "--persist-directory",
        default=None,
        help="Override persist_directory from config",
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
        persist_directory=args.persist_directory,
    )

    try:
        result = orchestrator.run(
            repo_path=args.repo_path,
            repo_id=repo_id,
            mode=args.mode,
            branch=args.branch,
            include_repo=bool(args.include_repo),
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
        print(f"status={result.status} mode={result.mode} repo_id={result.repo_id}")
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
        print(f"collections={result.collections}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
