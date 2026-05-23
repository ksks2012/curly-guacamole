"""
CodeKnowledgeBase — unified lifecycle orchestrator for multi-repo code indexing.

CodeKnowledgeBase composes:
  - RepoIndex   (repo-level summary collection: code_repo)
  - CodeIndexer (file/symbol/block collections)

It provides a single lifecycle entry-point for adding, updating, deleting,
and reindexing repository code data across both indexers.
"""

from __future__ import annotations

from rag.code.indexer import CodeIndexer
from rag.code.repo_index import RepoIndex
from rag.code.schema import CodeChunk, RepoManifest
from rag.code.symbol_store import SymbolStore
from rag.indexer import IndexStats


class CodeKnowledgeBase:
    """Unified lifecycle manager for repository code knowledge.

    Parameters
    ----------
    persist_directory       : Chroma storage directory path.
    embedding_function      : Any LangChain ``Embeddings`` instance.
    code_collection_prefix  : Prefix fallback for CodeIndexer collections.
    code_collection_names   : Optional per-level collection name overrides
                              for CodeIndexer.
    repo_collection_name    : Optional collection name override for RepoIndex.
    """

    def __init__(
        self,
        persist_directory: str,
        embedding_function,
        *,
        code_collection_prefix: str = "code",
        code_collection_names: dict[str, str] | None = None,
        repo_collection_name: str | None = None,
    ) -> None:
        self._repo_index = RepoIndex(
            persist_directory,
            embedding_function,
            collection_name=repo_collection_name,
        )
        self._code_indexer = CodeIndexer(
            persist_directory,
            embedding_function,
            collection_prefix=code_collection_prefix,
            collection_names=code_collection_names,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sum(*stats: IndexStats) -> IndexStats:
        total = IndexStats()
        for s in stats:
            total = total + s
        return total

    # ------------------------------------------------------------------
    # Single-repo lifecycle
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: tuple[RepoManifest, list[CodeChunk]],
        *,
        store: SymbolStore | None = None,
        include_repo: bool = True,
    ) -> IndexStats:
        """Ingest one repository into RepoIndex + CodeIndexer."""
        manifest, chunks = source
        code_stats = self._code_indexer.ingest((manifest, chunks), store=store)
        if not include_repo:
            return code_stats
        repo_stats = self._repo_index.ingest(manifest, chunks=chunks)
        return self._sum(repo_stats, code_stats)

    def add_repo(
        self,
        source: tuple[RepoManifest, list[CodeChunk]],
        *,
        store: SymbolStore | None = None,
        include_repo: bool = True,
    ) -> IndexStats:
        """Alias for ``ingest()``."""
        return self.ingest(source, store=store, include_repo=include_repo)

    def update(
        self,
        source: tuple[RepoManifest, list[CodeChunk]],
        *,
        store: SymbolStore | None = None,
        include_repo: bool = True,
    ) -> IndexStats:
        """Incrementally update one repository across both indexers."""
        manifest, chunks = source
        code_stats = self._code_indexer.update((manifest, chunks), store=store)
        if not include_repo:
            return code_stats
        repo_stats = self._repo_index.update(manifest, chunks=chunks)
        return self._sum(repo_stats, code_stats)

    def reindex(
        self,
        source: tuple[RepoManifest, list[CodeChunk]],
        *,
        store: SymbolStore | None = None,
        include_repo: bool = True,
    ) -> IndexStats:
        """Full rebuild for one repository across both indexers."""
        manifest, chunks = source
        code_stats = self._code_indexer.reindex((manifest, chunks), store=store)
        if not include_repo:
            return code_stats
        repo_stats = self._repo_index.reindex(manifest, chunks=chunks)
        return self._sum(repo_stats, code_stats)

    def delete(self, repo_id: str, *, include_repo: bool = True) -> IndexStats:
        """Delete one repository from both indexers."""
        code_stats = self._code_indexer.delete(repo_id)
        if not include_repo:
            return code_stats
        repo_stats = self._repo_index.delete(repo_id)
        return self._sum(repo_stats, code_stats)

    def remove_repo(self, repo_id: str, *, include_repo: bool = True) -> IndexStats:
        """Alias for ``delete()``."""
        return self.delete(repo_id, include_repo=include_repo)

    # ------------------------------------------------------------------
    # Multi-repo lifecycle
    # ------------------------------------------------------------------

    def ingest_many(
        self,
        sources: list[tuple[RepoManifest, list[CodeChunk]]],
        *,
        store_by_repo: dict[str, SymbolStore] | None = None,
        include_repo: bool = True,
    ) -> IndexStats:
        """Ingest many repositories and return aggregated stats."""
        total = IndexStats()
        for manifest, chunks in sources:
            store = (store_by_repo or {}).get(manifest.repo_id)
            total = total + self.ingest(
                (manifest, chunks),
                store=store,
                include_repo=include_repo,
            )
        return total

    def update_many(
        self,
        sources: list[tuple[RepoManifest, list[CodeChunk]]],
        *,
        store_by_repo: dict[str, SymbolStore] | None = None,
        include_repo: bool = True,
    ) -> IndexStats:
        """Update many repositories and return aggregated stats."""
        total = IndexStats()
        for manifest, chunks in sources:
            store = (store_by_repo or {}).get(manifest.repo_id)
            total = total + self.update(
                (manifest, chunks),
                store=store,
                include_repo=include_repo,
            )
        return total

    def reindex_many(
        self,
        sources: list[tuple[RepoManifest, list[CodeChunk]]],
        *,
        store_by_repo: dict[str, SymbolStore] | None = None,
        include_repo: bool = True,
    ) -> IndexStats:
        """Reindex many repositories and return aggregated stats."""
        total = IndexStats()
        for manifest, chunks in sources:
            store = (store_by_repo or {}).get(manifest.repo_id)
            total = total + self.reindex(
                (manifest, chunks),
                store=store,
                include_repo=include_repo,
            )
        return total

    def delete_many(self, repo_ids: list[str], *, include_repo: bool = True) -> IndexStats:
        """Delete many repositories and return aggregated stats."""
        total = IndexStats()
        for repo_id in repo_ids:
            total = total + self.delete(repo_id, include_repo=include_repo)
        return total

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def collection_stats(self) -> dict[str, int]:
        """Return merged collection stats for repo + code levels."""
        return {
            "repo": self._repo_index.collection_stats(),
            **self._code_indexer.collection_stats(),
        }

    @property
    def repo_index(self) -> RepoIndex:
        """Expose RepoIndex for advanced operations."""
        return self._repo_index

    @property
    def code_indexer(self) -> CodeIndexer:
        """Expose CodeIndexer for advanced operations."""
        return self._code_indexer
