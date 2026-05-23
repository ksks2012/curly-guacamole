"""
GCR1.4 — Multi-resolution Code Indexing.

Three Chroma collections managed by CodeIndexer:

  documents  — one document per source file (module-level summary).
               Shared with the document ingestion pipeline.
  symbols    — one document per symbol (class / function / method).
               Primary retrieval collection.
  code_block — one document per code chunk (full code text).
               Fine-grained reasoning collection.

Repo-level indexing (one document per repository) is handled by
``RepoIndex`` in ``rag.code.repo_index``.  ``CodeIndexer.index_manifest()``
remains as a backward-compatibility shim that delegates to ``RepoIndex``.

Collection naming
-----------------
Each collection name defaults to the mapping in ``_DEFAULT_COLLECTION_NAMES``
and can be overridden per-level via the ``collection_names`` constructor
argument.  Unknown levels fall back to ``"{prefix}_{level}"``.

Incremental indexing
---------------------
Every document stores ``content_hash`` in its metadata.  On each indexing
run the store compares incoming hashes against stored hashes:

  - hash unchanged → skipped (no embedding re-generated)
  - hash changed   → updated via ``update_documents``
  - new ID         → added via ``add_documents``
  - ID absent from new scan but present in store → pruned (per-repo scoped)

Pruning is scoped to the current ``repo_id`` so that documents from other
repositories in the same collection are never affected.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.logger import AppLogger
from rag.code.schema import CodeChunk, RepoManifest, Symbol
from rag.code.symbol_store import SymbolStore
from rag.indexer import BaseIndexer, ChangeSet, IndexStats, diff_by_content_hash

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# Text builders — what gets embedded at each resolution level
# ---------------------------------------------------------------------------

def _file_text(
    file_path: str,
    language: str,
    module_docstring: str | None,
    symbol_lines: list[str],
) -> str:
    """Per-file summary combining module docstring and declared symbols.

    Useful for answering "which file handles X?" queries.
    """
    parts: list[str] = [f"file: {file_path}  language: {language}"]
    if module_docstring:
        parts.append(module_docstring.strip())
    if symbol_lines:
        parts.append("\n".join(symbol_lines))
    return "\n".join(parts)


def _symbol_text(sym: Symbol, docstring: str | None) -> str:
    """Conceptual description of a symbol: type + qualified name + docstring.

    Embedded text for the primary retrieval collection.  Intentionally
    omits implementation details so the embedding captures the *meaning*
    of the symbol, not its syntax.
    """
    lines = [
        f"{sym.symbol_type} {sym.symbol_name}",
        f"file: {sym.file_path}  lines {sym.start_line}–{sym.end_line}",
    ]
    if docstring:
        lines.append(docstring.strip())
    return "\n".join(lines)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# CodeIndexer
# ---------------------------------------------------------------------------

class CodeIndexer(BaseIndexer):
    """Multi-resolution code indexer (GCR1.4 / Phase 2 Step 2.3a).

    Manages three Chroma collections at different granularities:

      documents — one document per source file (module-level summary).
                  Shared with the document ingestion pipeline; use
                  ``source_type`` metadata to distinguish origins.
      symbols   — one document per symbol (class / function / method).
                  Primary retrieval collection.
      code_block — one document per code chunk (full code text).
                   Fine-grained reasoning collection.

    The legacy ``code_repo`` collection (repo-level overview) is no longer
    written by default.  Repo-level metadata (repo_root, branch, scanned_at)
    is folded into every file-level document so it remains queryable without
    a dedicated collection.

    Implements ``BaseIndexer`` so document and code indexing participate in
    the same lifecycle.  ``source`` in all BaseIndexer methods is a
    ``(RepoManifest, list[CodeChunk])`` tuple; ``source_id`` is ``repo_id``.

    Parameters
    ----------
    persist_directory  : Chroma storage directory path.
    embedding_function : Any LangChain ``Embeddings`` instance.
    collection_prefix  : Legacy prefix used for the block collection name
                         (``"{prefix}_block"``) and as a fallback for any
                         level not listed in *collection_names*.
                         Default ``"code"`` → ``code_block``.
    collection_names   : Optional per-level name overrides.  Default mapping:
                         ``{"file": "documents", "symbol": "symbols",
                           "block": "code_block"}``.
                         Pass ``{"file": "rag_collection"}`` to share the
                         document collection used by the document pipeline.
    """

    LEVELS = ("file", "symbol", "block")

    # Default collection names after Step 2.3a consolidation.
    # Override per-level via the collection_names constructor argument.
    _DEFAULT_COLLECTION_NAMES: dict[str, str] = {
        "file":   "documents",   # shared with document pipeline
        "symbol": "symbols",     # consolidated from code_symbol
        "block":  "code_block",  # unchanged — fine-grained code text
    }

    def __init__(
        self,
        persist_directory: str,
        embedding_function,
        collection_prefix: str = "code",
        collection_names: dict[str, str] | None = None,
    ) -> None:
        self._persist_dir = persist_directory
        self._embed = embedding_function
        self._prefix = collection_prefix
        self._dbs: dict[str, Chroma] = {}
        # Merge caller overrides on top of class-level defaults.
        self._collection_names: dict[str, str] = {**self._DEFAULT_COLLECTION_NAMES}
        if collection_names:
            self._collection_names.update(collection_names)

    # ── Collection access ─────────────────────────────────────────────────

    def collection_name(self, level: str) -> str:
        """Return the Chroma collection name for *level*.

        Checks the per-instance ``_collection_names`` mapping first; falls
        back to ``"{prefix}_{level}"`` for any level not listed there.
        """
        return self._collection_names.get(level) or f"{self._prefix}_{level}"

    def _db(self, level: str) -> Chroma:
        if level not in self._dbs:
            self._dbs[level] = Chroma(
                persist_directory=self._persist_dir,
                embedding_function=self._embed,
                collection_name=self.collection_name(level),
            )
        return self._dbs[level]

    # ── Incremental upsert ────────────────────────────────────────────────

    def _upsert(
        self,
        level: str,
        docs: list[Document],
        ids: list[str],
        *,
        prune_missing: bool = True,
        repo_id: str = "",
    ) -> dict:
        """Incrementally upsert documents into a collection.

        Compares incoming ``content_hash`` values against stored metadata.
        Only changed or new documents trigger embedding generation.
        Pruning removes documents belonging to the same ``repo_id`` that are
        absent from the incoming set, keeping cross-repo documents intact.

        Returns
        -------
        dict with keys: added, updated, skipped, deleted.
        """
        stats = {"added": 0, "updated": 0, "skipped": 0, "deleted": 0}
        if not docs:
            return stats

        db = self._db(level)

        # Fetch existing IDs and content_hash metadata (no embeddings — cheap).
        try:
            if repo_id:
                existing_raw = db.get(
                    where={"repo_id": repo_id},
                    include=["metadatas"],
                )
            else:
                existing_raw = db.get(include=["metadatas"])
        except Exception:
            existing_raw = {"ids": [], "metadatas": []}

        existing: dict[str, str] = {}  # id → stored content_hash
        for eid, emeta in zip(
            existing_raw.get("ids", []),
            existing_raw.get("metadatas", []) or [],
        ):
            existing[eid] = (emeta or {}).get("content_hash", "")

        # Classify changes via the shared content-hash diff utility.
        doc_by_id = dict(zip(ids, docs))
        incoming = {
            doc_id: doc.metadata.get("content_hash", "")
            for doc_id, doc in doc_by_id.items()
        }
        changeset = diff_by_content_hash(existing, incoming)

        if changeset.added:
            add_docs = [doc_by_id[i] for i in changeset.added]
            db.add_documents(add_docs, ids=changeset.added)
            stats["added"] += len(changeset.added)
            log.debug("CodeIndexer[%s] +%d added", level, len(changeset.added))

        if changeset.modified:
            mod_docs = [doc_by_id[i] for i in changeset.modified]
            db.update_documents(changeset.modified, mod_docs)
            stats["updated"] += len(changeset.modified)
            log.debug("CodeIndexer[%s] ~%d updated", level, len(changeset.modified))

        stats["skipped"] += len(changeset.skipped)

        if prune_missing and changeset.deleted:
            db.delete(changeset.deleted)
            stats["deleted"] += len(changeset.deleted)
            log.debug("CodeIndexer[%s] -%d deleted", level, len(changeset.deleted))

        return stats

    # ── Level: repo ───────────────────────────────────────────────────────

    def index_manifest(
        self,
        manifest: RepoManifest,
        chunks: list[CodeChunk] | None = None,
    ) -> dict:
        """Backward-compatibility shim — delegates to RepoIndex.index_manifest().

        Direct use of ``RepoIndex`` is preferred for new code.

        Parameters
        ----------
        manifest : RepoManifest from RepoScanner.
        chunks   : Optional CodeChunk list; used to extract module docstrings.
        """
        from rag.code.repo_index import RepoIndex  # lazy import avoids circularity
        ri = RepoIndex(self._persist_dir, self._embed)
        return ri.index_manifest(manifest, chunks)

    # ── Level: file ───────────────────────────────────────────────────────

    def index_files(
        self,
        chunks: list[CodeChunk],
        manifest: RepoManifest | None = None,
    ) -> dict:
        """Index one document per source file.

        The embedded text combines the module docstring with a structured
        listing of all symbols declared in the file.  Suitable for queries
        like "which file implements the retry logic?".

        Parameters
        ----------
        chunks   : All CodeChunk objects for the repository.
        manifest : Optional RepoManifest used to enrich metadata
                   (language, is_test, is_generated).
        """
        by_file: dict[str, list[CodeChunk]] = {}
        for c in chunks:
            by_file.setdefault(c.file_path, []).append(c)

        docs: list[Document] = []
        ids:  list[str]      = []
        repo_id = ""

        for file_path, file_chunks in by_file.items():
            module_chunk = next((c for c in file_chunks if c.chunk_type == "module"), None)
            if module_chunk is None:
                continue
            repo_id = repo_id or module_chunk.repo_id

            # One line per declared symbol
            sym_lines: list[str] = []
            for c in sorted(file_chunks, key=lambda x: x.start_line):
                if c.chunk_type == "module":
                    continue
                label = f"  {c.chunk_type} {c.name}"
                if c.docstring:
                    label += f":  {c.docstring.split(chr(10))[0].strip()}"
                sym_lines.append(label)

            language = ""
            is_test = False
            is_generated = False
            if manifest and file_path in manifest.files:
                rf = manifest.files[file_path]
                language = rf.language
                is_test = rf.is_test
                is_generated = rf.is_generated

            text = _file_text(file_path, language, module_chunk.docstring, sym_lines)

            # Include repo-level fields in every file document so callers
            # can filter or display repo context without a separate collection.
            repo_root   = manifest.repo_root   if manifest else ""
            branch      = manifest.branch      if manifest else ""
            scanned_at  = manifest.scanned_at  if manifest else ""

            doc = Document(
                page_content=text,
                metadata={
                    "source_type":  "code",
                    "repo_id":      module_chunk.repo_id,
                    "repo_root":    repo_root,
                    "branch":       branch,
                    "scanned_at":   scanned_at,
                    "file_path":    file_path,
                    "language":     language,
                    "is_test":      is_test,
                    "is_generated": is_generated,
                    "content_hash": module_chunk.content_hash,
                },
            )
            docs.append(doc)
            ids.append(f"{module_chunk.repo_id}::{file_path}::file")

        return self._upsert("file", docs, ids, repo_id=repo_id)

    # ── Level: symbol ─────────────────────────────────────────────────────

    def index_symbols(
        self,
        symbols: Iterable[Symbol],
        chunks: list[CodeChunk] | None = None,
    ) -> dict:
        """Index one document per symbol — the primary retrieval collection.

        Embedded text = symbol type + qualified name + docstring.
        Module-level symbols are skipped (covered by the file collection).

        Parameters
        ----------
        symbols : Iterable of Symbol objects (from SymbolStore).
        chunks  : Optional CodeChunk list used to enrich docs with docstrings.
        """
        # Docstring lookup: (file_path, symbol_name) → docstring
        chunk_docstrings: dict[tuple[str, str], str | None] = {}
        if chunks:
            for c in chunks:
                chunk_docstrings[(c.file_path, c.name)] = c.docstring

        docs: list[Document] = []
        ids:  list[str]      = []
        repo_id = ""

        for sym in symbols:
            if sym.symbol_type == "module":
                continue  # covered by file collection
            repo_id = repo_id or sym.repo_id
            docstring = chunk_docstrings.get((sym.file_path, sym.symbol_name))
            text = _symbol_text(sym, docstring)
            doc = Document(
                page_content=text,
                metadata={
                    **sym.to_dict(),
                    "content_hash": _sha256(text),
                },
            )
            docs.append(doc)
            ids.append(sym.symbol_id)

        return self._upsert("symbol", docs, ids, repo_id=repo_id)

    # ── Level: block ──────────────────────────────────────────────────────

    def index_blocks(self, chunks: list[CodeChunk]) -> dict:
        """Index one document per code chunk (full source text).

        The block collection stores complete code for fine-grained retrieval:
        fetching the exact implementation of a function or method body.

        Parameters
        ----------
        chunks : All CodeChunk objects (including module / class / function /
                 method chunks) for the repository.
        """
        docs: list[Document] = []
        ids:  list[str]      = []
        repo_id = chunks[0].repo_id if chunks else ""

        for chunk in chunks:
            doc = Document(
                page_content=chunk.code,
                metadata=chunk.to_meta(),
            )
            docs.append(doc)
            ids.append(chunk.chunk_id)

        return self._upsert("block", docs, ids, repo_id=repo_id)

    # ── Convenience: all 4 levels ─────────────────────────────────────────

    def index_all(
        self,
        manifest: RepoManifest,
        chunks: list[CodeChunk],
        store: SymbolStore | None = None,
    ) -> dict[str, dict]:
        """Index file, symbol, and block collections in a single call.

        The legacy ``code_repo`` collection is no longer written here;
        repository-level metadata is now embedded in every file document.
        Call ``index_manifest()`` directly if you still need the repo
        collection for other purposes.

        Parameters
        ----------
        manifest : RepoManifest produced by RepoScanner.
        chunks   : All CodeChunk objects from PythonASTParser.
        store    : Optional pre-built SymbolStore.  Built from *chunks* if None.

        Returns
        -------
        dict mapping level name → per-level stats dict
        (keys: added, updated, skipped, deleted).
        """
        if store is None:
            store = SymbolStore.from_chunks(chunks)

        return {
            "file":   self.index_files(chunks, manifest),
            "symbol": self.index_symbols(store, chunks),
            "block":  self.index_blocks(chunks),
        }

    # ── Deletion ──────────────────────────────────────────────────────────

    def delete_repo(self, repo_id: str) -> IndexStats:
        """Remove all documents for *repo_id* from every collection."""
        total_deleted = 0
        for level in self.LEVELS:
            try:
                db = self._db(level)
                result = db.get(where={"repo_id": repo_id}, include=[])
                ids = result.get("ids", [])
                if ids:
                    db.delete(ids)
                    total_deleted += len(ids)
                    log.info(
                        "CodeIndexer.delete_repo: removed %d docs from [%s]",
                        len(ids), level,
                    )
            except Exception as e:
                log.warning("CodeIndexer.delete_repo[%s]: %s", level, e)
        return IndexStats(deleted=total_deleted)

    # ── Query helpers ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        level: str = "symbol",
        k: int = 5,
        filter: dict | None = None,
    ) -> list[Document]:
        """Similarity search against one collection.

        Parameters
        ----------
        query  : Natural language query string.
        level  : Collection to search: ``"repo"``, ``"file"``,
                 ``"symbol"`` (default), or ``"block"``.
        k      : Number of results to return.
        filter : Optional Chroma metadata filter dict.
        """
        db = self._db(level)
        kwargs: dict = {"k": k}
        if filter:
            kwargs["filter"] = filter
        return db.similarity_search(query, **kwargs)

    def search_with_scores(
        self,
        query: str,
        level: str = "symbol",
        k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """Similarity search returning ``(Document, relevance_score)`` pairs.

        Chroma L2 distance is converted: ``relevance = 1 / (1 + distance)``
        so scores are in (0, 1] with higher meaning more relevant.
        """
        db = self._db(level)
        kwargs: dict = {"k": k}
        if filter:
            kwargs["filter"] = filter
        raw = db.similarity_search_with_score(query, **kwargs)
        return [(doc, round(1 / (1 + dist), 4)) for doc, dist in raw]

    def collection_stats(self) -> dict[str, int]:
        """Return document count for each of the four collections."""
        stats: dict[str, int] = {}
        for level in self.LEVELS:
            try:
                result = self._db(level).get(include=[])
                stats[level] = len(result.get("ids", []))
            except Exception:
                stats[level] = 0
        return stats

    # ── BaseIndexer lifecycle ─────────────────────────────────────────────

    @staticmethod
    def _agg(level_stats: dict[str, dict]) -> IndexStats:
        """Aggregate per-level stat dicts into a single IndexStats."""
        return IndexStats.aggregate(
            [IndexStats.from_dict(s) for s in level_stats.values()]
        )

    def ingest(
        self,
        source: tuple["RepoManifest", list["CodeChunk"]],
        store: "SymbolStore | None" = None,
        **_,
    ) -> IndexStats:
        """Index file, symbol, and block collections from *(manifest, chunks)*.

        Equivalent to ``index_all()``.  Already incremental — only changed
        or new documents trigger embedding generation.

        Parameters
        ----------
        source : ``(RepoManifest, list[CodeChunk])`` tuple.
        store  : Optional pre-built SymbolStore; built from chunks if None.
        """
        manifest, chunks = source
        return self._agg(self.index_all(manifest, chunks, store=store))

    def update(
        self,
        source: tuple["RepoManifest", list["CodeChunk"]],
        store: "SymbolStore | None" = None,
        **_,
    ) -> IndexStats:
        """Incrementally update indexed content from *(manifest, chunks)*.

        Aliases ``ingest()`` — ``_upsert()`` already handles add/update/skip.
        """
        return self.ingest(source, store=store)

    def delete(self, source_id: str) -> IndexStats:
        """Remove all documents for repo *source_id* from every collection.

        Parameters
        ----------
        source_id : ``repo_id`` value used when the repository was indexed.
        """
        return self.delete_repo(source_id)

    def reindex(
        self,
        source: tuple["RepoManifest", list["CodeChunk"]],
        store: "SymbolStore | None" = None,
        **_,
    ) -> IndexStats:
        """Full rebuild: delete existing repo data then re-ingest.

        Parameters
        ----------
        source : ``(RepoManifest, list[CodeChunk])`` tuple.
        store  : Optional pre-built SymbolStore.
        """
        manifest, chunks = source
        del_stats = self.delete(manifest.repo_id)
        ing_stats = self.ingest(source, store=store)
        return IndexStats(
            added=ing_stats.added,
            updated=ing_stats.updated,
            skipped=ing_stats.skipped,
            deleted=del_stats.deleted,
        )
