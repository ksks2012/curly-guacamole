"""
GCR1.4 — Repo-level Index: RepoIndex.

RepoIndex manages the ``code_repo`` Chroma collection.  Each repository is
stored as a single document that summarises its architecture: branch info,
language breakdown, and a per-file listing with module docstrings.

This class implements ``BaseIndexer`` so it participates in the same lifecycle
(ingest / update / delete / reindex) as ``CodeIndexer`` and ``DocumentIndexer``.

Typical usage
-------------
>>> ri = RepoIndex(persist_dir, embed_fn)
>>> ri.ingest(manifest, chunks=chunks)      # index one repo
>>> ri.search("where is the auth layer?", top_k=3)
>>> ri.delete("my-project")

``CodeIndexer.index_manifest()`` delegates here for backward compatibility.
"""

from __future__ import annotations

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.logger import AppLogger
from rag.code.schema import CodeChunk, RepoManifest
from rag.indexer import BaseIndexer, IndexStats, diff_by_content_hash

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# Text builder
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _repo_text(manifest: RepoManifest, module_docs: dict[str, str]) -> str:
    """Architecture overview text for a repository.

    Combines repo/branch metadata, language breakdown, and per-file module
    docstrings so the repo document answers "what does this repository do?"
    queries.
    """
    source = manifest.source_files()
    lang_counts: dict[str, int] = {}
    for f in source:
        lang_counts[f.language] = lang_counts.get(f.language, 0) + 1
    lang_line = "  ".join(
        f"{lang}={cnt}"
        for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1])
    )
    lines = [
        f"repository: {manifest.repo_id}  branch: {manifest.branch}",
        f"files: {len(manifest.files)}  source_files: {len(source)}",
        f"languages: {lang_line}",
        "",
    ]
    for f in sorted(source, key=lambda x: x.file_path):
        doc = module_docs.get(f.file_path, "")
        first = doc.split("\n")[0].strip() if doc else ""
        lines.append(f"  {f.file_path}" + (f"  —  {first}" if first else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RepoIndex
# ---------------------------------------------------------------------------

class RepoIndex(BaseIndexer):
    """Repo-level Chroma index (one document per repository).

    Manages a single ``code_repo`` Chroma collection.  Each document
    summarises a repository with branch info, language statistics, and
    per-file module docstrings — designed for architecture-level queries.

    Implements ``BaseIndexer`` so repo-level indexing participates in the
    same lifecycle as ``CodeIndexer`` and ``DocumentIndexer``.

    ``source`` for all BaseIndexer methods is a ``RepoManifest``.
    ``source_id`` is ``repo_id``.

    Parameters
    ----------
    persist_directory  : Chroma storage directory path.
    embedding_function : Any LangChain ``Embeddings`` instance.
    collection_name    : Override for the Chroma collection name.
                         Default: ``"code_repo"``.
    """

    COLLECTION_NAME: str = "code_repo"

    def __init__(
        self,
        persist_directory: str,
        embedding_function,
        collection_name: str | None = None,
    ) -> None:
        self._persist_dir = persist_directory
        self._embed = embedding_function
        self._collection_name = collection_name or self.COLLECTION_NAME
        self._db_instance: Chroma | None = None

    # ── Collection access ─────────────────────────────────────────────────

    def _db(self) -> Chroma:
        if self._db_instance is None:
            self._db_instance = Chroma(
                persist_directory=self._persist_dir,
                embedding_function=self._embed,
                collection_name=self._collection_name,
            )
        return self._db_instance

    # ── Incremental upsert ────────────────────────────────────────────────

    def _upsert(self, docs: list[Document], ids: list[str]) -> dict:
        """Upsert repo documents using content-hash diff.

        Pruning is intentionally disabled: each repo_id is independent,
        so there is never a stale document to prune within this collection.

        Returns
        -------
        dict with keys: added, updated, skipped, deleted.
        """
        stats = {"added": 0, "updated": 0, "skipped": 0, "deleted": 0}
        if not docs:
            return stats

        db = self._db()
        try:
            existing_raw = db.get(include=["metadatas"])
        except Exception:
            existing_raw = {"ids": [], "metadatas": []}

        existing: dict[str, str] = {}
        for eid, emeta in zip(
            existing_raw.get("ids", []),
            existing_raw.get("metadatas", []) or [],
        ):
            existing[eid] = (emeta or {}).get("content_hash", "")

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
            log.debug("RepoIndex +%d added", len(changeset.added))

        if changeset.modified:
            mod_docs = [doc_by_id[i] for i in changeset.modified]
            db.update_documents(changeset.modified, mod_docs)
            stats["updated"] += len(changeset.modified)
            log.debug("RepoIndex ~%d updated", len(changeset.modified))

        stats["skipped"] += len(changeset.skipped)
        return stats

    # ── Core indexing logic ───────────────────────────────────────────────

    def index_manifest(
        self,
        manifest: RepoManifest,
        chunks: list[CodeChunk] | None = None,
    ) -> dict:
        """Index a single repo-level document summarising the repository.

        The embedded text covers branch, language breakdown, and a per-file
        list with module docstrings — suitable for architecture queries like
        "what does this repo do?" or "where is the auth layer?".

        Parameters
        ----------
        manifest : RepoManifest from RepoScanner.
        chunks   : Optional CodeChunk list; used to extract module docstrings.

        Returns
        -------
        dict with keys: added, updated, skipped, deleted.
        """
        module_docs: dict[str, str] = {}
        if chunks:
            for c in chunks:
                if c.chunk_type == "module" and c.docstring:
                    module_docs[c.file_path] = c.docstring

        text = _repo_text(manifest, module_docs)

        # content_hash = hash of all file hashes → changes when any file changes
        all_hashes = "".join(
            f.content_hash
            for f in sorted(manifest.files.values(), key=lambda x: x.file_path)
        )
        repo_hash = _sha256(all_hashes)

        doc = Document(
            page_content=text,
            metadata={
                "repo_id":      manifest.repo_id,
                "branch":       manifest.branch,
                "scanned_at":   manifest.scanned_at,
                "file_count":   len(manifest.files),
                "source_count": len(manifest.source_files()),
                "content_hash": repo_hash,
            },
        )
        doc_id = f"{manifest.repo_id}::repo"
        return self._upsert([doc], [doc_id])

    # ── Query ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[Document]:
        """Similarity search against the repo collection.

        Parameters
        ----------
        query   : Natural language query.
        top_k   : Maximum number of results.
        filters : Optional Chroma metadata filter dict.
        """
        db = self._db()
        kwargs: dict = {"k": top_k}
        if filters:
            kwargs["filter"] = filters
        return db.similarity_search(query, **kwargs)

    def collection_stats(self) -> int:
        """Return document count in the repo collection."""
        try:
            return len(self._db().get(include=[]).get("ids", []))
        except Exception:
            return 0

    # ── BaseIndexer lifecycle ─────────────────────────────────────────────

    def ingest(
        self,
        source: "RepoManifest",
        chunks: "list[CodeChunk] | None" = None,
        **_,
    ) -> IndexStats:
        """Index the repo-level document for *source* (a RepoManifest).

        Parameters
        ----------
        source : RepoManifest from RepoScanner.
        chunks : Optional CodeChunk list for richer module-docstring coverage.
        """
        return IndexStats.from_dict(self.index_manifest(source, chunks))

    def update(
        self,
        source: "RepoManifest",
        chunks: "list[CodeChunk] | None" = None,
        **_,
    ) -> IndexStats:
        """Incrementally update the repo document from *source*.

        Aliases ``ingest()`` — ``_upsert()`` handles add/update/skip.
        """
        return self.ingest(source, chunks=chunks)

    def delete(self, source_id: str) -> IndexStats:
        """Remove the repo document for *source_id* (``repo_id`` value).

        Parameters
        ----------
        source_id : The ``repo_id`` used when the repository was indexed.
        """
        try:
            db = self._db()
            result = db.get(where={"repo_id": source_id}, include=[])
            ids = result.get("ids", [])
            if ids:
                db.delete(ids)
                log.info(
                    "RepoIndex.delete: removed %d doc(s) for repo_id=%r",
                    len(ids), source_id,
                )
            return IndexStats(deleted=len(ids))
        except Exception as e:
            log.warning("RepoIndex.delete: %s", e)
            return IndexStats()

    def reindex(
        self,
        source: "RepoManifest",
        chunks: "list[CodeChunk] | None" = None,
        **_,
    ) -> IndexStats:
        """Full rebuild: delete existing repo document then re-index.

        Parameters
        ----------
        source : RepoManifest from RepoScanner.
        chunks : Optional CodeChunk list for richer module-docstring coverage.
        """
        del_stats = self.delete(source.repo_id)
        ing_stats = self.ingest(source, chunks=chunks)
        return IndexStats(
            added=ing_stats.added,
            updated=ing_stats.updated,
            skipped=ing_stats.skipped,
            deleted=del_stats.deleted,
        )