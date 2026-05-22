"""
GCR2.4 — Commit Semantic Indexing: CommitIndexer.

CommitIndexer stores CommitRecord objects in a Chroma collection named
``"commits"`` (configurable) so that git commits can be retrieved by
semantic queries such as "When did reranking get introduced?".

Design
------
- One Chroma document per commit.  Document ID = CommitRecord.commit_id.
- Incremental upsert: incoming ``content_hash`` is compared against stored
  metadata; only changed or new records trigger a new embedding.
- Pruning removes all documents for a given repo_id that are absent from an
  incoming batch, keeping cross-repo documents intact.
- ``search()`` returns plain CommitRecord objects reconstructed from Chroma
  metadata so callers do not need to work with LangChain Document objects
  directly.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from utils.logger import AppLogger
from rag.code.schema import CommitRecord
from rag.indexer import ChangeSet, IndexStats, diff_by_content_hash

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# CommitIndexer
# ---------------------------------------------------------------------------

class CommitIndexer:
    """Chroma-backed store for CommitRecord semantic index.

    Parameters
    ----------
    persist_directory  : Chroma storage directory path.
    embedding_function : Any LangChain ``Embeddings`` instance.
    collection_name    : Chroma collection name (default ``"commits"``).
    """

    DEFAULT_COLLECTION = "commits"

    def __init__(
        self,
        persist_directory: str,
        embedding_function,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self._db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
        self._collection_name = collection_name
        log.info("CommitIndexer ready: collection=%s  dir=%s", collection_name, persist_directory)

    # ── Write ─────────────────────────────────────────────────────────────

    def upsert(
        self,
        records: list[CommitRecord],
        *,
        prune_missing: bool = True,
        repo_id: str = "",
    ) -> IndexStats:
        """Incrementally upsert CommitRecord documents into Chroma.

        Compares ``content_hash`` values against stored metadata.  Only
        changed or new records generate a new embedding.

        When *prune_missing* is True (default), documents for *repo_id* that
        are absent from *records* are deleted.  Cross-repo documents are
        never affected.

        Returns
        -------
        IndexStats with added / updated / skipped / deleted counts.
        """
        if not records:
            return IndexStats()

        docs = [r.to_document() for r in records]
        ids  = [r.commit_id    for r in records]

        # Fetch stored hashes for the scope of the incoming batch.
        try:
            scope_repo = repo_id or (records[0].repo_id if records else "")
            if scope_repo:
                existing_raw = self._db.get(
                    where={"repo_id": scope_repo},
                    include=["metadatas"],
                )
            else:
                existing_raw = self._db.get(include=["metadatas"])
        except Exception:
            existing_raw = {"ids": [], "metadatas": []}

        existing: dict[str, str] = {}
        for eid, emeta in zip(
            existing_raw.get("ids", []),
            existing_raw.get("metadatas", []) or [],
        ):
            existing[eid] = (emeta or {}).get("content_hash", "")

        doc_by_id = dict(zip(ids, docs))
        incoming  = {
            doc_id: doc.metadata.get("content_hash", "")
            for doc_id, doc in doc_by_id.items()
        }
        cs = diff_by_content_hash(existing, incoming)

        if cs.added:
            self._db.add_documents([doc_by_id[i] for i in cs.added], ids=cs.added)
            log.debug("CommitIndexer +%d added", len(cs.added))

        if cs.modified:
            self._db.update_documents(cs.modified, [doc_by_id[i] for i in cs.modified])
            log.debug("CommitIndexer ~%d updated", len(cs.modified))

        if prune_missing and cs.deleted:
            self._db.delete(cs.deleted)
            log.debug("CommitIndexer -%d pruned", len(cs.deleted))

        return IndexStats(
            added=len(cs.added),
            updated=len(cs.modified),
            skipped=len(cs.skipped),
            deleted=len(cs.deleted) if prune_missing else 0,
        )

    def delete_repo(self, repo_id: str) -> int:
        """Delete all commit documents for *repo_id*.

        Returns the number of documents deleted.
        """
        try:
            existing = self._db.get(
                where={"repo_id": repo_id},
                include=["metadatas"],
            )
            ids = existing.get("ids", [])
            if ids:
                self._db.delete(ids)
            log.debug("CommitIndexer.delete_repo: %d removed for %s", len(ids), repo_id)
            return len(ids)
        except Exception as exc:
            log.warning("CommitIndexer.delete_repo failed: %s", exc)
            return 0

    # ── Read ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        repo_id: str | None = None,
        k: int = 10,
    ) -> list[CommitRecord]:
        """Semantic similarity search over commit summaries.

        Parameters
        ----------
        query   : Natural-language query (e.g. "When did reranking get added?").
        repo_id : Optional repository filter.
        k       : Maximum number of results to return.

        Returns
        -------
        CommitRecord objects sorted by relevance (most relevant first).
        """
        kwargs: dict = {"k": k}
        if repo_id is not None:
            kwargs["filter"] = {"repo_id": repo_id}
        try:
            docs = self._db.similarity_search(query, **kwargs)
        except Exception as exc:
            log.warning("CommitIndexer.search failed: %s", exc)
            return []
        return [self._doc_to_record(d) for d in docs]

    def count(self, repo_id: str | None = None) -> int:
        """Return the number of stored commit documents."""
        try:
            if repo_id:
                result = self._db.get(where={"repo_id": repo_id}, include=[])
            else:
                result = self._db.get(include=[])
            return len(result.get("ids", []))
        except Exception:
            return 0

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _doc_to_record(doc: Document) -> CommitRecord:
        m = doc.metadata
        return CommitRecord.from_dict({
            "commit_id":        m.get("commit_id", ""),
            "repo_id":          m.get("repo_id", ""),
            "commit_hash":      m.get("commit_hash", ""),
            "author":           m.get("author", ""),
            "date":             m.get("date", ""),
            "message":          "",   # not stored in metadata; use page_content
            "files_changed":    m.get("files_changed", ""),
            "affected_symbols": m.get("affected_symbols", ""),
            "summary":          doc.page_content,
            "content_hash":     m.get("content_hash", ""),
        })
