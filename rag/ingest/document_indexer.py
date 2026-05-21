"""
DocumentIndexer — unified lifecycle wrapper for document ingestion.

Combines DocumentIngester (parse + chunk) and Indexer (LangChain record
manager + Chroma writes) behind the BaseIndexer interface so that document
indexing participates in the same lifecycle as CodeIndexer.

Usage
-----
    indexer = DocumentIndexer(
        db=chroma_db,
        namespace="langchain/documents",
        db_url="sqlite:///my_db/record_manager_cache.sql",
    )

    # First-time ingest
    stats = indexer.ingest("docs/architecture.md", doc_id="arch")

    # Incremental update (only changed chunks re-embedded)
    stats = indexer.update("docs/architecture.md", doc_id="arch")

    # Remove all chunks for a document
    stats = indexer.delete("arch")

    # Full rebuild
    stats = indexer.reindex("docs/architecture.md", doc_id="arch")
"""

from __future__ import annotations

import os
from pathlib import Path

from rag.indexer import BaseIndexer, IndexStats, Indexer
from rag.ingest.document_ingester import DocumentIngester
from utils.logger import AppLogger

log = AppLogger.get(__name__)


class DocumentIndexer(BaseIndexer):
    """Lifecycle wrapper: DocumentIngester + Indexer behind BaseIndexer.

    Parameters
    ----------
    db          : LangChain VectorStore (Chroma) instance.
    namespace   : SQLRecordManager namespace string.
    db_url      : SQLAlchemy-compatible URL for the record manager database.
    embeddings  : LangChain Embeddings instance (required for semantic
                  chunking strategy; ignored otherwise).
    batch_limit : Passed to the inner Indexer — controls cleanup strategy
                  and batch size selection.
    """

    def __init__(
        self,
        db,
        namespace: str,
        db_url: str,
        embeddings=None,
        batch_limit: int = 256,
    ) -> None:
        self._ingester = DocumentIngester(embeddings=embeddings)
        self._indexer  = Indexer(
            db=db,
            namespace=namespace,
            db_url=db_url,
            batch_limit=batch_limit,
        )

    # ------------------------------------------------------------------
    # BaseIndexer interface
    # ------------------------------------------------------------------

    def ingest(self, source: str, doc_id: str | None = None, **kwargs) -> IndexStats:
        """Parse *source* path and index all resulting chunks.

        Parameters
        ----------
        source  : Absolute or relative path to a supported document file.
        doc_id  : Grouping / filter key stored in chunk metadata.
                  Defaults to the filename stem when not supplied.
        **kwargs: Forwarded to ``DocumentIngester.ingest()`` — chunk_size,
                  chunk_overlap, title, tags, workspace, strategy, etc.

        Returns
        -------
        IndexStats with added/updated/skipped/deleted counts.
        """
        resolved_id = doc_id or Path(source).stem
        docs = self._ingester.ingest(path=source, doc_id=resolved_id, **kwargs)
        raw  = self._indexer.run(docs)
        return IndexStats.from_dict(raw)

    def update(self, source: str, doc_id: str | None = None, **kwargs) -> IndexStats:
        """Incrementally update indexed content from *source*.

        Delegates to ``ingest()`` — the inner Indexer's SQLRecordManager
        already handles deduplication and skips unchanged chunks.
        """
        return self.ingest(source, doc_id=doc_id, **kwargs)

    def delete(self, source_id: str) -> IndexStats:
        """Remove all chunks whose ``doc_id`` metadata equals *source_id*.

        Parameters
        ----------
        source_id : Value of the ``doc_id`` metadata field to remove.
        """
        db = self._indexer.db
        try:
            result = db.get(where={"doc_id": source_id}, include=[])
            ids = result.get("ids", [])
            if ids:
                db.delete(ids)
                log.info(
                    "DocumentIndexer.delete: removed %d chunks for doc_id=%r",
                    len(ids), source_id,
                )
            return IndexStats(deleted=len(ids))
        except Exception as exc:
            log.error("DocumentIndexer.delete failed for doc_id=%r: %s", source_id, exc)
            return IndexStats()

    def reindex(self, source: str, doc_id: str | None = None, **kwargs) -> IndexStats:
        """Full rebuild: delete existing chunks then re-ingest from *source*.

        Parameters
        ----------
        source  : Path to the document file.
        doc_id  : Same doc_id used during the original ingest call.
                  Defaults to the filename stem.
        **kwargs: Forwarded to ``ingest()``.
        """
        resolved_id = doc_id or Path(source).stem
        del_stats = self.delete(resolved_id)
        ing_stats = self.ingest(source, doc_id=resolved_id, **kwargs)
        return IndexStats(
            added=ing_stats.added,
            updated=ing_stats.updated,
            skipped=ing_stats.skipped,
            deleted=del_stats.deleted,
        )
