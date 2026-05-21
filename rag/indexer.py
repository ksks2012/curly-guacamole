"""
Indexer — manages SQLRecordManager initialisation and document ingestion.

Responsibilities:
  - Own the SQLRecordManager lifecycle (schema creation, namespace)
  - Choose the correct cleanup strategy and batch size based on document count
  - Call LangChain's index() to write documents to the vector store

LocalLlamaClient holds an Indexer and delegates all write operations to it.

BaseIndexer / IndexStats
------------------------
``BaseIndexer`` is an ABC that ``DocumentIndexer`` and ``CodeIndexer`` both
implement.  It defines the four lifecycle operations every indexer must
support:

    ingest(source, **kwargs)   — add content from *source* for the first time
    update(source, **kwargs)   — incrementally re-index changed content
    delete(source_id)          — remove all content identified by *source_id*
    reindex(source, **kwargs)  — full rebuild: delete + ingest

``IndexStats`` is the unified return type so callers can log or display
added/updated/skipped/deleted counts without knowing which backend ran.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_classic.indexes import SQLRecordManager, index
from langchain_core.documents import Document

from utils.logger import AppLogger

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# IndexStats — unified return type for all indexer operations
# ---------------------------------------------------------------------------

@dataclass
class IndexStats:
    """Counts of documents affected by one indexer operation."""

    added:   int = 0
    updated: int = 0
    skipped: int = 0
    deleted: int = 0

    def __add__(self, other: "IndexStats") -> "IndexStats":
        return IndexStats(
            added=self.added     + other.added,
            updated=self.updated + other.updated,
            skipped=self.skipped + other.skipped,
            deleted=self.deleted + other.deleted,
        )

    def __repr__(self) -> str:
        return (
            f"IndexStats(added={self.added}, updated={self.updated}, "
            f"skipped={self.skipped}, deleted={self.deleted})"
        )

    @classmethod
    def from_dict(cls, d: dict) -> "IndexStats":
        """Build from LangChain index() result dict (num_added, etc.)."""
        return cls(
            added=d.get("num_added",   d.get("added",   0)),
            updated=d.get("num_updated", d.get("updated", 0)),
            skipped=d.get("num_skipped", d.get("skipped", 0)),
            deleted=d.get("num_deleted", d.get("deleted", 0)),
        )

    @classmethod
    def aggregate(cls, stats: list["IndexStats"]) -> "IndexStats":
        """Sum a list of IndexStats into one."""
        result = cls()
        for s in stats:
            result = result + s
        return result


# ---------------------------------------------------------------------------
# BaseIndexer — abstract lifecycle contract
# ---------------------------------------------------------------------------

class BaseIndexer(ABC):
    """Abstract base class for all indexers.

    Every indexer — document, code, git — must implement these four lifecycle
    operations so that higher-level orchestration code can treat them
    interchangeably.

    Subclasses
    ----------
    DocumentIndexer  — wraps DocumentIngester + Indexer (LangChain records)
    CodeIndexer      — wraps multi-resolution Chroma collections
    GitIndexer       — (future) wraps git snapshot store
    """

    @abstractmethod
    def ingest(self, source, **kwargs) -> IndexStats:
        """Add content from *source* to the index for the first time.

        For document indexers *source* is a file path.
        For code indexers *source* is a (RepoManifest, chunks) pair.
        """

    @abstractmethod
    def update(self, source, **kwargs) -> IndexStats:
        """Incrementally update indexed content from *source*.

        Only changed or new content should trigger embedding generation.
        Implementations that are already incremental may alias this to ingest.
        """

    @abstractmethod
    def delete(self, source_id: str) -> IndexStats:
        """Remove all indexed content identified by *source_id*.

        For document indexers *source_id* is the ``doc_id`` metadata key.
        For code indexers *source_id* is the ``repo_id``.
        """

    @abstractmethod
    def reindex(self, source, **kwargs) -> IndexStats:
        """Full rebuild: delete existing content then re-ingest from *source*.

        Combines ``delete(source_id)`` + ``ingest(source, **kwargs)`` so
        callers do not need to manage the two-step sequence manually.
        """


class Indexer:
    """
    Wraps SQLRecordManager and the LangChain index() function.

    Args:
        db          : Chroma (or any LangChain VectorStore) instance.
        namespace   : SQLRecordManager namespace string.
        db_url      : SQLAlchemy-compatible URL for the record manager database.
        batch_limit : Threshold controlling cleanup strategy and batch size.
                      When len(docs) > batch_limit, 'scoped_full' cleanup is used;
                      otherwise 'incremental'. Defaults to 256.
    """

    def __init__(self, db, namespace: str, db_url: str, batch_limit: int = 256):
        self.db = db
        self.batch_limit = batch_limit
        log.info("Indexer: namespace=%r  db_url=%s  batch_limit=%d",
                 namespace, db_url, batch_limit)
        try:
            self.record_manager = SQLRecordManager(namespace, db_url=db_url)
            self.record_manager.create_schema()
            log.info("Indexer: record manager ready")
        except Exception as e:
            log.error("Indexer: failed to initialise record manager: %s", e, exc_info=True)
            raise

    def run(self, docs: list[Document]) -> dict:
        """Index documents into the vector store using the record manager.

        Selects cleanup strategy automatically:
          - len(docs) > batch_limit → 'scoped_full'  (conservative, smaller batches)
          - len(docs) <= batch_limit → 'incremental' (full-batch, faster)

        Returns:
            dict with keys num_added, num_updated, num_skipped, num_deleted.
        """
        stats = {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}
        try:
            if len(docs) > self.batch_limit:
                cleanup = "scoped_full"
                batch_size = min(100, self.batch_limit)
            else:
                cleanup = "incremental"
                batch_size = max(1, len(docs))

            log.info("Indexer.run: %d docs  cleanup=%s  batch_size=%d",
                     len(docs), cleanup, batch_size)

            stats = index(
                docs_source=docs,
                record_manager=self.record_manager,
                vector_store=self.db,
                cleanup=cleanup,
                source_id_key="source_id",
                key_encoder="sha256",
                batch_size=batch_size,
            )
            log.info("Indexer.run complete: %s", stats)
        except Exception as e:
            log.error("Indexer.run failed: %s", e, exc_info=True)

        return stats
