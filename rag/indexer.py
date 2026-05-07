"""
Indexer — manages SQLRecordManager initialisation and document ingestion.

Responsibilities:
  - Own the SQLRecordManager lifecycle (schema creation, namespace)
  - Choose the correct cleanup strategy and batch size based on document count
  - Call LangChain's index() to write documents to the vector store

LocalLlamaClient holds an Indexer and delegates all write operations to it.
"""

from langchain_classic.indexes import SQLRecordManager, index
from langchain_core.documents import Document

from utils.logger import AppLogger

log = AppLogger.get(__name__)


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
