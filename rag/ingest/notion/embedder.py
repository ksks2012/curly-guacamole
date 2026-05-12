"""
Step 1.2 — Embedding Pipeline for Notion pages.

Flow per page
-------------
    RawStore.list_pages()
        → NotionClient.get_all_blocks(notion_page_id)   [live API]
        → NotionClient.raw_blocks_to_models()
        → NotionChunker.chunk()
        → Chunk.to_document(page, workspace)             [LangChain Document]
        → Indexer.run()                                  [Chroma + RecordManager]

Deduplication
-------------
All documents produced from one page share ``source_id = page.id``.
The LangChain ``index()`` function uses ``source_id_key="source_id"`` with
``cleanup="incremental"`` so that:
    - Unchanged chunks are skipped (SHA-256 fingerprint match).
    - Removed chunks (page was re-chunked after a Notion edit) are deleted.
    - New chunks are added.

Usage
-----
    from rag.ingest.notion.embedder import NotionEmbedder
    from rag.knowledge.store import RawStore
    from rag.ingest.notion.client import NotionClient
    from utils.config import AppConfig

    config  = AppConfig()
    store   = RawStore(config.raw_db_path)
    client  = NotionClient(config.notion_token)
    embedder = NotionEmbedder(config, store, client)

    result = embedder.embed_workspace(workspace_id)
    print(result)   # EmbedResult(pages_embedded=2, chunks_added=90, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.indexer import Indexer
from rag.knowledge.models import Page, Workspace
from rag.knowledge.store import RawStore
from rag.ingest.notion.chunker import NotionChunker
from rag.ingest.notion.client import NotionClient

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class EmbedResult:
    pages_embedded: int       = 0
    pages_skipped:  int       = 0
    chunks_added:   int       = 0
    chunks_updated: int       = 0
    chunks_skipped: int       = 0
    errors:         int       = 0
    error_titles:   list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "pages_embedded": self.pages_embedded,
            "pages_skipped":  self.pages_skipped,
            "chunks_added":   self.chunks_added,
            "chunks_updated": self.chunks_updated,
            "chunks_skipped": self.chunks_skipped,
            "errors":         self.errors,
        }

    def __str__(self) -> str:
        return (
            f"EmbedResult("
            f"pages_embedded={self.pages_embedded}, "
            f"pages_skipped={self.pages_skipped}, "
            f"chunks_added={self.chunks_added}, "
            f"chunks_updated={self.chunks_updated}, "
            f"chunks_skipped={self.chunks_skipped}, "
            f"errors={self.errors})"
        )


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class NotionEmbedder:
    """Embed all Notion pages for a workspace into the Chroma vector store.

    Args:
        config  : AppConfig instance.
        store   : RawStore holding workspace + page metadata.
        client  : NotionClient for live block fetching.
        chunker : Optional NotionChunker; a default instance is used if omitted.
    """

    def __init__(
        self,
        config: AppConfig,
        store: RawStore,
        client: NotionClient,
        chunker: NotionChunker | None = None,
    ) -> None:
        self._store   = store
        self._client  = client
        self._chunker = chunker or NotionChunker()

        log.info("Building embeddings client → %s", config.embed_base)
        self._embedding_version = config.embed_model
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=config.api_key,
            openai_api_base=config.embed_base,
            model=config.embed_model,
        )

        log.info("Opening Chroma store → %s", config.persist_directory)
        self._chroma = Chroma(
            persist_directory=config.persist_directory,
            embedding_function=self._embeddings,
            collection_name=config.setup_rag_collection or "rag_collection",
        )

        self._indexer = Indexer(
            db=self._chroma,
            namespace=config.setup_rag_collection or "rag_collection",
            db_url=config.db_url,
        )

        log.info("NotionEmbedder ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_workspace(self, workspace_id: str) -> EmbedResult:
        """Embed all pages belonging to *workspace_id*.

        Loads workspace and page metadata from RawStore, fetches live blocks
        from the Notion API, chunks, and indexes into Chroma.

        Returns:
            EmbedResult with per-page and per-chunk counts.
        """
        workspace = self._store.get_workspace(workspace_id)
        if workspace is None:
            raise ValueError(f"Workspace {workspace_id!r} not found in RawStore")

        pages = self._store.list_pages(workspace_id)
        log.info(
            "embed_workspace: workspace=%r  pages=%d",
            workspace.name, len(pages),
        )

        result = EmbedResult()
        for page in pages:
            try:
                stats = self._embed_page(page, workspace)
                result.pages_embedded += 1
                result.chunks_added   += stats.get("num_added",   0)
                result.chunks_updated += stats.get("num_updated", 0)
                result.chunks_skipped += stats.get("num_skipped", 0)
            except Exception as exc:
                result.errors += 1
                result.error_titles.append(page.title)
                log.error(
                    "embed_workspace: error on page %r (%s): %s",
                    page.title, page.id[:8], exc, exc_info=True,
                )

        log.info("embed_workspace complete: %s", result)
        return result

    def embed_page(self, page_id: str) -> EmbedResult:
        """Embed a single page by its RawStore page UUID.

        Convenience method for incremental / on-demand embedding.
        """
        page = self._store.get_page(page_id)
        if page is None:
            raise ValueError(f"Page {page_id!r} not found in RawStore")
        workspace = self._store.get_workspace(page.workspace_id)
        if workspace is None:
            raise ValueError(f"Workspace {page.workspace_id!r} not found in RawStore")

        result = EmbedResult()
        try:
            stats = self._embed_page(page, workspace)
            result.pages_embedded  = 1
            result.chunks_added   = stats.get("num_added",   0)
            result.chunks_updated = stats.get("num_updated", 0)
            result.chunks_skipped = stats.get("num_skipped", 0)
        except Exception as exc:
            result.errors = 1
            result.error_titles.append(page.title)
            log.error(
                "embed_page: error on %r (%s): %s",
                page.title, page.id[:8], exc, exc_info=True,
            )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed_page(self, page: Page, workspace: Workspace) -> dict:
        """Fetch blocks, chunk, build Documents, and index for one page.

        Returns the raw stats dict from Indexer.run().
        """
        if not page.notion_page_id:
            log.warning(
                "_embed_page: page %r has no notion_page_id — skipping", page.title
            )
            return {}

        log.info(
            "_embed_page: %r  notion_id=%s",
            page.title, page.notion_page_id[:8],
        )

        # 1. Fetch blocks from Notion API
        raw_blocks = self._client.get_all_blocks(page.notion_page_id)
        blocks     = NotionClient.raw_blocks_to_models(raw_blocks, page.id)
        log.debug("  fetched %d blocks", len(blocks))

        # 2. Chunk
        chunks = self._chunker.chunk(blocks, page.id)
        log.debug("  produced %d chunks", len(chunks))

        if not chunks:
            log.warning("  no chunks produced for page %r", page.title)
            return {}

        # 3. Convert to LangChain Documents
        docs = [
            chunk.to_document(
                page, workspace,
                embedding_version=self._embedding_version,
                chunk_version="notion-chunker-v1",
            )
            for chunk in chunks
        ]

        # 4. Index into Chroma (handles dedup via SQLRecordManager)
        stats = self._indexer.run(docs)
        log.info(
            "  indexed: added=%d  updated=%d  skipped=%d  deleted=%d",
            stats.get("num_added",   0),
            stats.get("num_updated", 0),
            stats.get("num_skipped", 0),
            stats.get("num_deleted", 0),
        )
        return stats
