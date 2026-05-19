"""
Notion controller — logic layer for the Notion tab.

Responsibilities:
  - Wrap NotionRAGClient (lazy-init on first use)
  - Expose sync, embed, sync_and_embed operations (returns plain dicts)
  - Expose hybrid search and RAG query over Notion content
  - Cache page list for the tab's sidebar

Has NO dependency on NiceGUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.ingest.notion.pipeline import NotionRAGClient, SyncEmbedResult

log = AppLogger.get(__name__)


@dataclass
class SearchResult:
    query:   str = ""
    chunks:  list[dict] = field(default_factory=list)   # {content, score, meta}
    answer:  str = ""
    mode:    str = "hybrid"   # "hybrid" | "rag"
    error:   str = ""


class NotionController:
    """Bridges NotionRAGClient and the Notion tab display layer."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._rag: NotionRAGClient | None = None
        self._pages: list = []          # rag.knowledge.models.Page objects
        self._last_sync: dict = {}
        self._last_embed: dict = {}
        self._last_result: SearchResult = SearchResult()

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    @property
    def rag(self) -> NotionRAGClient:
        if self._rag is None:
            log.info("NotionController: initialising NotionRAGClient")
            self._rag = NotionRAGClient(self._config)
        return self._rag

    def is_configured(self) -> bool:
        """Return True when a Notion token is set in config."""
        return bool(self._config.notion_token)

    # ------------------------------------------------------------------
    # Sync / embed
    # ------------------------------------------------------------------

    def sync(self, full_sync: bool = False) -> dict:
        """Pull changes from Notion into RawStore.  Returns SyncResult as dict."""
        try:
            if full_sync:
                self._rag = NotionRAGClient(self._config, full_sync=True)
            result = self.rag.sync()
            self._last_sync = result.as_dict()
            self._reload_pages()
            log.info("NotionController.sync: %s", self._last_sync)
            return self._last_sync
        except Exception as exc:
            log.error("NotionController.sync failed: %s", exc, exc_info=True)
            return {"error": str(exc)}

    def embed(self) -> dict:
        """Embed synced pages into Chroma.  Returns EmbedResult as dict."""
        try:
            result = self.rag.embed()
            self._last_embed = result.as_dict()
            log.info("NotionController.embed: %s", self._last_embed)
            return self._last_embed
        except Exception as exc:
            log.error("NotionController.embed failed: %s", exc, exc_info=True)
            return {"error": str(exc)}

    def sync_and_embed(self, full_sync: bool = False) -> dict:
        """Sync then embed in one step."""
        try:
            if full_sync:
                self._rag = NotionRAGClient(self._config, full_sync=True)
            result: SyncEmbedResult = self.rag.sync_and_embed()
            self._last_sync  = result.sync.as_dict()
            self._last_embed = result.embed.as_dict()
            self._reload_pages()
            combined = result.as_dict()
            log.info("NotionController.sync_and_embed: %s", combined)
            return combined
        except Exception as exc:
            log.error("NotionController.sync_and_embed failed: %s", exc, exc_info=True)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def _reload_pages(self) -> None:
        try:
            self._pages = self.rag.list_pages()
        except Exception as exc:
            log.warning("NotionController._reload_pages failed: %s", exc)

    def load_pages(self) -> list:
        """Load page list from RawStore (does NOT call Notion API)."""
        self._reload_pages()
        return self._pages

    @property
    def pages(self) -> list:
        return self._pages

    # ------------------------------------------------------------------
    # Search / query
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, k: int = 5, fetch_k: int = 20) -> SearchResult:
        """Hybrid search over Notion chunks (vector + BM25)."""
        result = SearchResult(query=query, mode="hybrid")
        try:
            hits = self.rag.hybrid_search(query, k=k, fetch_k=fetch_k)
            result.chunks = [
                {"content": doc.page_content, "score": score, "meta": doc.metadata}
                for doc, score in hits
            ]
        except Exception as exc:
            result.error = str(exc)
            log.error("NotionController.hybrid_search failed: %s", exc, exc_info=True)
        self._last_result = result
        return result

    def rag_query(self, question: str, k: int = 5, fetch_k: int = 20) -> SearchResult:
        """Full RAG pipeline: retrieve Notion chunks then generate an answer."""
        result = SearchResult(query=question, mode="rag")
        try:
            answer = self.rag.query_with_filter(question, k=k, fetch_k=fetch_k)
            result.answer = answer
            # Also run hybrid search so we can show source chunks
            hits = self.rag.hybrid_search(question, k=k, fetch_k=fetch_k)
            result.chunks = [
                {"content": doc.page_content, "score": score, "meta": doc.metadata}
                for doc, score in hits
            ]
        except Exception as exc:
            result.error = str(exc)
            log.error("NotionController.rag_query failed: %s", exc, exc_info=True)
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def last_sync(self) -> dict:
        return self._last_sync

    @property
    def last_embed(self) -> dict:
        return self._last_embed

    @property
    def last_result(self) -> SearchResult:
        return self._last_result
