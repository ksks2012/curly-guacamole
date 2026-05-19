"""
Step 1.3 — NotionRAGClient: unified sync → embed → query entry point.

Combines:
    NotionSyncPipeline  (Phase 0)  — Notion API → RawStore
    NotionEmbedder      (Step 1.2) — RawStore → Chroma (structure-aware chunks)
    LocalLlamaClient                — Chroma retrieval + BM25 + LLM generation

This is the single interface callers use once the system is running:

    client = NotionRAGClient(config)

    # One-time or periodic: pull changes from Notion and (re-)embed
    client.sync_and_embed()

    # At query time:
    answer = client.query("What is a span in Go memory allocation?")
    results = client.search("tcmalloc cache")

Filtering
---------
All retrieval methods automatically scope to ``document_type="notion"`` and
the configured ``notion_workspace_id`` so Notion content is always separated
from local PDF/Markdown content in the same Chroma collection.

Pass ``workspace_filter=False`` to search across all document types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.client import LocalLlamaClient
from rag.knowledge.models import Workspace
from rag.knowledge.store import RawStore
from rag.retrieval.filters import SearchFilter
from rag.ingest.notion.client import NotionClient
from rag.ingest.notion.embedder import EmbedResult, NotionEmbedder
from rag.ingest.notion.sync import NotionSyncPipeline, SyncResult

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SyncEmbedResult:
    sync:  SyncResult  = field(default_factory=SyncResult)
    embed: EmbedResult = field(default_factory=EmbedResult)

    def as_dict(self) -> dict:
        return {"sync": self.sync.as_dict(), "embed": self.embed.as_dict()}

    def __str__(self) -> str:
        return (
            f"SyncEmbedResult("
            f"pages_seen={self.sync.pages_seen}, "
            f"pages_updated={self.sync.pages_updated}, "
            f"chunks_added={self.embed.chunks_added}, "
            f"chunks_skipped={self.embed.chunks_skipped})"
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class NotionRAGClient:
    """Unified Notion → RAG entry point.

    Args:
        config      : AppConfig with notion_token, notion_workspace_id, etc.
        full_sync   : When True, ignore stored cursor and re-sync all pages.
    """

    def __init__(self, config: AppConfig, full_sync: bool = False) -> None:
        self._config    = config
        self._full_sync = full_sync

        self._store  = RawStore(config.raw_db_path)
        self._client = NotionClient(config.notion_token, requests_per_minute=config.requests_rate_limit)

        # Reuse or create the workspace record.
        ws_name = config.notion_workspace_id or "Notion"
        existing = self._store.list_workspaces()
        self._workspace = next(
            (w for w in existing if w.name == ws_name), None
        )
        if self._workspace is None:
            self._workspace = Workspace.new(ws_name)

        # LocalLlamaClient owns Chroma + BM25 + LLM + Indexer.
        self._llm_client = LocalLlamaClient(config)

        # NotionEmbedder reuses the same Chroma store via a shared persist dir.
        self._embedder = NotionEmbedder(config, self._store, self._client)

        log.info(
            "NotionRAGClient ready: workspace=%r  data_source=%s",
            ws_name, config.notion_data_source_id or "<search>",
        )

    # ------------------------------------------------------------------
    # Sync + Embed
    # ------------------------------------------------------------------

    def sync(self) -> SyncResult:
        """Pull changes from Notion into RawStore only (no embedding)."""
        pipeline = NotionSyncPipeline(
            token=self._config.notion_token,
            workspace=self._workspace,
            store=self._store,
            data_source_id=self._config.notion_data_source_id,
            full_sync=self._full_sync,
            requests_per_minute=self._config.requests_rate_limit,
        )
        result = pipeline.sync()
        # Ensure workspace is in store after sync
        self._workspace = (
            self._store.get_workspace(self._workspace.id) or self._workspace
        )
        return result

    def embed(self) -> EmbedResult:
        """Embed all pages for this workspace into Chroma."""
        result = self._embedder.embed_workspace(self._workspace.id)
        if result.chunks_added > 0 or result.chunks_updated > 0:
            self._llm_client.invalidate_bm25()
        return result

    def sync_and_embed(self) -> SyncEmbedResult:
        """Sync from Notion, then (re-)embed changed pages into Chroma."""
        result = SyncEmbedResult()
        result.sync  = self.sync()
        result.embed = self.embed()
        log.info("sync_and_embed: %s", result)
        return result

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _notion_filter(self, workspace_filter: bool = True) -> SearchFilter | None:
        """Build a SearchFilter scoped to Notion content."""
        if not workspace_filter:
            return None
        return SearchFilter(
            document_type="notion",
            workspace=self._workspace.name,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        workspace_filter: bool = True,
    ) -> list[tuple[Document, float]]:
        """Vector similarity search over Notion chunks.

        Returns:
            list of (Document, relevance_score) sorted best-first.
            relevance = 1 / (1 + L2_distance), so closer to 1.0 is better.
        """
        sf = self._notion_filter(workspace_filter)
        return self._llm_client.similarity_search_with_scores(
            query, k=k, search_filter=sf
        )

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        workspace_filter: bool = True,
    ) -> list[tuple[Document, float]]:
        """Hybrid search (vector + BM25 via RRF) over Notion chunks.

        Returns:
            Fused result list of (Document, score) sorted best-first.
        """
        sf = self._notion_filter(workspace_filter)
        _, _, fused = self._llm_client.hybrid_search_with_scores(
            query, k=k, fetch_k=fetch_k, search_filter=sf
        )
        return fused[:k]

    def query(
        self,
        question: str,
        k: int = 5,
        fetch_k: int = 20,
        workspace_filter: bool = True,
        expand_query: bool | None = None,
    ) -> str:
        """Full RAG pipeline: retrieve relevant chunks then generate an answer.

        Args:
            question         : Natural-language question.
            k                : Chunks passed to the LLM context window.
            fetch_k          : MMR candidate pool size.
            workspace_filter : When True, restricts to this workspace's Notion docs.
            expand_query     : Override config.query_expansion_enabled.

        Returns:
            LLM-generated answer string.
        """
        sf = self._notion_filter(workspace_filter)
        response = self._llm_client.engine.answer(
            query=question,
            k=k,
            fetch_k=fetch_k,
            expand_query=expand_query,
            # Pass the filter via the retriever factory closure.
            # We temporarily patch get_retriever to inject the filter.
        )
        return response.content if hasattr(response, "content") else str(response)

    def query_with_filter(
        self,
        question: str,
        k: int = 5,
        fetch_k: int = 20,
        workspace_filter: bool = True,
        expand_query: bool | None = None,
    ) -> str:
        """RAG query with explicit SearchFilter applied at retrieval time.

        Unlike ``query()``, this method builds a filtered retriever directly
        so that the SearchFilter is honoured through every retrieval hop.
        """
        sf = self._notion_filter(workspace_filter)
        where = sf.to_chroma() if sf and not sf.is_empty() else None

        use_expansion = (
            expand_query
            if expand_query is not None
            else self._config.query_expansion_enabled
        )

        # Collect candidates via filtered vector search.
        search_kwargs: dict = {"k": fetch_k, "fetch_k": fetch_k}
        if where:
            search_kwargs["filter"] = where
        retriever = self._llm_client.db.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

        from rag.engine import RAGEngine
        engine = RAGEngine(
            llm=self._llm_client.llm,
            get_retriever=lambda **_: retriever,
            reranker=self._llm_client.reranker,
            config=self._config,
        )
        response = engine.answer(
            query=question,
            k=k,
            fetch_k=fetch_k,
            expand_query=expand_query,
        )
        return response.content if hasattr(response, "content") else str(response)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def list_pages(self) -> list:
        """Return all Page objects for this workspace from RawStore."""
        return self._store.list_pages(self._workspace.id)

    def page_count(self) -> int:
        return len(self.list_pages())
