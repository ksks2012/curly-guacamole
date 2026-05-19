"""
Knowledge controller — logic layer for the Knowledge Card tab.

Responsibilities:
  - Browse all indexed chunks from Chroma with rich metadata
  - Maintain filter state (doc_id, tag, topic, text search)
  - Expose sorted/filtered chunk list to the UI layer

Has NO dependency on NiceGUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.logger import AppLogger
from rag.client import LocalLlamaClient

log = AppLogger.get(__name__)


@dataclass
class KnowledgeFilter:
    doc_id: str = ""
    tag:    str = ""
    topic:  str = ""
    text:   str = ""

    def is_empty(self) -> bool:
        return not (self.doc_id or self.tag or self.topic or self.text)

    def summary(self) -> str:
        parts = []
        if self.doc_id:
            parts.append(f"doc={self.doc_id[:20]}")
        if self.tag:
            parts.append(f"tag={self.tag}")
        if self.topic:
            parts.append(f"topic={self.topic}")
        if self.text:
            parts.append(f'text="{self.text[:20]}"')
        return "  ·  ".join(parts) if parts else "none"


class KnowledgeController:
    """Bridges Chroma chunk browsing and the Knowledge Card display layer."""

    def __init__(self, client: LocalLlamaClient) -> None:
        self._client = client
        self._all_chunks: list[dict] = []
        self._filter = KnowledgeFilter()
        self._loaded = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def filter(self) -> KnowledgeFilter:
        return self._filter

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def total_count(self) -> int:
        return len(self._all_chunks)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Fetch all chunks from Chroma and cache them locally."""
        try:
            self._all_chunks = self._client.browse_chunks(limit=500)
            self._loaded = True
            log.info("KnowledgeController.reload: %d chunks loaded", len(self._all_chunks))
        except Exception as exc:
            log.error("KnowledgeController.reload failed: %s", exc, exc_info=True)
            self._all_chunks = []
            self._loaded = False

    # ------------------------------------------------------------------
    # Filter mutations
    # ------------------------------------------------------------------

    def set_filter_doc_id(self, doc_id: str) -> None:
        self._filter.doc_id = doc_id or ""

    def set_filter_tag(self, tag: str) -> None:
        self._filter.tag = tag or ""

    def set_filter_topic(self, topic: str) -> None:
        self._filter.topic = topic or ""

    def set_filter_text(self, text: str) -> None:
        self._filter.text = (text or "").strip().lower()

    def clear_filter(self) -> None:
        self._filter = KnowledgeFilter()

    # ------------------------------------------------------------------
    # Filtered view
    # ------------------------------------------------------------------

    def filtered_chunks(self) -> list[dict]:
        """Return chunks matching the active filter."""
        result = self._all_chunks
        f = self._filter

        if f.doc_id:
            result = [
                c for c in result
                if c["metadata"].get("doc_id") == f.doc_id
                or c["metadata"].get("page_id") == f.doc_id
            ]

        if f.tag:
            result = [
                c for c in result
                if f.tag in [
                    t.strip()
                    for t in str(c["metadata"].get("tags", "")).split(",")
                    if t.strip()
                ]
            ]

        if f.topic:
            result = [
                c for c in result
                if f.topic in [
                    t.strip()
                    for t in str(c["metadata"].get("topics", "")).split(",")
                    if t.strip()
                ]
            ]

        if f.text:
            needle = f.text
            result = [
                c for c in result
                if needle in c["content"].lower()
                or needle in str(c["metadata"].get("ka_summary", "")).lower()
                or needle in str(c["metadata"].get("ka_keywords", "")).lower()
                or needle in str(c["metadata"].get("page_title", "")).lower()
                or needle in str(c["metadata"].get("title", "")).lower()
            ]

        return result

    # ------------------------------------------------------------------
    # Facet helpers (for filter dropdowns)
    # ------------------------------------------------------------------

    def list_doc_title_map(self) -> dict[str, str]:
        return self._client.list_doc_title_map()

    def list_topics(self) -> list[str]:
        return self._client.list_field_values("topics")

    def list_tags(self) -> list[str]:
        return self._client.list_tags()

    def enrich_all(self, overwrite: bool = False) -> dict:
        """Run knowledge extraction for every indexed doc_id.

        Intended to be called in a background thread (slow LLM calls per chunk).
        Returns aggregated stats dict with keys ``enriched``, ``skipped``, ``failed``,
        ``docs_processed``, ``docs_failed``.
        """
        doc_ids = self._client.list_doc_ids()
        totals = {"enriched": 0, "skipped": 0, "failed": 0,
                  "docs_processed": 0, "docs_failed": 0}
        for doc_id in doc_ids:
            try:
                stats = self._client.enrich_doc(doc_id, overwrite=overwrite)
                totals["enriched"] += stats.get("enriched", 0)
                totals["skipped"]  += stats.get("skipped", 0)
                totals["failed"]   += stats.get("failed", 0)
                totals["docs_processed"] += 1
            except Exception as exc:
                log.error("enrich_all: failed for doc_id=%r: %s", doc_id, exc)
                totals["docs_failed"] += 1
        return totals
