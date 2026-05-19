"""
Knowledge controller — logic layer for the Knowledge Card tab.

Responsibilities:
  - Browse all indexed chunks from Chroma with rich metadata
  - Maintain filter state (doc_id, keyword, entity, topic_cluster,
    ka_topic, text search)
  - Expose sorted/filtered chunk list to the UI layer

Chip-to-field mapping
---------------------
  Blue   (keyword)       → ka_keywords   — B.1 KnowledgeExtractor
  Purple (entity)        → ka_entities   — B.1 KnowledgeExtractor
  Green  (topic_cluster) → topic_id      — B.3 TopicClusterer (scalar)
  Teal   (ka_topic)      → ka_topics     — B.1 KnowledgeExtractor (CSV)

Has NO dependency on NiceGUI.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.logger import AppLogger
from rag.client import LocalLlamaClient
from rag.knowledge.clusterer import TopicMap  # noqa: F401 (type hint only)

log = AppLogger.get(__name__)


@dataclass
class KnowledgeFilter:
    doc_id:        str = ""
    keyword:       str = ""
    entity:        str = ""
    topic_cluster: str = ""   # matches topic_id (scalar, B.3)
    ka_topic:      str = ""   # matches ka_topics (CSV, B.1)
    text:          str = ""

    def is_empty(self) -> bool:
        return not (
            self.doc_id or self.keyword or self.entity
            or self.topic_cluster or self.ka_topic or self.text
        )

    def summary(self) -> str:
        parts = []
        if self.doc_id:
            parts.append(f"doc={self.doc_id[:20]}")
        if self.keyword:
            parts.append(f"kw={self.keyword}")
        if self.entity:
            parts.append(f"ent={self.entity}")
        if self.topic_cluster:
            parts.append(f"cluster={self.topic_cluster}")
        if self.ka_topic:
            parts.append(f"topic={self.ka_topic}")
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

    def set_filter_keyword(self, keyword: str) -> None:
        self._filter.keyword = keyword or ""

    def set_filter_entity(self, entity: str) -> None:
        self._filter.entity = entity or ""

    def set_filter_topic_cluster(self, topic: str) -> None:
        self._filter.topic_cluster = topic or ""

    def set_filter_ka_topic(self, topic: str) -> None:
        self._filter.ka_topic = topic or ""

    def set_filter_text(self, text: str) -> None:
        self._filter.text = (text or "").strip().lower()

    def clear_filter(self) -> None:
        self._filter = KnowledgeFilter()

    # ------------------------------------------------------------------
    # Filtered view
    # ------------------------------------------------------------------

    @staticmethod
    def _csv_values(raw: str) -> list[str]:
        return [t.strip() for t in str(raw or "").split(",") if t.strip()]

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

        if f.keyword:
            result = [
                c for c in result
                if f.keyword in self._csv_values(c["metadata"].get("ka_keywords", ""))
            ]

        if f.entity:
            result = [
                c for c in result
                if f.entity in self._csv_values(c["metadata"].get("ka_entities", ""))
            ]

        if f.topic_cluster:
            result = [
                c for c in result
                if c["metadata"].get("topic_id") == f.topic_cluster
            ]

        if f.ka_topic:
            result = [
                c for c in result
                if f.ka_topic in self._csv_values(c["metadata"].get("ka_topics", ""))
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

    def list_keywords(self) -> list[str]:
        return self._client.list_field_values("ka_keywords")

    def list_entities(self) -> list[str]:
        return self._client.list_field_values("ka_entities")

    def list_topic_clusters(self) -> list[str]:
        """topic_id — scalar written by B.3 TopicClusterer."""
        return self._client.list_field_values("topic_id")

    def list_ka_topics(self) -> list[str]:
        """ka_topics — CSV written by B.1 KnowledgeExtractor."""
        return self._client.list_field_values("ka_topics")

    def cluster_topics(self, n_clusters: int = 8) -> "TopicMap":
        """Run B.3 TopicClusterer and write topic_id into all Chroma chunks."""
        return self._client.cluster_topics(n_clusters=n_clusters)

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
