"""
KnowledgeManager — knowledge enrichment and QA operations extracted from LocalLlamaClient.

Owns the QA Chroma collection, QA indexer, extractor, and generator.
LocalLlamaClient holds a KnowledgeManager instance and delegates to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document

from rag.knowledge.extractor import KnowledgeExtractor
from rag.knowledge.qa_generator import QAGenerator
from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from rag.indexer import Indexer
    from rag.knowledge.clusterer import TopicClusterer, TopicMap

log = AppLogger.get(__name__)


class KnowledgeManager:
    """Enrichment, QA generation, and topic clustering for indexed documents.

    Args:
        db           : Main Chroma collection (read-only: fetches chunks by doc_id).
        qa_db        : Dedicated QA Chroma collection (read/write).
        qa_indexer   : Indexer wired to *qa_db*.
        extractor    : KnowledgeExtractor instance.
        qa_generator : QAGenerator instance.
        clusterer    : TopicClusterer instance (optional; needed for B.3).
    """

    def __init__(
        self,
        db:           "Chroma",
        qa_db:        "Chroma",
        qa_indexer:   "Indexer",
        extractor:    KnowledgeExtractor,
        qa_generator: QAGenerator,
        clusterer:    "TopicClusterer | None" = None,
    ) -> None:
        self._db           = db
        self._qa_db        = qa_db
        self._qa_indexer   = qa_indexer
        self.extractor     = extractor
        self.qa_generator  = qa_generator
        self.clusterer     = clusterer

    # ------------------------------------------------------------------
    # B.1 — knowledge extraction
    # ------------------------------------------------------------------

    def enrich_doc(self, doc_id: str, overwrite: bool = False) -> dict:
        """Run knowledge extraction on all chunks for *doc_id* in Chroma.

        Updates metadata in-place (no re-embedding).

        Returns:
            dict with keys ``enriched``, ``skipped``, ``failed``.
        """
        log.info("enrich_doc: doc_id=%r  overwrite=%s", doc_id, overwrite)

        result    = self._db.get(where={"doc_id": {"$eq": doc_id}}, include=["documents", "metadatas"])
        ids       = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not ids:
            log.warning("enrich_doc: no chunks found for doc_id=%r", doc_id)
            return {"enriched": 0, "skipped": 0, "failed": 0}

        stats = {"enriched": 0, "skipped": 0, "failed": 0}
        new_ids, new_metas = [], []

        for chroma_id, text, meta in zip(ids, documents, metadatas):
            meta = meta or {}
            if not overwrite and KnowledgeExtractor.is_enriched(meta):
                stats["skipped"] += 1
                log.debug("  skip (already enriched): %s", chroma_id)
                continue

            artifact = self.extractor.extract_one(text or "")
            if not artifact.get("ka_summary"):
                stats["failed"] += 1
                log.warning("  extraction produced no summary: %s", chroma_id)
                continue

            new_ids.append(chroma_id)
            new_metas.append({**meta, **artifact})
            stats["enriched"] += 1

        if new_ids:
            self._db._collection.update(ids=new_ids, metadatas=new_metas)
            log.info("enrich_doc done: enriched=%d  skipped=%d  failed=%d",
                     stats["enriched"], stats["skipped"], stats["failed"])
        return stats

    # ------------------------------------------------------------------
    # B.2 — QA generation
    # ------------------------------------------------------------------

    def generate_qa(self, doc_id: str, overwrite: bool = False) -> dict:
        """Generate and index QA pairs for all chunks of *doc_id*.

        Args:
            doc_id    : Document identifier.
            overwrite : When False (default), skip if QA pairs already exist.

        Returns:
            dict with keys ``generated``, ``indexed``, ``skipped``, ``failed``.
        """
        log.info("generate_qa: doc_id=%r  overwrite=%s", doc_id, overwrite)

        existing     = self._qa_db.get(where={"doc_id": {"$eq": doc_id}}, include=["metadatas"])
        existing_ids = existing.get("ids") or []

        if existing_ids and not overwrite:
            log.info("generate_qa: %d QA pairs exist, skipping (overwrite=False)", len(existing_ids))
            return {"generated": 0, "indexed": 0, "skipped": len(existing_ids), "failed": 0}

        if existing_ids:
            self._qa_db._collection.delete(where={"doc_id": {"$eq": doc_id}})
            log.info("generate_qa: deleted %d existing QA pairs", len(existing_ids))

        result    = self._db.get(where={"doc_id": {"$eq": doc_id}}, include=["documents", "metadatas"])
        ids       = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not ids:
            log.warning("generate_qa: no chunks found for doc_id=%r", doc_id)
            return {"generated": 0, "indexed": 0, "skipped": 0, "failed": 0}

        source_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(documents, metadatas)
            if text
        ]

        pairs = self.qa_generator.generate_for_docs(source_docs)
        if not pairs:
            return {"generated": 0, "indexed": 0, "skipped": 0, "failed": len(source_docs)}

        stats   = self._qa_indexer.run([p.to_document() for p in pairs])
        indexed = stats.get("num_added", 0) + stats.get("num_updated", 0)
        log.info("generate_qa done: generated=%d  indexed=%d", len(pairs), indexed)
        return {"generated": len(pairs), "indexed": indexed, "skipped": 0, "failed": 0}

    def qa_search(self, query: str, k: int = 5) -> list[dict]:
        """Search the QA index for questions matching *query*.

        Returns a ranked list of QA pairs with keys: ``question``, ``answer``,
        ``chunk_id``, ``doc_id``, ``score`` (float 0-1).
        """
        raw = self._qa_db.similarity_search_with_score(query, k=k)
        return [
            {
                "question": doc.page_content,
                "answer":   doc.metadata.get("answer",   ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "doc_id":   doc.metadata.get("doc_id",   ""),
                "score":    round(1 / (1 + dist), 4),
            }
            for doc, dist in raw
        ]

    # ------------------------------------------------------------------
    # B.3 — topic clustering
    # ------------------------------------------------------------------

    def cluster_topics(
        self, n_clusters: int = 8, doc_id: str | None = None
    ) -> "TopicMap":
        """Cluster chunks by embedding and label each cluster with a topic_id.

        Writes ``topic_id`` into Chroma metadata for every processed chunk.

        Args:
            n_clusters : Number of KMeans clusters (topics).
            doc_id     : If given, cluster only chunks from this document.

        Returns:
            TopicMap with cluster_labels and chunk_topics.

        Raises:
            RuntimeError: If the client was built without a TopicClusterer.
        """
        if self.clusterer is None:
            raise RuntimeError(
                "cluster_topics requires a TopicClusterer — "
                "pass clusterer= when constructing KnowledgeManager"
            )
        return self.clusterer.fit_and_assign(n_clusters=n_clusters, doc_id=doc_id)
