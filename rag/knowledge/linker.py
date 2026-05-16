"""
B.4 Cross-document Linking — semantic relation discovery.

Builds two levels of links using cosine similarity over stored Chroma embeddings:

  1. Chunk-level  (related_chunks):
     For each chunk, find the top-K most similar chunks that belong to a
     *different* document.  Results are written back into Chroma metadata so
     every retrieval result can carry "see also" links at zero query-time cost.

  2. Page-level  (related_pages):
     For each doc_id, compute a centroid embedding (mean of all its chunk
     embeddings).  Find the top-K most similar doc_ids.  Write those into
     every chunk of the document so page-level navigation is available.

Storage convention (Chroma metadata, all JSON-encoded strings):
  related_chunk_ids    : '["id1", "id2", ...]'
  related_chunk_scores : '[0.91, 0.87, ...]'
  related_doc_ids      : '["doc_a", "doc_b", ...]'
  related_doc_scores   : '[0.88, 0.82, ...]'

Note: Chroma metadata values must be scalars (str/int/float/bool).
Lists are serialised with json.dumps and deserialised on read.

Usage
-----
    linker = CrossDocLinker(db=client.db)

    # Link everything
    stats = linker.link_chunks()
    stats = linker.link_pages()

    # Scoped re-link after ingesting a new document
    stats = linker.link_chunks(doc_id="my_new_doc")

    # Read links at retrieval time
    related = linker.get_related_chunks("chroma-id-123")
    related = linker.get_related_pages("my_doc")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_chroma import Chroma

log = AppLogger.get(__name__)

# Metadata keys
_KEY_CHUNK_IDS    = "related_chunk_ids"
_KEY_CHUNK_SCORES = "related_chunk_scores"
_KEY_DOC_IDS      = "related_doc_ids"
_KEY_DOC_SCORES   = "related_doc_scores"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class LinkStats:
    """Summary returned by link_chunks() and link_pages()."""
    linked:  int = 0
    skipped: int = 0

    def __repr__(self) -> str:
        return f"LinkStats(linked={self.linked}, skipped={self.skipped})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Return the (N, N) pairwise cosine-similarity matrix for row vectors in X.

    Values are in [-1, 1]; higher = more similar.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    X_norm = X / norms
    return (X_norm @ X_norm.T).astype(np.float32)


def _top_k_exclude_self(
    sim_row: np.ndarray,
    idx_self: int,
    exclude_mask: np.ndarray,  # boolean array; True = exclude
    top_k: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of the top-K entries in *sim_row* after masking."""
    mask = exclude_mask.copy()
    mask[idx_self] = True                         # always exclude self
    sim_row = sim_row.copy()
    sim_row[mask] = -2.0                          # push masked entries below threshold
    above = np.where(sim_row >= threshold)[0]
    if len(above) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    scores_above = sim_row[above]
    order = np.argsort(scores_above)[::-1][:top_k]
    chosen = above[order]
    return chosen, sim_row[chosen]


# ---------------------------------------------------------------------------
# CrossDocLinker
# ---------------------------------------------------------------------------

class CrossDocLinker:
    """Builds and queries semantic links between chunks and documents.

    Args
    ----
    db : Main Chroma collection (read + write via ``_collection.update``).
    """

    def __init__(self, db: "Chroma") -> None:
        self._db = db

    # ------------------------------------------------------------------
    # B.4.1 — chunk-level linking
    # ------------------------------------------------------------------

    def link_chunks(
        self,
        top_k:     int   = 5,
        threshold: float = 0.75,
        doc_id:    str | None = None,
    ) -> LinkStats:
        """Find top-K cross-document similar chunks and write links to metadata.

        Args
        ----
        top_k     : Number of related chunks to store per chunk.
        threshold : Minimum cosine similarity to include a link.
        doc_id    : When given, only *source* chunks from this document are
                    re-linked.  Target chunks are still drawn from all docs.

        Returns
        -------
        LinkStats with counts of linked and skipped chunks.
        """
        # Fetch ALL embeddings for target candidates
        all_result = self._db.get(include=["embeddings", "documents", "metadatas"])
        all_ids    = all_result.get("ids") or []
        raw_emb    = all_result.get("embeddings")
        all_emb    = raw_emb if raw_emb is not None else []
        all_docs   = all_result.get("documents") or []
        all_metas  = all_result.get("metadatas") or []

        if not all_ids:
            log.warning("link_chunks: no chunks in collection")
            return LinkStats()

        all_ids    = list(all_ids)
        all_doc_ids = [
            (m or {}).get("doc_id", "") for m in all_metas
        ]

        X = np.array(all_emb, dtype=np.float32)
        sim = _cosine_matrix(X)               # (N, N)

        # Determine which source indices to process
        if doc_id:
            source_indices = [i for i, d in enumerate(all_doc_ids) if d == doc_id]
        else:
            source_indices = list(range(len(all_ids)))

        if not source_indices:
            log.warning("link_chunks: no chunks found for doc_id=%r", doc_id)
            return LinkStats()

        stats        = LinkStats()
        update_ids   : list[str] = []
        update_metas : list[dict] = []

        for i in source_indices:
            src_doc = all_doc_ids[i]
            # Exclude all chunks from the *same* document
            same_doc_mask = np.array(
                [d == src_doc for d in all_doc_ids], dtype=bool
            )
            chosen_idx, chosen_scores = _top_k_exclude_self(
                sim[i], i, same_doc_mask, top_k, threshold
            )

            meta = dict(all_metas[i] or {})

            if len(chosen_idx) == 0:
                stats.skipped += 1
            else:
                meta[_KEY_CHUNK_IDS]    = json.dumps([all_ids[j]    for j in chosen_idx])
                meta[_KEY_CHUNK_SCORES] = json.dumps([round(float(s), 4) for s in chosen_scores])
                stats.linked += 1

            update_ids.append(all_ids[i])
            update_metas.append(meta)

        self._db._collection.update(ids=update_ids, metadatas=update_metas)
        log.info(
            "link_chunks: linked=%d  skipped=%d  (doc_id=%r)",
            stats.linked, stats.skipped, doc_id,
        )
        return stats

    # ------------------------------------------------------------------
    # B.4.2 — page-level linking
    # ------------------------------------------------------------------

    def link_pages(
        self,
        top_k:     int   = 5,
        threshold: float = 0.70,
    ) -> LinkStats:
        """Find top-K related documents via centroid similarity.

        Writes ``related_doc_ids`` and ``related_doc_scores`` into every
        chunk of each document so callers can navigate at the page level.

        Returns
        -------
        LinkStats: ``linked`` = number of doc_ids that received at least one link.
        """
        all_result = self._db.get(include=["embeddings", "metadatas"])
        all_ids    = all_result.get("ids") or []
        raw_emb    = all_result.get("embeddings")
        all_emb    = raw_emb if raw_emb is not None else []
        all_metas  = all_result.get("metadatas") or []

        if not all_ids:
            log.warning("link_pages: no chunks in collection")
            return LinkStats()

        all_ids   = list(all_ids)
        all_doc_ids = [
            (m or {}).get("doc_id", "") for m in all_metas
        ]

        X = np.array(all_emb, dtype=np.float32)

        # Build centroid per doc_id
        unique_docs: list[str] = []
        seen: set[str] = set()
        for d in all_doc_ids:
            if d and d not in seen:
                unique_docs.append(d)
                seen.add(d)

        if len(unique_docs) < 2:
            log.info("link_pages: fewer than 2 documents — no cross-doc links possible")
            return LinkStats()

        centroids: list[np.ndarray] = []
        doc_chunk_indices: dict[str, list[int]] = {d: [] for d in unique_docs}
        for idx, d in enumerate(all_doc_ids):
            if d in doc_chunk_indices:
                doc_chunk_indices[d].append(idx)

        for d in unique_docs:
            idxs = doc_chunk_indices[d]
            centroids.append(X[idxs].mean(axis=0))

        C = np.stack(centroids, axis=0)           # (D, dim)
        doc_sim = _cosine_matrix(C)               # (D, D)

        stats             = LinkStats()
        # Collect (chunk_id → updated metadata) for bulk write
        update_ids  : list[str] = []
        update_metas: list[dict] = []
        # Pre-load current metadatas for merge
        meta_by_id = {cid: dict(m or {}) for cid, m in zip(all_ids, all_metas)}

        for di, src_doc in enumerate(unique_docs):
            # No same-doc mask needed — we work at the doc level
            same_doc_mask = np.zeros(len(unique_docs), dtype=bool)
            chosen_di, chosen_scores = _top_k_exclude_self(
                doc_sim[di], di, same_doc_mask, top_k, threshold
            )

            if len(chosen_di) > 0:
                rel_docs   = json.dumps([unique_docs[j]    for j in chosen_di])
                rel_scores = json.dumps([round(float(s), 4) for s in chosen_scores])
                stats.linked += 1
            else:
                rel_docs   = json.dumps([])
                rel_scores = json.dumps([])
                stats.skipped += 1

            for cidx in doc_chunk_indices[src_doc]:
                m = meta_by_id[all_ids[cidx]]
                m[_KEY_DOC_IDS]    = rel_docs
                m[_KEY_DOC_SCORES] = rel_scores
                update_ids.append(all_ids[cidx])
                update_metas.append(m)

        self._db._collection.update(ids=update_ids, metadatas=update_metas)
        log.info(
            "link_pages: %d docs  linked=%d  skipped=%d",
            len(unique_docs), stats.linked, stats.skipped,
        )
        return stats

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_related_chunks(self, chunk_id: str) -> list[dict]:
        """Return related chunks stored in *chunk_id*'s metadata.

        Returns
        -------
        List of dicts with keys: ``id``, ``doc_id``, ``score``, ``text``.
        Empty list if no links have been computed yet or chunk not found.
        """
        result = self._db.get(
            ids=[chunk_id], include=["metadatas", "documents"]
        )
        ids       = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        if not ids or not metadatas:
            return []

        meta = metadatas[0] or {}
        try:
            related_ids    = json.loads(meta.get(_KEY_CHUNK_IDS,    "[]"))
            related_scores = json.loads(meta.get(_KEY_CHUNK_SCORES, "[]"))
        except (json.JSONDecodeError, TypeError):
            return []

        if not related_ids:
            return []

        # Fetch text + metadata of related chunks
        rel_result = self._db.get(
            ids=related_ids, include=["documents", "metadatas"]
        )
        rel_ids    = rel_result.get("ids")       or []
        rel_docs   = rel_result.get("documents") or []
        rel_metas  = rel_result.get("metadatas") or []

        score_map = dict(zip(related_ids, related_scores))
        return [
            {
                "id":    cid,
                "doc_id": (m or {}).get("doc_id", ""),
                "score":  score_map.get(cid, 0.0),
                "text":   (text or "")[:200],
            }
            for cid, text, m in zip(rel_ids, rel_docs, rel_metas)
        ]

    def get_related_pages(self, doc_id: str) -> list[dict]:
        """Return related documents for *doc_id* using stored page-level links.

        Reads the link metadata from the first available chunk of the document.

        Returns
        -------
        List of dicts with keys: ``doc_id``, ``score``.
        Empty list if no links have been computed yet or doc not found.
        """
        result = self._db.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["metadatas"],
        )
        ids       = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        if not ids:
            return []

        # Any chunk carries the same page-level links; use the first one
        meta = metadatas[0] or {}
        try:
            rel_docs   = json.loads(meta.get(_KEY_DOC_IDS,    "[]"))
            rel_scores = json.loads(meta.get(_KEY_DOC_SCORES, "[]"))
        except (json.JSONDecodeError, TypeError):
            return []

        return [
            {"doc_id": d, "score": s}
            for d, s in zip(rel_docs, rel_scores)
        ]
