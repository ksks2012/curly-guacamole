"""BM25 in-memory retriever and Reciprocal Rank Fusion (RRF) for hybrid search.

Design
------
* ``BM25Index`` builds a BM25Okapi index from a list of LangChain ``Document``
  objects.  The index lives in memory and is rebuilt whenever the document
  corpus changes (triggered via ``LocalLlamaClient.invalidate_bm25()``).
* ``rrf_fuse`` merges vector-search results with BM25 results using Reciprocal
  Rank Fusion — a parameter-free, rank-based merging strategy that is robust
  to score-scale differences between the two retrieval systems.

Tokenisation
------------
Simple alphanumeric + underscore extraction, lowercased.  Works well for
prose, code identifiers, numbers, and CJK is treated as full characters
(rank_bm25 handles unicode code-points correctly in BM25Okapi).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from utils.logger import AppLogger

if TYPE_CHECKING:
    pass  # avoid circular import in type hints

log = AppLogger.get(__name__)

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\s\u3000-\u303f\uff00-\uffef]+")
# Splits on whitespace and common CJK punctuation; keeps CJK characters together.


def _tokenize(text: str) -> list[str]:
    """Tokenise *text* into lowercase terms for BM25 indexing."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Metadata matcher (mirrors Chroma where-clause operators)
# ---------------------------------------------------------------------------

def _match(metadata: dict, where: dict) -> bool:
    """Return True if *metadata* satisfies the Chroma-style *where* clause.

    Supported operators: ``$and``, ``$eq``, ``$contains``, ``$gte``, ``$lte``.
    Unknown operators are ignored (conservative: treated as matching).
    """
    if "$and" in where:
        return all(_match(metadata, cond) for cond in where["$and"])

    for field, condition in where.items():
        val = str(metadata.get(field, ""))
        if "$eq" in condition:
            if val != str(condition["$eq"]):
                return False
        if "$contains" in condition:
            if str(condition["$contains"]) not in val:
                return False
        if "$gte" in condition:
            if val < str(condition["$gte"]):
                return False
        if "$lte" in condition:
            if val > str(condition["$lte"]):
                return False
    return True


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """In-memory BM25 index over a corpus of LangChain ``Document`` objects.

    Usage::

        idx = BM25Index()
        idx.build(docs)          # (re)build from a list[Document]
        results = idx.search(query, k=20, where={"workspace": {"$eq": "work"}})
    """

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._bm25 = None  # BM25Okapi instance, None until build() called

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, docs: list[Document]) -> None:
        """(Re)build the BM25 index from *docs*.

        This is the only time-consuming step (~milliseconds for typical corpora).
        Must be called again whenever the document corpus changes.
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for hybrid search: pip install rank-bm25"
            ) from exc

        self._docs = list(docs)
        if not self._docs:
            self._bm25 = None
            log.debug("BM25Index.build: empty corpus — index cleared")
            return

        tokenized = [_tokenize(doc.page_content) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        log.debug("BM25Index.build: %d documents indexed", len(self._docs))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 20,
        where: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """Return up to *k* (Document, normalised_score) pairs sorted best-first.

        Args:
            query: Query string.
            k: Maximum number of results.
            where: Optional Chroma-style metadata filter applied **before** BM25
                   scoring.  Only docs that pass the filter are considered.

        Returns:
            List of ``(Document, score)`` where score is the raw BM25 score
            (non-negative float; higher = more relevant).
            Docs with BM25 score == 0 are excluded.
        """
        if self._bm25 is None or not self._docs:
            return []

        # Apply metadata pre-filter
        if where:
            indices = [
                i for i, doc in enumerate(self._docs)
                if _match(doc.metadata, where)
            ]
        else:
            indices = list(range(len(self._docs)))

        if not indices:
            return []

        tokens = _tokenize(query)
        all_scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            ((i, float(all_scores[i])) for i in indices if all_scores[i] > 0),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        return [(self._docs[i], score) for i, score in ranked]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of documents in the index."""
        return len(self._docs)

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rrf_fuse(
    vector: list[tuple[Document, float]],
    bm25: list[tuple[Document, float]],
    top_k: int,
    rrf_k: int = 60,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[tuple[Document, float]]:
    """Merge vector and BM25 results using Reciprocal Rank Fusion.

    Formula (per result list *i* with weight *w_i*)::

        score(d) = Σ_i  w_i / (rrf_k + rank_i(d))

    where ``rank_i(d)`` is 1-based rank in result list *i*.  Documents absent
    from a list receive no contribution from that list.

    Deduplication is performed on ``chunk_id`` metadata (falls back to
    ``page_content`` hash if ``chunk_id`` is absent).

    Args:
        vector: Vector-search results sorted best-first.
        bm25:   BM25-search results sorted best-first.
        top_k:  Number of results to return.
        rrf_k:  RRF dampening constant (default 60 — standard literature value).
        vector_weight: Weight applied to vector list contribution.
        bm25_weight:   Weight applied to BM25 list contribution.

    Returns:
        Up to *top_k* ``(Document, rrf_score)`` pairs sorted best-first.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}

    def _key(doc: Document) -> str:
        return str(doc.metadata.get("chunk_id") or hash(doc.page_content[:200]))

    for rank, (doc, _) in enumerate(vector, start=1):
        key = _key(doc)
        scores[key] = scores.get(key, 0.0) + vector_weight / (rrf_k + rank)
        docs[key] = doc

    for rank, (doc, _) in enumerate(bm25, start=1):
        key = _key(doc)
        scores[key] = scores.get(key, 0.0) + bm25_weight / (rrf_k + rank)
        if key not in docs:
            docs[key] = doc

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(docs[key], round(score, 6)) for key, score in fused]
