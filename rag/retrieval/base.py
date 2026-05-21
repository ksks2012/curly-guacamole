"""
Unified Retrieval Interface — Phase 1.

Defines the shared contract that DocumentRetriever, CodeRetriever, and
HybridRetriever must satisfy.  By programming to this interface, higher-level
components (reranker, query expansion, RAGEngine) can work with both document
and code retrieval without knowing which backend is active.

Usage
-----
>>> from rag.retrieval.base import BaseRetriever, RetrievalResult
>>> from rag.retrieval.document_retriever import DocumentRetriever
>>> from rag.retrieval.code_retriever import CodeRetriever
>>> from rag.retrieval.hybrid_retriever import HybridRetriever
>>>
>>> doc_ret  = DocumentRetriever(searcher, use_hybrid=True)
>>> code_ret = CodeRetriever(code_indexer, level="symbol")
>>> hybrid   = HybridRetriever([doc_ret, code_ret])
>>>
>>> results = hybrid.search("How does authentication work?", top_k=10)
>>> for r in results:
...     print(r.source, r.score, r.content[:80])

Design notes
------------
- ``BaseRetriever`` is a ``runtime_checkable`` Protocol, so
  ``isinstance(x, BaseRetriever)`` works without formal inheritance.
- ``RetrievalResult`` is a plain dataclass — cheap to create, easy to test,
  serialisable.
- ``filters`` is always a raw Chroma ``where``-dict.  Each concrete retriever
  translates it to its own backend's format.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single result returned by any retrieval backend.

    Attributes
    ----------
    content  : The text content to pass to the LLM.
    score    : Relevance score — higher means more relevant.
               Vector-search results use ``1 / (1 + L2_distance)``.
               BM25 and RRF scores are raw floats.
    source   : Backend label: ``"document"`` or ``"code"``.
    metadata : Original Chroma metadata dict, passed through unchanged.
               Useful for provenance display and post-filtering.
    """

    content:  str
    score:    float
    source:   Literal["document", "code"]
    metadata: dict = field(default_factory=dict)

    def unique_key(self) -> str:
        """Stable deduplication key for RRF merging.

        Prefers ``chunk_id`` (documents) or ``symbol_id`` (code) from metadata.
        Falls back to a SHA-256 prefix of the content when neither is present.
        """
        cid = (
            self.metadata.get("chunk_id")
            or self.metadata.get("symbol_id")
            or self.metadata.get("snapshot_id")
        )
        if cid:
            return str(cid)
        return hashlib.sha256(self.content[:400].encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# BaseRetriever Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseRetriever(Protocol):
    """Minimal interface that every retrieval backend must satisfy.

    Concrete implementations:
    - :class:`~rag.retrieval.document_retriever.DocumentRetriever`
    - :class:`~rag.retrieval.code_retriever.CodeRetriever`
    - :class:`~rag.retrieval.hybrid_retriever.HybridRetriever`

    The Protocol is ``runtime_checkable`` so you can write:
    ``assert isinstance(retriever, BaseRetriever)``
    in tests without importing the concrete class.
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Return up to *top_k* results sorted best-first (highest score first).

        Parameters
        ----------
        query   : Natural-language query string.
        top_k   : Maximum number of results to return.
        filters : Optional Chroma-style ``where`` dict applied server-side.
                  Pass ``None`` to disable filtering.

        Returns
        -------
        list[RetrievalResult] sorted descending by score.
        """
        ...
