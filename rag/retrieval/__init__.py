"""rag/retrieval — unified retrieval stack.

Unified interface (Phase 1):
    base                : RetrievalResult, BaseRetriever Protocol
    document_retriever  : DocumentRetriever (wraps Searcher)
    code_retriever      : CodeRetriever (wraps CodeIndexer)
    hybrid_retriever    : HybridRetriever (RRF fusion of multiple retrievers)

Support modules:
    filters             : SearchFilter — Chroma where-clause builder
    searcher            : Searcher — document vector/BM25/hybrid search
    bm25                : BM25Index, rrf_fuse
"""

from rag.retrieval.base import BaseRetriever, RetrievalResult
from rag.retrieval.document_retriever import DocumentRetriever
from rag.retrieval.code_retriever import CodeRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "DocumentRetriever",
    "CodeRetriever",
    "HybridRetriever",
]
