"""rag/retrieval — unified retrieval stack.

Unified interface (Phase 1):
    base                : RetrievalResult, BaseRetriever Protocol
    document_retriever  : DocumentRetriever (wraps Searcher)
    code_retriever      : CodeRetriever (wraps CodeIndexer)
    hierarchical_code_retriever : HierarchicalCodeRetriever (RepoIndex + CodeRetriever)
    related_code_retriever : RelatedCodeRetriever (relation-enriched code blocks)
    hybrid_retriever    : HybridRetriever (RRF fusion of multiple retrievers)

Pipeline (Step 1.4):
    pipeline            : RetrievalPipeline, PipelineBuilder,
                          PipelineContext, PipelineStep
    pipeline.steps      : QueryExpansionStep, RetrieveStep, DeduplicateStep,
                          RRFStep, RerankerStep

Support modules:
    filters             : SearchFilter — Chroma where-clause builder
    searcher            : Searcher — document vector/BM25/hybrid search
    bm25                : BM25Index, rrf_fuse
"""

from rag.retrieval.base import BaseRetriever, RetrievalResult
from rag.retrieval.code_result_filter import CodeResultFilter
from rag.retrieval.document_retriever import DocumentRetriever
from rag.retrieval.code_retriever import CodeRetriever
from rag.retrieval.hierarchical_code_retriever import HierarchicalCodeRetriever
from rag.retrieval.related_code_retriever import RelatedCodeRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.pipeline import (
    PipelineBuilder,
    PipelineContext,
    PipelineStep,
    RetrievalPipeline,
)

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "CodeResultFilter",
    "DocumentRetriever",
    "CodeRetriever",
    "HierarchicalCodeRetriever",
    "RelatedCodeRetriever",
    "HybridRetriever",
    "PipelineContext",
    "PipelineStep",
    "RetrievalPipeline",
    "PipelineBuilder",
]
