"""rag/retrieval/pipeline — pluggable retrieval pipeline.

Each pipeline step satisfies the PipelineStep Protocol and operates on a
shared PipelineContext object.  Steps are composed via RetrievalPipeline and
assembled using PipelineBuilder.

Public surface
--------------
    PipelineContext    : shared state passed between steps
    PipelineStep       : runtime-checkable Protocol (step contract)
    RetrievalPipeline  : ordered sequence of steps; run() returns list[RetrievalResult]
    PipelineBuilder    : fluent builder + factory class methods

Step catalogue (rag.retrieval.pipeline.steps):
    QueryExpansionStep : LLM-based query expansion
    RetrieveStep       : calls a BaseRetriever for every query in ctx.queries
    DeduplicateStep    : flattens result_lists, dedup by unique_key()
    RRFStep            : Reciprocal Rank Fusion across result_lists
    RerankerStep       : cross-encoder rerank of ctx.candidates → ctx.results
"""

from rag.retrieval.pipeline.context import PipelineContext
from rag.retrieval.pipeline.step import PipelineStep
from rag.retrieval.pipeline.pipeline import RetrievalPipeline, PipelineBuilder

__all__ = [
    "PipelineContext",
    "PipelineStep",
    "RetrievalPipeline",
    "PipelineBuilder",
]
