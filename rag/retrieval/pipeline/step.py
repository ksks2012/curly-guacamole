"""PipelineStep Protocol — the contract every pipeline step must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.retrieval.pipeline.context import PipelineContext


@runtime_checkable
class PipelineStep(Protocol):
    """A single processing step in a RetrievalPipeline.

    Steps mutate the shared ``PipelineContext`` in place and return it so
    pipelines can be composed easily:

        for step in self.steps:
            ctx = step.run(ctx)

    Concrete implementations live in ``rag.retrieval.pipeline.steps``:
        QueryExpansionStep — expand ctx.queries via LLM
        RetrieveStep       — search a BaseRetriever for every query
        DeduplicateStep    — flatten result_lists, dedup → ctx.candidates
        RRFStep            — RRF-fuse result_lists → ctx.candidates
        RerankerStep       — rerank ctx.candidates → ctx.results
    """

    @property
    def name(self) -> str:
        """Human-readable step identifier used in logging."""
        ...

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute the step on *ctx* and return the (mutated) context."""
        ...
