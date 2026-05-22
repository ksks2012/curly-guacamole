"""
RAG Engine — answer_query pipeline.

Responsibilities:
  - Delegates retrieval to a RetrievalPipeline (query expansion, retrieve,
    fuse, rerank — all steps are plugin-able and swappable)
  - Builds context blocks from RetrievalResult list
  - Invokes LLM with RAG_PROMPT / MEMORY_AWARE_RAG_PROMPT
  - Updates ConversationMemory after each response

LocalLlamaClient holds a RAGEngine and delegates answer_query to it.
The engine has no knowledge of embeddings, Chroma, or indexing — it only
speaks the RetrievalPipeline / BaseRetriever interfaces.

Backward compatibility
----------------------
``RAGEngine`` still accepts either a ``RetrievalPipeline`` or any object
satisfying the ``BaseRetriever`` Protocol in its ``retriever`` parameter.
When a bare retriever is passed, a minimal DeduplicateStep-based pipeline
is created automatically at first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.prompt import (
    MEMORY_AWARE_RAG_PROMPT,
    RAG_PROMPT,
)
from rag.retrieval.base import RetrievalResult
from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from rag.memory.manager import ConversationMemory
    from rag.reranker import BaseReranker
    from rag.retrieval.base import BaseRetriever
    from rag.retrieval.pipeline import RetrievalPipeline
    from utils.config import AppConfig

log = AppLogger.get(__name__)


class RAGEngine:
    """
    Encapsulates the full retrieval-augmented generation pipeline.

    Args:
        llm       : ChatOpenAI instance used for generation.
        retriever : A ``RetrievalPipeline`` **or** any ``BaseRetriever``-
                    compatible object.  When a bare retriever is supplied a
                    default DeduplicateStep pipeline is built automatically.
        reranker  : Kept for backward compatibility; ignored when a full
                    ``RetrievalPipeline`` is supplied (the pipeline owns
                    its own RerankerStep if desired).
        config    : AppConfig — reads query_expansion_enabled/n.
        memory    : ConversationMemory (optional).
    """

    def __init__(
        self,
        llm: "ChatOpenAI",
        retriever: "RetrievalPipeline | BaseRetriever",
        reranker: "BaseReranker | None",
        config: "AppConfig",
        memory: "ConversationMemory | None" = None,
    ) -> None:
        self.llm      = llm
        self.retriever = retriever   # RetrievalPipeline or bare BaseRetriever
        self.reranker  = reranker
        self.config    = config
        self.memory    = memory
        self._pipeline: "RetrievalPipeline | None" = None   # lazily resolved

    def _resolve_pipeline(self) -> "RetrievalPipeline":
        """Return the active RetrievalPipeline, building one if needed."""
        from rag.retrieval.pipeline import RetrievalPipeline

        if isinstance(self.retriever, RetrievalPipeline):
            return self.retriever

        # Bare BaseRetriever — wrap in a minimal pipeline (build once, cache)
        if self._pipeline is None or getattr(self._pipeline, "_bare_source", None) is not self.retriever:
            from rag.retrieval.pipeline import PipelineBuilder
            use_expansion = self.config.query_expansion_enabled
            b = PipelineBuilder(self.retriever, name="engine_auto")
            if use_expansion:
                b.with_expansion(self.llm, n=self.config.query_expansion_n)
            b.with_deduplicate()
            if self.reranker:
                b.with_reranker(self.reranker)
            self._pipeline = b.build()
            self._pipeline._bare_source = self.retriever   # type: ignore[attr-defined]
        return self._pipeline

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        expand_query: bool | None = None,
        filters: dict | None = None,
    ):
        """Run the full RAG pipeline and return the LLM response.

        Pipeline steps are defined by the active RetrievalPipeline:
          [optional] QueryExpansionStep  → extra phrasings
           ↓
          RetrieveStep(s)               → per-query search
           ↓
          DeduplicateStep or RRFStep    → fuse + dedup
           ↓
          [optional] RerankerStep       → cross-encoder rerank
           ↓
          LLM generation

        Args:
            query        : user question.
            k            : number of results passed to the LLM.
            fetch_k      : candidate pool size fetched per query phrasing.
            doc_id       : convenience shortcut — adds ``{"doc_id": doc_id}`` to
                           *filters* when supplied and *filters* is None.
            expand_query : per-call override for config.query_expansion_enabled.
                           Only effective when using a bare BaseRetriever (not a
                           pre-built RetrievalPipeline).
            filters      : raw Chroma ``where`` dict forwarded to all retrieve steps.
        """
        # Build filters — doc_id is a convenience shortcut
        active_filters = filters
        if active_filters is None and doc_id is not None:
            active_filters = {"doc_id": doc_id}

        # Per-call expansion override only applies to auto-built pipelines.
        # Pre-built RetrievalPipelines define their own expansion step.
        skip_expansion = False
        from rag.retrieval.pipeline import RetrievalPipeline as _RP
        if not isinstance(self.retriever, _RP):
            use_expansion = (
                expand_query
                if expand_query is not None
                else self.config.query_expansion_enabled
            )
            skip_expansion = not use_expansion

        pipeline = self._resolve_pipeline()
        results = pipeline.run(
            query,
            top_k=k,
            fetch_k=fetch_k,
            filters=active_filters,
            skip_expansion=skip_expansion,
        )
        log.info("Pipeline %r returned %d results", pipeline.name, len(results))

        results = self._build_context_and_generate(query, results)
        return results

    def _build_context_and_generate(self, query: str, results: list[RetrievalResult]):
        """Build context blocks, construct prompt, invoke LLM, update memory."""
        # Build context blocks with source tags so the LLM can cite them.
        # Tag format depends on result origin and metadata:
        #   document/notion → [<title> / <section>]
        #   document/pdf    → [page <n>, <filename>]
        #   code/symbol     → [<file_path> :: <symbol_name>]
        #   code/block      → [<file_path>:<start_line>-<end_line>]
        #   others          → [<title>] or [<filename>]
        context_blocks = []
        for result in results:
            meta  = result.metadata
            src   = result.source
            if src == "code":
                file_path   = meta.get("file_path", "unknown")
                symbol_name = meta.get("symbol_name") or meta.get("name")
                start       = meta.get("start_line")
                end         = meta.get("end_line")
                if symbol_name:
                    tag = f"[{file_path} :: {symbol_name}]"
                elif start is not None:
                    tag = f"[{file_path}:{start}-{end}]"
                else:
                    tag = f"[{file_path}]"
            else:
                dtype = meta.get("document_type", "")
                if dtype == "notion":
                    title   = meta.get("title", "Notion")
                    section = meta.get("section", "")
                    tag = f"[{title} / {section}]" if section else f"[{title}]"
                else:
                    pg   = meta.get("page")
                    name = meta.get("filename") or meta.get("title", "unknown")
                    tag  = f"[page {pg + 1}, {name}]" if pg is not None else f"[{name}]"
            context_blocks.append(f"{tag}\n{result.content}")
        context = "\n\n".join(context_blocks)

        # Build prompt — inject memory context when available
        memory_block = ""
        if self.memory is not None:
            try:
                memory_block = self.memory.build_context_block()
            except Exception as exc:
                log.warning("memory.build_context_block() failed: %s", exc)

        if memory_block:
            prompt = MEMORY_AWARE_RAG_PROMPT.format(
                memory_context=memory_block,
                context=context,
                question=query,
            )
        else:
            prompt = RAG_PROMPT.format(context=context, question=query)

        response = self.llm.invoke(prompt)

        # Update memory after response (fail-safe — never blocks the answer)
        if self.memory is not None:
            try:
                answer_text = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )
                doc_ids = list(
                    dict.fromkeys(
                        r.metadata.get("doc_id", "")
                        for r in results
                        if r.metadata.get("doc_id")
                    )
                )
                self.memory.add_turn(query, answer_text, doc_ids=doc_ids)
            except Exception as exc:
                log.warning("memory.add_turn() failed: %s", exc)

        return response
