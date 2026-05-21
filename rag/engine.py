"""
RAG Engine — answer_query pipeline.

Responsibilities:
  - Query expansion via LLM (optional)
  - Multi-query retrieval through a BaseRetriever (document, code, or unified)
  - Chunk-level deduplication via RetrievalResult.unique_key()
  - Reranking (delegates to BaseReranker on RetrievalResult list)
  - Prompt construction and LLM invocation

LocalLlamaClient holds a RAGEngine and delegates answer_query to it.
The engine has no knowledge of embeddings, Chroma, or indexing — it only
speaks the BaseRetriever Protocol defined in rag.retrieval.base.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rag.prompt import (
    MEMORY_AWARE_RAG_PROMPT,
    QUERY_EXPANSION_PROMPT,
    RAG_PROMPT,
)
from rag.retrieval.base import RetrievalResult
from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from rag.memory.manager import ConversationMemory
    from rag.reranker import BaseReranker
    from rag.retrieval.base import BaseRetriever
    from utils.config import AppConfig

log = AppLogger.get(__name__)


class RAGEngine:
    """
    Encapsulates the full retrieval-augmented generation pipeline.

    Args:
        llm       : ChatOpenAI instance used for expansion and generation.
        retriever : Any object satisfying the BaseRetriever Protocol
                    (DocumentRetriever, CodeRetriever, HybridRetriever, …).
                    The engine only calls ``retriever.search(query, top_k, filters)``.
        reranker  : BaseReranker or None — applied to RetrievalResult list.
        config    : AppConfig — reads query_expansion_enabled/n.
        memory    : ConversationMemory (optional). When supplied, injects
                    a context block into the prompt and records every
                    Q-A turn after the response is returned.
    """

    def __init__(
        self,
        llm: "ChatOpenAI",
        retriever: "BaseRetriever",
        reranker: "BaseReranker | None",
        config: "AppConfig",
        memory: "ConversationMemory | None" = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.config = config
        self.memory = memory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expand_query(self, query: str, n: int) -> list[str]:
        """Return n alternative phrasings of query.

        Falls back to an empty list so the caller always gets a valid list.
        """
        prompt = QUERY_EXPANSION_PROMPT.format(question=query, n=n)
        try:
            response = self.llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            expanded = json.loads(raw.strip())
            if isinstance(expanded, list):
                return [str(q) for q in expanded]
        except Exception as e:
            log.warning("query_expansion failed, using original query only: %s", e)
        return []

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

        Pipeline:
          [optional] query expansion  → N extra phrasings via LLM
           ↓
          retriever.search() per phrasing  (fetch_k candidates each)
           ↓
          de-duplicate by RetrievalResult.unique_key()
           ↓
          reranker                         (narrows down to k results when enabled)
           ↓
          LLM generation

        Args:
            query        : user question.
            k            : number of results passed to the LLM.
            fetch_k      : candidate pool size fetched per query phrasing.
            doc_id       : convenience shortcut — adds ``{"doc_id": doc_id}`` to
                           *filters* when supplied.  Ignored when *filters* is
                           already set.
            expand_query : per-call override for config.query_expansion_enabled.
            filters      : raw Chroma ``where`` dict forwarded to the retriever.
        """
        use_expansion = (
            expand_query
            if expand_query is not None
            else self.config.query_expansion_enabled
        )

        if use_expansion:
            extra_queries = self._expand_query(query, n=self.config.query_expansion_n)
            log.info("Query expansion: %d extra queries", len(extra_queries))
        else:
            extra_queries = []

        all_queries = [query] + extra_queries

        # Build filters dict — doc_id is a convenience shortcut
        active_filters = filters
        if active_filters is None and doc_id is not None:
            active_filters = {"doc_id": doc_id}

        # Retrieve candidates for every phrasing, de-duplicate by unique_key()
        seen_keys: set[str] = set()
        candidates: list[RetrievalResult] = []
        for q in all_queries:
            for result in self.retriever.search(q, top_k=fetch_k, filters=active_filters):
                key = result.unique_key()
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(result)

        log.info("Candidates after dedup: %d", len(candidates))

        # Rerank then keep top k
        # BaseReranker operates on LangChain Documents — wrap/unwrap around it.
        if self.reranker is not None and candidates:
            from langchain_core.documents import Document as LCDoc
            lc_docs = [LCDoc(page_content=r.content, metadata=r.metadata) for r in candidates]
            reranked_docs = self.reranker.rerank(query, lc_docs, top_k=k)
            # Reconstruct RetrievalResult list preserving source; score is no
            # longer meaningful after reranking so we use rank-based 1/(1+i).
            key_to_result = {r.unique_key(): r for r in candidates}
            results: list[RetrievalResult] = []
            for i, doc in enumerate(reranked_docs):
                dummy = RetrievalResult(content=doc.page_content, score=0.0,
                                        source="document", metadata=doc.metadata)
                matched = key_to_result.get(dummy.unique_key())
                if matched:
                    results.append(RetrievalResult(
                        content=matched.content,
                        score=1.0 / (1.0 + i),
                        source=matched.source,
                        metadata=matched.metadata,
                    ))
                else:
                    results.append(RetrievalResult(
                        content=doc.page_content,
                        score=1.0 / (1.0 + i),
                        source="document",
                        metadata=doc.metadata,
                    ))
        else:
            results = candidates[:k]

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
                # Collect unique doc_ids from retrieved chunks (for C.3 timeline)
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
