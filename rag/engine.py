"""
RAG Engine — answer_query pipeline.

Responsibilities:
  - Query expansion via LLM (optional)
  - Multi-query retrieval with chunk-level deduplication
  - Reranking (delegates to BaseReranker)
  - Prompt construction and LLM invocation

LocalLlamaClient holds a RAGEngine and delegates answer_query to it.
The engine has no knowledge of embeddings, Chroma, or indexing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

from langchain_core.documents import Document

from rag.prompt import RAG_PROMPT, QUERY_EXPANSION_PROMPT
from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI
    from rag.reranker import BaseReranker
    from utils.config import AppConfig

log = AppLogger.get(__name__)


class RAGEngine:
    """
    Encapsulates the full retrieval-augmented generation pipeline.

    Args:
        llm           : ChatOpenAI instance used for expansion and generation.
        get_retriever : Callable(k, fetch_k, doc_id) → LangChain retriever.
                        Provided by LocalLlamaClient so the engine stays
                        decoupled from the Chroma store.
        reranker      : BaseReranker or None.
        config        : AppConfig — reads query_expansion_enabled/n.
    """

    def __init__(
        self,
        llm: "ChatOpenAI",
        get_retriever: Callable,
        reranker: "BaseReranker | None",
        config: "AppConfig",
    ) -> None:
        self.llm = llm
        self.get_retriever = get_retriever
        self.reranker = reranker
        self.config = config

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
    ):
        """Run the full RAG pipeline and return the LLM response.

        Pipeline:
          [optional] query expansion  → N extra phrasings via LLM
           ↓
          vector search per phrasing  (fetch_k candidates each, via MMR)
           ↓
          de-duplicate by chunk_id
           ↓
          reranker                    (narrows down to k docs when enabled)
           ↓
          LLM generation

        Args:
            query        : user question.
            k            : number of docs passed to the LLM.
            fetch_k      : MMR candidate pool size per query.
            doc_id       : optional metadata filter for retrieval.
            expand_query : per-call override for config.query_expansion_enabled.
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

        # Retrieve candidates for every phrasing and de-duplicate by chunk_id
        seen_ids: set = set()
        candidates: list[Document] = []
        retriever = self.get_retriever(k=fetch_k, fetch_k=fetch_k, doc_id=doc_id)
        for q in all_queries:
            for doc in retriever.invoke(q):
                chunk_id = doc.metadata.get("chunk_id", id(doc))
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    candidates.append(doc)

        log.info("Candidates after dedup: %d", len(candidates))

        # Rerank then keep top k
        if self.reranker is not None:
            docs = self.reranker.rerank(query, candidates, top_k=k)
        else:
            docs = candidates[:k]

        # Build context blocks with source tags so the LLM can cite them.
        # Tag format depends on document origin:
        #   notion  → [<title> / <section>]
        #   pdf     → [page <n>, <filename>]
        #   others  → [<title>] or [<filename>]
        context_blocks = []
        for doc in docs:
            meta = doc.metadata
            dtype = meta.get("document_type", "")
            if dtype == "notion":
                title   = meta.get("title", "Notion")
                section = meta.get("section", "")
                tag = f"[{title} / {section}]" if section else f"[{title}]"
            else:
                pg   = meta.get("page")
                name = meta.get("filename") or meta.get("title", "unknown")
                tag  = f"[page {pg + 1}, {name}]" if pg is not None else f"[{name}]"
            context_blocks.append(f"{tag}\n{doc.page_content}")
        context = "\n\n".join(context_blocks)

        prompt = RAG_PROMPT.format(context=context, question=query)
        return self.llm.invoke(prompt)
