"""
Reranker module.

Pipeline:
    query
     ↓
    vector search (top fetch_k candidates)
     ↓
    reranker  ← this module
     ↓
    top k docs  → LLM

Two concrete implementations are provided:
  - CrossEncoderReranker  : uses a sentence-transformers cross-encoder model (local, fast)
  - LLMReranker           : asks the LLM to score each candidate (no extra model needed)

Both inherit from BaseReranker so they can be swapped out transparently.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from utils.config import AppConfig

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


class BaseReranker(ABC):
    """Abstract reranker. Subclasses must implement `rerank` and `rerank_with_scores`."""

    @abstractmethod
    def rerank_with_scores(
        self, query: str, docs: list[Document], top_k: int
    ) -> list[tuple[Document, float]]:
        """Return (document, score) pairs for the top_k most relevant documents, best-first."""
        ...

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """Return the top_k most relevant documents, best-first."""
        return [doc for doc, _ in self.rerank_with_scores(query, docs, top_k)]


# ---------------------------------------------------------------------------
# Cross-encoder reranker (sentence-transformers)
# ---------------------------------------------------------------------------

class CrossEncoderReranker(BaseReranker):
    """
    Reranker backed by a sentence-transformers cross-encoder model.

    Requires:  pip install sentence-transformers
    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
      - ~66 MB, good balance of speed and quality
      - Swap for 'cross-encoder/ms-marco-electra-base' for higher quality
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        # Prefer local cache to avoid unnecessary network round-trips.
        # Fall back to downloading only when the model is not cached yet.
        try:
            self.model = CrossEncoder(model_name, local_files_only=True)
            log.debug("CrossEncoder loaded from local cache: %s", model_name)
        except Exception:
            log.info(
                "CrossEncoder cache miss for %s — downloading from hub (first run only)",
                model_name,
            )
            self.model = CrossEncoder(model_name)

    def rerank_with_scores(
        self, query: str, docs: list[Document], top_k: int
    ) -> list[tuple[Document, float]]:
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)  # returns numpy array of floats

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [(doc, float(score)) for score, doc in ranked[:top_k]]


# ---------------------------------------------------------------------------
# LLM reranker (no extra model — uses the existing ChatOpenAI instance)
# ---------------------------------------------------------------------------

_LLM_RERANK_PROMPT = """\
You are a relevance judge. Given a query and a list of document excerpts, \
score each excerpt on its relevance to the query on a scale of 0 to 10 \
(10 = perfectly relevant, 0 = completely irrelevant).

Query: {query}

Documents:
{docs_block}

Reply ONLY with a JSON array of integers in the same order as the documents, \
e.g. [8, 3, 9, 1].  Do not include any other text.
"""


class LLMReranker(BaseReranker):
    """
    Reranker that asks the LLM to score each candidate document.

    No additional dependencies required; reuses the ChatOpenAI instance
    that is already wired into LocalLlamaClient.

    Trade-off: slightly slower (one extra LLM call per query) but works with
    any model and can leverage semantic understanding beyond embedding space.
    """

    def __init__(self, llm: "ChatOpenAI"):
        self.llm = llm

    def rerank_with_scores(
        self, query: str, docs: list[Document], top_k: int
    ) -> list[tuple[Document, float]]:
        if not docs:
            return []

        docs_block = "\n\n".join(
            f"[{i}] {doc.page_content[:400]}" for i, doc in enumerate(docs)
        )
        prompt = _LLM_RERANK_PROMPT.format(query=query, docs_block=docs_block)

        try:
            import json
            response = self.llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            scores = json.loads(raw.strip())

            if not isinstance(scores, list) or len(scores) != len(docs):
                raise ValueError("Unexpected response shape from LLM reranker")

            ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            return [(doc, float(score)) for score, doc in ranked[:top_k]]

        except Exception as e:
            # Fallback: return first top_k docs unchanged if LLM response is unparseable
            print(f"[LLMReranker] fallback to original order due to: {e}")
            return [(doc, 0.0) for doc in docs[:top_k]]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class RerankerFactory:
    """Instantiates the configured reranker without exposing construction details."""

    @staticmethod
    def build(config: "AppConfig", llm=None) -> "BaseReranker | None":
        """Return the appropriate BaseReranker, or None when reranking is disabled.

        Args:
            config : AppConfig — reads reranker_type and reranker_model.
            llm    : ChatOpenAI instance, required only when reranker_type == 'llm'.
        """
        kind = config.reranker_type
        if kind == "cross_encoder":
            return CrossEncoderReranker(model_name=config.reranker_model)
        if kind == "llm":
            if llm is None:
                raise ValueError("LLMReranker requires an llm instance; pass llm= to build()")
            return LLMReranker(llm)
        # 'none' or unrecognised → reranking disabled
        return None
