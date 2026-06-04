"""Top-level static retriever used for relation enrichment composition."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from rag.retrieval.base import RetrievalResult


@dataclass
class StaticDocumentList:
    """Expose a retriever-like ``search`` API over an in-memory row list."""

    rows: list[tuple[Document, float]]
    indexer: object | None = None

    def __post_init__(self) -> None:
        self._base_rows = [
            RetrievalResult(
                content=doc.page_content,
                score=float(score),
                source="code",
                metadata=dict(doc.metadata or {}),
            )
            for doc, score in self.rows
        ]

    def search(self, _query: str, top_k: int = 5, filters=None, repo_ids=None):
        return self._base_rows[: max(0, int(top_k))]
