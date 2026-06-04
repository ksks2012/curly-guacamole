"""UI-facing client protocols used by controllers.

Controllers depend on these narrow interfaces instead of a single concrete
LocalLlamaClient type.
"""

from __future__ import annotations

from typing import Protocol

from rag.retrieval.filters import SearchFilter


class SearchClientProtocol(Protocol):
    """Read/query operations needed by SearchController."""

    def search_code_blocks(
        self,
        query: str,
        *,
        k: int = 5,
        fetch_k: int = 20,
        use_rerank: bool = False,
        include_relations: bool = False,
    ) -> dict:
        ...

    def search_for_trace(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        use_rerank: bool = False,
        use_hybrid: bool = False,
        search_filter: SearchFilter | None = None,
    ) -> dict:
        ...

    def list_doc_ids(self) -> list[str]:
        ...

    def list_doc_title_map(self) -> dict[str, str]:
        ...

    def list_workspaces(self) -> list[str]:
        ...

    def list_document_types(self) -> list[str]:
        ...

    def list_tags(self) -> list[str]:
        ...

    def browse_code_blocks(
        self,
        *,
        repo_id: str | None = None,
        file_path: str | None = None,
        limit: int = 500,
        exclude_tests: bool = True,
    ) -> list[dict]:
        ...

    def list_code_repo_ids(self, *, limit: int = 5000) -> list[str]:
        ...


class _IngestPipelineProtocol(Protocol):
    def ingest(
        self,
        path: str,
        doc_id: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        title: str = "",
        tags: list[str] | None = None,
        workspace: str = "",
        importance: float = 0.0,
        strategy: str | None = None,
    ):
        ...


class _IndexerRunnerProtocol(Protocol):
    def run(self, chunks) -> dict:
        ...


class _VectorDbProtocol(Protocol):
    def get(self, **kwargs) -> dict:
        ...


class _UploadConfigProtocol(Protocol):
    upload_dir: str


class IndexClientProtocol(Protocol):
    """Index and metadata operations needed by IndexController."""

    config: _UploadConfigProtocol
    ingester: _IngestPipelineProtocol
    indexer: _IndexerRunnerProtocol
    db: _VectorDbProtocol

    def invalidate_bm25(self) -> None:
        ...

    def list_doc_ids(self) -> list[str]:
        ...

    def list_doc_title_map(self) -> dict[str, str]:
        ...

    def enrich_doc(self, doc_id: str, overwrite: bool = False) -> dict:
        ...


class KnowledgeClientProtocol(Protocol):
    """Knowledge-card operations needed by KnowledgeController."""

    def browse_chunks(
        self,
        doc_id: str | None = None,
        tag: str | None = None,
        topic: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        ...

    def list_doc_title_map(self) -> dict[str, str]:
        ...

    def list_field_values(self, field: str) -> list[str]:
        ...

    def cluster_topics(self, n_clusters: int = 8, doc_id: str | None = None):
        ...

    def list_doc_ids(self) -> list[str]:
        ...

    def enrich_doc(self, doc_id: str, overwrite: bool = False) -> dict:
        ...
