"""Composable capability facades used by LocalLlamaClient.

These classes split client responsibilities by domain (retrieval, knowledge,
generation, indexing) so callers can depend on a smaller interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document

if TYPE_CHECKING:
    from rag.engine import RAGEngine
    from rag.indexer import Indexer
    from rag.ingest.document_ingester import DocumentIngester
    from rag.knowledge.manager import KnowledgeManager
    from rag.retrieval.code_retrieval_service import CodeRetrievalService
    from rag.retrieval.filters import SearchFilter
    from rag.retrieval.searcher import Searcher


@dataclass
class RetrievalCapability:
    """Retrieval-only interface over search and code retrieval services."""

    db: Chroma
    searcher: "Searcher"
    code_retrieval: "CodeRetrievalService"

    def get_retriever(self, k: int = 5, fetch_k: int = 20, doc_id: str | None = None):
        search_kwargs: dict[str, Any] = {"k": k, "fetch_k": fetch_k}
        if doc_id is not None:
            search_kwargs["filter"] = {"doc_id": doc_id}
        return self.db.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    def similarity_search(self, query: str, k: int = 4, doc_id: str | None = None):
        return self.searcher.similarity_search(query, k=k, doc_id=doc_id)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        doc_id: str | None = None,
        search_filter: "SearchFilter | None" = None,
    ) -> list[tuple[Document, float]]:
        return self.searcher.similarity_search_with_scores(
            query,
            k=k,
            doc_id=doc_id,
            search_filter=search_filter,
        )

    def list_doc_ids(self) -> list[str]:
        return self.searcher.list_doc_ids()

    def list_doc_title_map(self) -> dict[str, str]:
        return self.searcher.list_doc_title_map()

    def list_field_values(self, field: str) -> list[str]:
        return self.searcher.list_field_values(field)

    def list_workspaces(self) -> list[str]:
        return self.searcher.list_workspaces()

    def list_document_types(self) -> list[str]:
        return self.searcher.list_document_types()

    def list_tags(self) -> list[str]:
        return self.searcher.list_tags()

    def browse_chunks(
        self,
        doc_id: str | None = None,
        tag: str | None = None,
        topic: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        return self.searcher.browse_chunks(doc_id=doc_id, tag=tag, topic=topic, limit=limit)

    def rebuild_bm25(self) -> None:
        self.searcher.rebuild_bm25()

    def invalidate_bm25(self) -> None:
        self.searcher.invalidate_bm25()

    def hybrid_search_with_scores(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        search_filter: "SearchFilter | None" = None,
    ) -> tuple:
        return self.searcher.hybrid_search_with_scores(
            query,
            k=k,
            fetch_k=fetch_k,
            search_filter=search_filter,
        )

    def search_for_debug(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        use_rerank: bool = False,
        use_hybrid: bool = False,
        search_filter: "SearchFilter | None" = None,
    ) -> dict:
        return self.searcher.search_for_debug(
            query,
            k=k,
            fetch_k=fetch_k,
            doc_id=doc_id,
            use_rerank=use_rerank,
            use_hybrid=use_hybrid,
            search_filter=search_filter,
        )

    def search_for_trace(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        use_rerank: bool = False,
        use_hybrid: bool = False,
        search_filter: "SearchFilter | None" = None,
    ) -> dict:
        return self.searcher.search_for_trace(
            query,
            k=k,
            fetch_k=fetch_k,
            doc_id=doc_id,
            use_rerank=use_rerank,
            use_hybrid=use_hybrid,
            search_filter=search_filter,
        )

    def browse_code_blocks(
        self,
        *,
        repo_id: str | None = None,
        file_path: str | None = None,
        limit: int = 500,
        exclude_tests: bool = True,
    ) -> list[dict]:
        return self.code_retrieval.browse_code_blocks(
            repo_id=repo_id,
            file_path=file_path,
            limit=limit,
            exclude_tests=exclude_tests,
        )

    def list_code_repo_ids(self, *, limit: int = 5000) -> list[str]:
        return self.code_retrieval.list_code_repo_ids(limit=limit)

    def search_code_blocks(
        self,
        query: str,
        *,
        k: int = 5,
        fetch_k: int = 20,
        use_rerank: bool = False,
        include_relations: bool = False,
    ) -> dict:
        return self.code_retrieval.search_code_blocks(
            query,
            k=k,
            fetch_k=fetch_k,
            use_rerank=use_rerank,
            include_relations=include_relations,
        )

    def code_block_persist_dirs(self) -> list[str]:
        return self.code_retrieval.code_block_persist_dirs()

    def enrich_code_results_with_relations(
        self,
        *,
        query: str,
        rows: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        return self.code_retrieval.enrich_code_results_with_relations(query=query, rows=rows)


@dataclass
class KnowledgeCapability:
    """Knowledge-only interface over KnowledgeManager."""

    knowledge: "KnowledgeManager"

    def enrich_doc(self, doc_id: str, overwrite: bool = False) -> dict:
        return self.knowledge.enrich_doc(doc_id, overwrite=overwrite)

    def generate_qa(self, doc_id: str, overwrite: bool = False) -> dict:
        return self.knowledge.generate_qa(doc_id, overwrite=overwrite)

    def qa_search(self, query: str, k: int = 5) -> list[dict]:
        return self.knowledge.qa_search(query, k=k)

    def cluster_topics(self, n_clusters: int = 8, doc_id: str | None = None):
        return self.knowledge.cluster_topics(n_clusters=n_clusters, doc_id=doc_id)

    def link_chunks(
        self,
        top_k: int = 5,
        threshold: float = 0.75,
        doc_id: str | None = None,
    ):
        return self.knowledge.link_chunks(top_k=top_k, threshold=threshold, doc_id=doc_id)

    def link_pages(self, top_k: int = 5, threshold: float = 0.70):
        return self.knowledge.link_pages(top_k=top_k, threshold=threshold)

    def get_related_chunks(self, chunk_id: str) -> list[dict]:
        return self.knowledge.get_related_chunks(chunk_id)

    def get_related_pages(self, doc_id: str) -> list[dict]:
        return self.knowledge.get_related_pages(doc_id)


@dataclass
class GenerationCapability:
    """Generation-only interface over RAGEngine."""

    engine: "RAGEngine"
    unified_pipeline_provider: Callable[[], Any]

    def answer_query(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        expand_query: bool | None = None,
    ):
        return self.engine.answer(
            query,
            k=k,
            fetch_k=fetch_k,
            doc_id=doc_id,
            expand_query=expand_query,
        )

    def answer_unified(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        expand_query: bool | None = None,
        filters: dict | None = None,
    ):
        prev = self.engine.retriever
        self.engine.retriever = self.unified_pipeline_provider()
        self.engine._pipeline = None
        try:
            return self.engine.answer(
                query,
                k=k,
                fetch_k=fetch_k,
                expand_query=expand_query,
                filters=filters,
            )
        finally:
            self.engine.retriever = prev
            self.engine._pipeline = None


@dataclass
class IndexingCapability:
    """Indexing-only interface over ingester and indexer."""

    db: Chroma
    embed: object
    persist_directory: str
    ingester: "DocumentIngester"
    indexer: "Indexer"
    knowledge: "KnowledgeManager"

    def add_texts(self, texts, metadatas=None, ids=None):
        if hasattr(self.db, "add_texts"):
            return self.db.add_texts(texts, metadatas=metadatas, ids=ids)
        self.db = Chroma.from_texts(
            texts,
            embedding=self.embed,
            persist_directory=self.persist_directory,
        )
        return None

    def add_document(
        self,
        path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        doc_id: str | None = None,
        extract_knowledge: bool = False,
    ) -> dict:
        chunks = self.ingester.ingest(
            path,
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if extract_knowledge:
            chunks = self.knowledge.extractor.enrich(chunks)
        return self.indexer.run(chunks)
