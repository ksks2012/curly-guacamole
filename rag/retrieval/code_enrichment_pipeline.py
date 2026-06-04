"""Composable pipeline for enriching code retrieval rows with graph relations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.code.graph_store import GraphStore
from rag.code.indexer import CodeIndexer
from rag.retrieval.code_block_store import CodeBlockStore
from rag.retrieval.related_code_retriever import RelatedCodeRetriever
from rag.retrieval.static_document_list import StaticDocumentList


@dataclass
class CodeEnrichmentPipeline:
    """Graph relation enrichment for code rows with explicit storage dependency."""

    config: object
    embed: object
    block_store: CodeBlockStore

    def enrich(
        self,
        *,
        query: str,
        rows: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        if not rows:
            return rows

        graph = self._build_graph_store()
        if graph is None:
            return rows

        block_persist_dir, block_db = self.block_store.first_database()
        if block_db is None or block_persist_dir is None:
            return rows

        block_indexer = self._build_block_indexer(block_persist_dir)
        retriever = RelatedCodeRetriever(
            StaticDocumentList(rows, indexer=block_indexer),
            graph,
            block_fetcher=partial(self.fetch_block_metadata, block_db),
            file_blocks_fetcher=partial(self.fetch_file_block_metadatas, block_db),
        )

        try:
            enriched = retriever.search(query, top_k=len(rows), filters=None)
        except Exception:
            return rows
        return self._to_document_rows(enriched)

    def _build_graph_store(self) -> GraphStore | None:
        try:
            return GraphStore(self.config.graph_db_path)
        except Exception:
            return None

    def _build_block_indexer(self, block_persist_dir: str) -> CodeIndexer | None:
        try:
            return CodeIndexer(block_persist_dir, self.embed)
        except Exception:
            return None

    @staticmethod
    def fetch_block_metadata(block_db: Chroma, repo_id: str, target_id: str) -> dict | None:
        where = {
            "$and": [
                {"repo_id": {"$eq": repo_id}},
                {"chunk_id": {"$eq": target_id}},
            ]
        }
        raw = block_db.get(where=where, include=["metadatas"])
        metas = raw.get("metadatas") or []
        if not metas:
            return None
        return dict(metas[0] or {})

    @staticmethod
    def fetch_file_block_metadatas(block_db: Chroma, repo_id: str, file_path: str) -> list[dict]:
        where = {
            "$and": [
                {"repo_id": {"$eq": repo_id}},
                {"file_path": {"$eq": file_path}},
            ]
        }
        raw = block_db.get(where=where, include=["metadatas"])
        metas = raw.get("metadatas") or []
        return [dict(m or {}) for m in metas]

    @staticmethod
    def _to_document_rows(enriched_rows) -> list[tuple[Document, float]]:
        out: list[tuple[Document, float]] = []
        for row in enriched_rows:
            out.append(
                (
                    Document(page_content=row.content, metadata=dict(row.metadata or {})),
                    float(row.score),
                )
            )
        return out
