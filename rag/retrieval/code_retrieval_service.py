"""Code retrieval service used by LocalLlamaClient as a composed dependency."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.code.graph_store import GraphStore
from rag.code.indexer import CodeIndexer
from rag.retrieval.base import RetrievalResult
from rag.retrieval.code_query_scope import parse_code_query_scope, rerank_code_rows_by_scope
from rag.retrieval.code_result_filter import CodeResultFilter
from rag.retrieval.related_code_retriever import RelatedCodeRetriever


@dataclass
class CodeRetrievalService:
    """Encapsulates code retrieval flows: browse, search, and relation enrichment."""

    config: object
    embed: object
    reranker: object
    persist_directory: str
    code_result_filter: CodeResultFilter

    def code_block_persist_dirs(self) -> list[str]:
        dirs = [
            str(self.config.code_rag_root),
            str(self.persist_directory),
        ]
        out: list[str] = []
        seen: set[str] = set()
        for d in dirs:
            k = str(d).strip()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(k)
        return out

    def browse_code_blocks(
        self,
        *,
        repo_id: str | None = None,
        file_path: str | None = None,
        limit: int = 500,
        exclude_tests: bool = True,
    ) -> list[dict]:
        conditions: list[dict] = []
        if repo_id:
            conditions.append({"repo_id": {"$eq": repo_id}})
        if file_path:
            conditions.append({"file_path": {"$eq": file_path}})

        kwargs: dict = {"include": ["documents", "metadatas"], "limit": limit}
        if len(conditions) == 1:
            kwargs["where"] = conditions[0]
        elif len(conditions) > 1:
            kwargs["where"] = {"$and": conditions}

        for persist_dir in self.code_block_persist_dirs():
            try:
                block_db = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embed,
                    collection_name="code_block",
                )
            except Exception:
                continue

            try:
                result = block_db.get(**kwargs)
            except Exception:
                continue

            docs = result.get("documents") or []
            metas = result.get("metadatas") or []

            rows: list[dict] = []
            for text, meta in zip(docs, metas):
                if not text:
                    continue
                rows.append({"content": text, "metadata": dict(meta or {})})

            rows = self.code_result_filter.filter_content_rows(rows, exclude_tests=exclude_tests)
            if rows:
                return rows

        return []

    def list_code_repo_ids(self, *, limit: int = 5000) -> list[str]:
        out: set[str] = set()
        for persist_dir in self.code_block_persist_dirs():
            try:
                block_db = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embed,
                    collection_name="code_block",
                )
            except Exception:
                continue

            try:
                raw = block_db.get(include=["metadatas"], limit=max(1, int(limit)))
            except Exception:
                continue

            for meta in (raw.get("metadatas") or []):
                repo_id = str((meta or {}).get("repo_id", "")).strip()
                if repo_id:
                    out.add(repo_id)

        return sorted(out)

    def search_code_blocks(
        self,
        query: str,
        *,
        k: int = 5,
        fetch_k: int = 20,
        use_rerank: bool = False,
        include_relations: bool = False,
    ) -> dict:
        query_scope = parse_code_query_scope(query)
        semantic_query = query_scope.semantic_query or query
        raw: list[tuple[Document, float]] = []
        for persist_dir in self.code_block_persist_dirs():
            try:
                block_db = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embed,
                    collection_name="code_block",
                )
            except Exception:
                continue

            try:
                raw = block_db.similarity_search_with_score(semantic_query, k=fetch_k)
            except Exception:
                continue

            if raw:
                break

        if not raw:
            return {"vector": [], "bm25": None, "hybrid": None, "reranked": None, "trace": []}

        raw = self.code_result_filter.filter_scored_documents(raw, exclude_tests=True)
        if not raw:
            return {"vector": [], "bm25": None, "hybrid": None, "reranked": None, "trace": []}

        vector = [(doc, round(1 / (1 + dist), 4)) for doc, dist in raw]
        vector = rerank_code_rows_by_scope(vector, query_scope)

        if include_relations:
            vector = self.enrich_code_results_with_relations(query=semantic_query, rows=vector)

        reranked = None
        if use_rerank and self.reranker is not None:
            reranked = self.reranker.rerank_with_scores(
                semantic_query,
                [doc for doc, _ in vector],
                top_k=k,
            )

        return {
            "vector": vector,
            "bm25": None,
            "hybrid": None,
            "reranked": reranked,
            "trace": [],
        }

    def enrich_code_results_with_relations(
        self,
        *,
        query: str,
        rows: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        if not rows:
            return rows

        try:
            graph = GraphStore(self.config.graph_db_path)
        except Exception:
            return rows

        candidates = self.code_block_persist_dirs()
        block_db = None
        block_persist_dir = ""
        for persist_dir in candidates:
            try:
                block_db = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embed,
                    collection_name="code_block",
                )
                block_persist_dir = str(persist_dir)
                break
            except Exception:
                continue
        if block_db is None:
            return rows

        try:
            block_indexer = CodeIndexer(block_persist_dir, self.embed)
        except Exception:
            block_indexer = None

        class _StaticRetriever:
            def __init__(
                self,
                base_rows: list[tuple[Document, float]],
                *,
                indexer: CodeIndexer | None = None,
            ) -> None:
                self._base_rows = [
                    RetrievalResult(
                        content=doc.page_content,
                        score=float(score),
                        source="code",
                        metadata=dict(doc.metadata or {}),
                    )
                    for doc, score in base_rows
                ]
                self._indexer = indexer

            def search(self, _query: str, top_k: int = 5, filters=None, repo_ids=None):
                return self._base_rows[: max(0, int(top_k))]

        def _fetch_block(repo_id: str, target_id: str) -> dict | None:
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

        def _fetch_file_blocks(repo_id: str, file_path: str) -> list[dict]:
            where = {
                "$and": [
                    {"repo_id": {"$eq": repo_id}},
                    {"file_path": {"$eq": file_path}},
                ]
            }
            raw = block_db.get(where=where, include=["metadatas"])
            metas = raw.get("metadatas") or []
            return [dict(m or {}) for m in metas]

        retriever = RelatedCodeRetriever(
            _StaticRetriever(rows, indexer=block_indexer),
            graph,
            block_fetcher=_fetch_block,
            file_blocks_fetcher=_fetch_file_blocks,
        )

        try:
            enriched = retriever.search(query, top_k=len(rows), filters=None)
        except Exception:
            return rows

        out: list[tuple[Document, float]] = []
        for r in enriched:
            out.append((Document(page_content=r.content, metadata=dict(r.metadata or {})), float(r.score)))
        return out
