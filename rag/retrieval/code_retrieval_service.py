"""Code retrieval service used by LocalLlamaClient as a composed dependency."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from rag.retrieval.code_block_store import CodeBlockStore
from rag.retrieval.code_enrichment_pipeline import CodeEnrichmentPipeline
from rag.retrieval.code_query_scope import parse_code_query_scope, rerank_code_rows_by_scope
from rag.retrieval.code_result_filter import CodeResultFilter
from rag.retrieval.collections import CODE_BLOCK_COLLECTION


@dataclass
class CodeRetrievalService:
    """Encapsulates code retrieval flows: browse, search, and relation enrichment."""

    config: object
    embed: object
    reranker: object
    persist_directory: str
    code_result_filter: CodeResultFilter
    collection_name: str = CODE_BLOCK_COLLECTION
    _enrichment_pipeline: CodeEnrichmentPipeline | None = field(default=None, init=False, repr=False)

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

    def _code_block_store(self) -> CodeBlockStore:
        return CodeBlockStore(
            embed=self.embed,
            persist_dirs=self.code_block_persist_dirs(),
            collection_name=self.collection_name,
        )

    def _code_enrichment_pipeline(self) -> CodeEnrichmentPipeline:
        if self._enrichment_pipeline is None:
            self._enrichment_pipeline = CodeEnrichmentPipeline(
                config=self.config,
                embed=self.embed,
                block_store=self._code_block_store(),
            )
        return self._enrichment_pipeline

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

        for _, block_db in self._code_block_store().iter_databases():

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
        """List distinct repo IDs by scanning ``code_block`` collection metadatas."""
        out: set[str] = set()
        for _, block_db in self._code_block_store().iter_databases():

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
        for _, block_db in self._code_block_store().iter_databases():

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
        return self._code_enrichment_pipeline().enrich(query=query, rows=rows)
