import os

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.client_components import build_client_components
from utils.config import AppConfig
from utils.file_processor import load_and_chunk_pdf, write_json
from utils.logger import AppLogger
from rag.knowledge.clusterer import TopicClusterer
from rag.knowledge.extractor import KnowledgeExtractor
from rag.knowledge.linker import CrossDocLinker
from rag.knowledge.manager import KnowledgeManager
from rag.knowledge.qa_generator import QAGenerator
from rag.retrieval.document_retriever import DocumentRetriever
from rag.retrieval.code_retrieval_service import CodeRetrievalService
from rag.retrieval.code_result_filter import CodeResultFilter
from rag.retrieval.filters import SearchFilter
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.pipeline import PipelineBuilder
from rag.retrieval.searcher import Searcher

log = AppLogger.get(__name__)

_DEFAULT_CODE_RESULT_FILTER = CodeResultFilter()


class LocalLlamaClient:
    """Wires a local embedding server, Chroma vector store, and local LLM.

    Retrieval is unified under the BaseRetriever Protocol:
      self.doc_retriever    — DocumentRetriever wrapping self.searcher
      self.code_retriever   — CodeRetriever (set externally when CodeIndexer
                              is available; None until then)
      self.unified_retriever — HybridRetriever([doc, code]) when both backends
                              are available; falls back to doc_retriever only.

    All knowledge/QA operations delegate to ``self.knowledge``.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        code_result_filter: CodeResultFilter | None = None,
    ) -> None:
        self.config = config
        log.info("Initialising LocalLlamaClient")
        log.debug("  embed_base=%s  embed_model=%s", config.embed_base, config.embed_model)
        log.debug("  llm_base=%s    llm_model=%s",   config.llm_base,   config.llm_model)
        log.debug("  persist_directory=%s",           config.persist_directory)
        log.debug("  db_url=%s",                      config.db_url)
        log.debug("  reranker_type=%s",               config.reranker_type)

        log.info("Building client subsystems")
        components = build_client_components(config)

        # Use one global filter instance unless explicitly overridden.
        self._code_result_filter = code_result_filter or components.code_result_filter

        self.embed = components.embed
        self.persist_directory = config.persist_directory
        self.db = components.db
        self.llm = components.llm
        self.indexer = components.indexer
        self.ingester = components.ingester
        self.reranker = components.reranker
        self.searcher = components.searcher
        log.info("Reranker: %s", type(self.reranker).__name__ if self.reranker else "disabled")

        # ── Unified Retrieval Layer ────────────────────────────────────────
        # doc_retriever is always available after __init__.
        # code_retriever starts as None; call attach_code_retriever() after
        # a CodeIndexer is ready to enable cross-domain unified search.
        self.doc_retriever: DocumentRetriever = components.doc_retriever
        self.code_retriever = None   # set via attach_code_retriever()

        # Build canonical pipelines via PipelineBuilder.
        # doc_pipeline is always available; unified_pipeline is rebuilt when
        # a CodeRetriever is attached.
        self.doc_pipeline = components.doc_pipeline
        self.unified_pipeline = self.doc_pipeline   # updated by _rebuild_unified()

        # Convenience aliases — keep unified_retriever/doc_retriever for callers
        # that accessed them directly in Step 1.3 tests.
        self.unified_retriever = self.doc_retriever  # updated by _rebuild_unified()

        self.engine = components.engine
        self.knowledge = components.knowledge
        self.memory_manager = components.memory_manager
        self.user_memory = components.user_memory
        self.timeline = components.timeline
        self.research = components.research
        self.memory = components.memory

        self._code_retrieval = CodeRetrievalService(
            config=self.config,
            embed=self.embed,
            reranker=self.reranker,
            persist_directory=self.persist_directory,
            code_result_filter=self._code_result_filter,
        )

        log.info("LocalLlamaClient ready")

    # ------------------------------------------------------------------
    # Unified Retrieval Layer helpers
    # ------------------------------------------------------------------

    def attach_code_retriever(self, code_indexer, *, level: str = "symbol") -> None:
        """Attach a CodeRetriever backed by *code_indexer* and rebuild pipelines.

        Call this after a CodeIndexer has been initialised to enable cross-domain
        hybrid search via answer_unified().

        Args:
            code_indexer : An initialised ``rag.code.indexer.CodeIndexer``.
            level        : Chroma collection level: repo / file / symbol / block.
        """
        from rag.retrieval.code_retriever import CodeRetriever
        self.code_retriever = CodeRetriever(
            code_indexer,
            level=level,
            reranker=self.reranker,
        )
        self._rebuild_unified()
        log.info("CodeRetriever attached (level=%s); unified pipeline updated", level)

    def _rebuild_unified(self) -> None:
        """Rebuild unified_pipeline (and alias unified_retriever) after code_retriever changes."""
        if self.code_retriever is not None:
            self.unified_retriever = HybridRetriever(
                [self.doc_retriever, self.code_retriever],
                reranker=self.reranker,
            )
            self.unified_pipeline = PipelineBuilder.unified_pipeline(
                [self.doc_retriever, self.code_retriever],
                reranker=self.reranker,
            )
        else:
            self.unified_retriever = self.doc_retriever
            self.unified_pipeline  = self.doc_pipeline

    # ------------------------------------------------------------------
    # Compatibility shims — expose subsystem internals callers rely on
    # ------------------------------------------------------------------

    @property
    def bm25_index(self):
        return self.searcher.bm25_index

    @property
    def extractor(self) -> KnowledgeExtractor:
        return self.knowledge.extractor

    @property
    def qa_generator(self) -> QAGenerator:
        return self.knowledge.qa_generator

    @property
    def clusterer(self) -> TopicClusterer:
        return self.knowledge.clusterer

    @property
    def linker(self) -> CrossDocLinker:
        return self.knowledge.linker

    # ------------------------------------------------------------------
    # Retrieval — delegate to Searcher
    # ------------------------------------------------------------------

    def get_retriever(self, k: int = 5, fetch_k: int = 20, doc_id: str | None = None):
        """Returns an MMR retriever scoped to *doc_id* when supplied."""
        search_kwargs: dict = {"k": k, "fetch_k": fetch_k}
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
            query, k=k, doc_id=doc_id, search_filter=search_filter
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

    def browse_code_blocks(
        self,
        *,
        repo_id: str | None = None,
        file_path: str | None = None,
        limit: int = 500,
        exclude_tests: bool = True,
    ) -> list[dict]:
        """Return code chunks from the ``code_block`` collection.

        Each item has keys:
            - ``content``  : full code text
            - ``metadata`` : stored CodeChunk metadata
        """
        return self._code_retrieval.browse_code_blocks(
            repo_id=repo_id,
            file_path=file_path,
            limit=limit,
            exclude_tests=exclude_tests,
        )

    def list_code_repo_ids(self, *, limit: int = 5000) -> list[str]:
        """Return distinct repo_id values found in ``code_block`` metadata."""
        return self._code_retrieval.list_code_repo_ids(limit=limit)

    def search_code_blocks(
        self,
        query: str,
        *,
        k: int = 5,
        fetch_k: int = 20,
        use_rerank: bool = False,
        include_relations: bool = False,
    ) -> dict:
        """Search directly in the ``code_block`` collection.

        Returns a dashboard-compatible payload:
            {"vector", "bm25", "hybrid", "reranked", "trace"}
        """
        return self._code_retrieval.search_code_blocks(
            query,
            k=k,
            fetch_k=fetch_k,
            use_rerank=use_rerank,
            include_relations=include_relations,
        )

    @staticmethod
    def _is_test_code_metadata(meta: dict) -> bool:
        """Backward-compatible shim; prefer CodeResultFilter in new code."""
        return _DEFAULT_CODE_RESULT_FILTER.is_test_metadata(meta)

    def _code_block_persist_dirs(self) -> list[str]:
        return self._code_retrieval.code_block_persist_dirs()

    def _enrich_code_results_with_relations(
        self,
        *,
        query: str,
        rows: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        return self._code_retrieval.enrich_code_results_with_relations(query=query, rows=rows)

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
            query, k=k, fetch_k=fetch_k, search_filter=search_filter
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
            query, k=k, fetch_k=fetch_k, doc_id=doc_id,
            use_rerank=use_rerank, use_hybrid=use_hybrid,
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
            query, k=k, fetch_k=fetch_k, doc_id=doc_id,
            use_rerank=use_rerank, use_hybrid=use_hybrid,
            search_filter=search_filter,
        )

    # ------------------------------------------------------------------
    # Knowledge — delegate to KnowledgeManager
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def answer_query(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        expand_query: bool | None = None,
    ):
        """Answer using the document retrieval pipeline (default)."""
        return self.engine.answer(
            query, k=k, fetch_k=fetch_k, doc_id=doc_id, expand_query=expand_query
        )

    def answer_unified(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        expand_query: bool | None = None,
        filters: dict | None = None,
    ):
        """Answer using the unified pipeline (document + code when available).

        Unlike answer_query, this method does NOT accept doc_id because the
        unified pipeline spans multiple backends.  Use the *filters* parameter
        to pass a raw Chroma where-dict for document-side filtering.
        """
        prev = self.engine.retriever
        self.engine.retriever = self.unified_pipeline
        self.engine._pipeline = None   # clear auto-built pipeline cache
        try:
            return self.engine.answer(
                query, k=k, fetch_k=fetch_k, expand_query=expand_query,
                filters=filters,
            )
        finally:
            self.engine.retriever = prev
            self.engine._pipeline = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_texts(self, texts, metadatas=None, ids=None):
        if hasattr(self.db, "add_texts"):
            return self.db.add_texts(texts, metadatas=metadatas, ids=ids)
        self.db = Chroma.from_texts(
            texts, embedding=self.embed, persist_directory=self.persist_directory
        )

    def add_document(
        self,
        path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        doc_id: str | None = None,
        extract_knowledge: bool = False,
    ) -> dict:
        """Ingest a document file, optionally running knowledge extraction before indexing."""
        try:
            chunks = self.ingester.ingest(
                path, doc_id=doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            log.info("add_document: %s  %d chunks  doc_id=%r  extract_knowledge=%s",
                     path, len(chunks), doc_id, extract_knowledge)
            if extract_knowledge:
                chunks = self.knowledge.extractor.enrich(chunks)
            return self.indexer.run(chunks)
        except Exception as e:
            log.error("add_document failed: %s", e, exc_info=True)
            raise

    def add_pdf(
        self,
        pdf_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        doc_id: str | None = None,
    ):
        """Backward-compatible wrapper — delegates to add_document."""
        return self.add_document(
            pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, doc_id=doc_id
        )

    def persist(self):
        pass  # langchain_chroma >= 0.1 auto-persists when persist_directory is set
