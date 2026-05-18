import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config import AppConfig
from utils.file_processor import load_and_chunk_pdf, write_json
from utils.logger import AppLogger
from rag.engine import RAGEngine
from rag.indexer import Indexer
from rag.ingest.document_ingester import DocumentIngester
from rag.knowledge.clusterer import TopicClusterer
from rag.knowledge.extractor import KnowledgeExtractor
from rag.knowledge.linker import CrossDocLinker
from rag.knowledge.manager import KnowledgeManager
from rag.knowledge.qa_generator import QAGenerator
from rag.memory.manager          import ConversationMemory
from rag.memory.store            import MemoryStore
from rag.memory.user_memory      import UserMemoryManager
from rag.memory.timeline         import KnowledgeTimeline
from rag.memory.research_session import ResearchSessionManager
from rag.reranker import RerankerFactory
from rag.retrieval.filters import SearchFilter
from rag.retrieval.searcher import Searcher

log = AppLogger.get(__name__)


class LocalLlamaClient:
    """Wires a local embedding server, Chroma vector store, and local LLM.

    All retrieval operations delegate to ``self.searcher``.
    All knowledge/QA operations delegate to ``self.knowledge``.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        log.info("Initialising LocalLlamaClient")
        log.debug("  embed_base=%s  embed_model=%s", config.embed_base, config.embed_model)
        log.debug("  llm_base=%s    llm_model=%s",   config.llm_base,   config.llm_model)
        log.debug("  persist_directory=%s",           config.persist_directory)
        log.debug("  db_url=%s",                      config.db_url)
        log.debug("  reranker_type=%s",               config.reranker_type)

        log.info("Building embeddings client → %s", config.embed_base)
        self.embed = OpenAIEmbeddings(
            openai_api_key=config.api_key,
            openai_api_base=config.embed_base,
            model=config.embed_model,
        )

        log.info("Opening Chroma store → %s  collection=%s",
                 config.persist_directory, config.setup_rag_collection)
        self.persist_directory = config.persist_directory
        self.db = Chroma(
            persist_directory=config.persist_directory,
            embedding_function=self.embed,
            collection_name=config.setup_rag_collection or "rag_collection",
        )
        log.info("Chroma store ready")

        log.info("Building LLM client → %s", config.llm_base)
        self.llm = ChatOpenAI(
            base_url=config.llm_base,
            api_key=config.api_key,
            model=config.llm_model,
            **config.llm_kwargs,
        )

        self.indexer  = Indexer(
            db=self.db,
            namespace=config.setup_rag_collection,
            db_url=config.db_url,
            batch_limit=config.batch_limit,
        )
        self.ingester = DocumentIngester(embeddings=self.embed)
        self.reranker = RerankerFactory.build(config, llm=self.llm)
        log.info("Reranker: %s", type(self.reranker).__name__ if self.reranker else "disabled")

        self.engine = RAGEngine(
            llm=self.llm,
            get_retriever=self.get_retriever,
            reranker=self.reranker,
            config=config,
        )

        self.searcher = Searcher(db=self.db, reranker=self.reranker)

        _qa_collection = (config.setup_rag_collection or "rag_collection") + "_qa"
        _qa_db = Chroma(
            persist_directory=config.persist_directory,
            embedding_function=self.embed,
            collection_name=_qa_collection,
        )
        self.knowledge = KnowledgeManager(
            db=self.db,
            qa_db=_qa_db,
            qa_indexer=Indexer(
                db=_qa_db,
                namespace=_qa_collection,
                db_url=config.db_url,
                batch_limit=config.batch_limit,
            ),
            extractor=KnowledgeExtractor(self.llm),
            qa_generator=QAGenerator(self.llm),
            clusterer=TopicClusterer(llm=self.llm, db=self.db),
            linker=CrossDocLinker(db=self.db),
        )

        # Stage C — Memory subsystems (C.1 + C.2 + C.3 + C.4)
        _mem_store        = MemoryStore(db_path=config.memory_db_path)
        self.user_memory  = UserMemoryManager(store=_mem_store)
        self.timeline     = KnowledgeTimeline(store=_mem_store)
        self.research     = ResearchSessionManager(store=_mem_store)
        self.memory       = ConversationMemory(
            store=_mem_store,
            llm=self.llm,
            session_id=config.memory_default_session,
            max_recent=config.memory_max_recent,
            max_topics=config.memory_max_topics,
            extract_topics=config.memory_extract_topics,
            auto_infer_project=config.memory_auto_infer_project,
            user_memory=self.user_memory,
            timeline=self.timeline,
            research=self.research,
        )
        self.memory.ensure_session()
        # Wire memory into the RAG engine so every answer_query() auto-updates it
        self.engine.memory = self.memory

        log.info("LocalLlamaClient ready")

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
    # Memory shims — delegate to ConversationMemory
    # ------------------------------------------------------------------

    def set_active_project(self, project: str) -> None:
        self.memory.set_active_project(project)

    def infer_project(self) -> str:
        return self.memory.infer_project()

    def get_memory_state(self):
        return self.memory.get_state()

    def list_sessions(self) -> list[dict]:
        return self.memory.list_sessions()

    def clear_memory_session(self) -> None:
        self.memory.clear_session()

    def switch_memory_session(self, session_id: str) -> None:
        self.memory.ensure_session(session_id)

    # ------------------------------------------------------------------
    # User Memory shims (C.2)
    # ------------------------------------------------------------------

    def get_user_interests(self, n: int = 10) -> list[dict]:
        """Return top *n* user research interests by recency-weighted score."""
        return self.user_memory.get_top_interests(n)

    def get_user_profile(self):
        """Return a UserProfile snapshot."""
        return self.user_memory.get_profile()

    # ------------------------------------------------------------------
    # Timeline shims (C.3)
    # ------------------------------------------------------------------

    def get_timeline_recent(self, days: int = 30) -> list[dict]:
        """Return daily timeline entries for the last *days* days."""
        return self.timeline.get_recent(days)

    def get_timeline_period(self, year: int, month: int | None = None) -> list[dict]:
        """Return timeline entries for *year* (or *year*+*month*)."""
        return self.timeline.get_period(year, month)

    def get_yearly_summary(self, year: int) -> dict[str, int]:
        """Return ``{topic: appearance_count}`` for *year*, descending."""
        return self.timeline.get_yearly_summary(year)

    # ------------------------------------------------------------------
    # Research Session shims (C.4)
    # ------------------------------------------------------------------

    def start_research_session(self, name: str, tags: list[str] = [], set_active: bool = True):
        """Create a new research session and optionally set it as active."""
        from rag.memory.research_session import ResearchSession
        session: ResearchSession = self.research.create(name, tags=tags)
        if set_active:
            self.research.set_active(session.session_id)
        return session

    def get_research_session(self, session_id: str | None = None):
        """Return the active session (or by *session_id*)."""
        if session_id:
            return self.research.get(session_id)
        return self.research.get_active_session()

    def list_research_sessions(self, archived: bool = False) -> list:
        """List active (or archived) research sessions."""
        return self.research.list_archived() if archived else self.research.list_active()

    def archive_research_session(self, session_id: str | None = None) -> None:
        """Archive a session (defaults to the active one)."""
        sid = session_id or self.research.active_session_id
        if sid:
            self.research.archive(sid)

    def add_research_note(self, content: str, source_doc_ids: list[str] = [], session_id: str | None = None):
        """Add a note to the active research session (or *session_id*)."""
        return self.research.add_note(content, session_id=session_id, source_doc_ids=source_doc_ids)

    def get_research_notes(self, session_id: str | None = None) -> list:
        """Return all notes for the active session (or *session_id*)."""
        return self.research.get_notes(session_id)

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
        return self.engine.answer(
            query, k=k, fetch_k=fetch_k, doc_id=doc_id, expand_query=expand_query
        )

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
