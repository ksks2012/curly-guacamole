import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config import AppConfig
from utils.file_processor import load_and_chunk_pdf, write_json
from utils.logger import AppLogger
from rag.engine import RAGEngine
from rag.indexer import Indexer
from rag.reranker import RerankerFactory

log = AppLogger.get(__name__)

# ---------------------------------------------------------------------------
# All prompt templates live in rag/prompt.py.
# RAG_PROMPT and QUERY_EXPANSION_PROMPT are imported from there.
# ---------------------------------------------------------------------------


class LocalLlamaClient:
    """
    Wraps a local embedding server (OpenAI-compatible), a Chroma vector store,
    and a local LLM (OpenAI-compatible).
    """

    def __init__(self, config: AppConfig):
        # keep config for runtime settings
        self.config = config
        log.info("Initialising LocalLlamaClient")
        log.debug("  embed_base=%s  embed_model=%s", config.embed_base, config.embed_model)
        log.debug("  llm_base=%s    llm_model=%s", config.llm_base, config.llm_model)
        log.debug("  persist_directory=%s", config.persist_directory)
        log.debug("  db_url=%s", config.db_url)
        log.debug("  reranker_type=%s", config.reranker_type)

        # Embedding: points to your embedding server (llama.cpp server)
        log.info("Building embeddings client → %s", config.embed_base)
        self.embed = OpenAIEmbeddings(
            openai_api_key=config.api_key,
            openai_api_base=config.embed_base,
            model=config.embed_model,
        )

        # Vector store (Chroma)
        log.info("Opening Chroma store → %s", config.persist_directory)
        self.persist_directory = config.persist_directory
        self.db = Chroma(
            persist_directory=config.persist_directory,
            embedding_function=self.embed,
        )
        log.info("Chroma store ready")

        # LLM (chat) - points to your LLM server (also OpenAI-compatible)
        log.info("Building LLM client → %s", config.llm_base)
        self.llm = ChatOpenAI(
            base_url=config.llm_base,
            api_key=config.api_key,
            model=config.llm_model,
            **config.llm_kwargs,
        )

        self.indexer = Indexer(
            db=self.db,
            namespace=config.setup_rag_collection,
            db_url=config.db_url,
            batch_limit=config.batch_limit,
        )

        self.reranker = RerankerFactory.build(config, llm=self.llm)
        log.info("Reranker: %s", type(self.reranker).__name__ if self.reranker else "disabled")

        self.engine = RAGEngine(
            llm=self.llm,
            get_retriever=self.get_retriever,
            reranker=self.reranker,
            config=config,
        )
        log.info("LocalLlamaClient ready")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_retriever(self, k: int = 5, fetch_k: int = 20, doc_id: str | None = None):
        """
        Returns an MMR retriever.
          k       : number of documents returned to the LLM
          fetch_k : candidate pool size before MMR re-ranking (larger = more diverse / slower)
          doc_id  : when provided, restricts retrieval to chunks from that document
        """
        search_kwargs: dict = {"k": k, "fetch_k": fetch_k}
        if doc_id is not None:
            search_kwargs["filter"] = {"doc_id": doc_id}
        return self.db.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

    def similarity_search(self, query: str, k: int = 4, doc_id: str | None = None):
        """Returns a list of similar documents from Chroma (LangChain Document objects)."""
        kwargs: dict = {"k": k}
        if doc_id is not None:
            kwargs["filter"] = {"doc_id": doc_id}
        return self.db.similarity_search(query, **kwargs)

    def similarity_search_with_scores(
        self, query: str, k: int = 5, doc_id: str | None = None
    ) -> list[tuple[Document, float]]:
        """Returns (Document, score) pairs sorted best-first (lower L2 distance = better).

        Chroma returns L2 distance; we convert to a 0-1 relevance score so the
        dashboard can display a human-readable value: relevance = 1 / (1 + distance).
        """
        log.debug("similarity_search_with_scores: query=%r  k=%d  doc_id=%s", query, k, doc_id)
        kwargs: dict = {"k": k}
        if doc_id is not None:
            kwargs["filter"] = {"doc_id": doc_id}
        raw = self.db.similarity_search_with_score(query, **kwargs)
        log.debug("  raw results: %d  (L2 distances: %s)",
                  len(raw), [round(d, 4) for _, d in raw])
        # Convert L2 distance → relevance score in [0, 1]
        return [(doc, round(1 / (1 + dist), 4)) for doc, dist in raw]

    def list_doc_ids(self) -> list[str]:
        """Return all distinct doc_id values stored in the Chroma collection.

        Fetches only metadata (no vectors or documents) so it is lightweight.
        Used by the dashboard filter dropdown.
        """
        result = self.db.get(include=["metadatas"])
        ids = {
            m.get("doc_id")
            for m in (result.get("metadatas") or [])
            if m and m.get("doc_id")
        }
        return sorted(ids)

    def search_for_debug(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        use_rerank: bool = False,
    ) -> dict:
        """Returns vector results and optionally reranked results for the debug dashboard.

        Returns a dict:
            "vector"   : list[tuple[Document, float]] — (doc, relevance_score), fetch_k items
            "reranked" : list[tuple[Document, float]] | None — (doc, rerank_score), k items
                         None when use_rerank is False or no reranker is configured.
        """
        log.info("search_for_debug: query=%r  k=%d  fetch_k=%d  use_rerank=%s  doc_id=%s",
                 query, k, fetch_k, use_rerank, doc_id)
        vector_results = self.similarity_search_with_scores(query, k=fetch_k, doc_id=doc_id)
        log.info("  vector results: %d", len(vector_results))

        if not use_rerank or self.reranker is None:
            if use_rerank and self.reranker is None:
                log.warning("  use_rerank=True but no reranker is configured — skipping")
            return {"vector": vector_results, "reranked": None}

        log.info("  reranking %d candidates → top %d", len(vector_results), k)
        docs = [doc for doc, _ in vector_results]
        reranked = self.reranker.rerank_with_scores(query, docs, top_k=k)
        log.info("  reranked results: %d", len(reranked))
        return {"vector": vector_results, "reranked": reranked}

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
        """Delegates to RAGEngine.answer — see engine.py for the full pipeline."""
        return self.engine.answer(
            query, k=k, fetch_k=fetch_k, doc_id=doc_id, expand_query=expand_query
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_texts(self, texts, metadatas=None, ids=None):
        """Adds raw texts to Chroma."""
        if hasattr(self.db, "add_texts"):
            return self.db.add_texts(texts, metadatas=metadatas, ids=ids)
        self.db = Chroma.from_texts(
            texts, embedding=self.embed, persist_directory=self.persist_directory
        )

    def add_pdf(
        self,
        pdf_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        doc_id: str | None = None,
    ):
        """Loads a PDF, splits it into chunks with metadata, and indexes via Indexer.

        doc_id  : identifier stored in chunk metadata for retrieval-time filtering;
                  defaults to the PDF filename when not provided.
        """
        try:
            chunks = load_and_chunk_pdf(
                pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, doc_id=doc_id
            )
            log.info("Loaded PDF: %d chunks (chunk_size=%d, overlap=%d)",
                     len(chunks), chunk_size, chunk_overlap)
            log.debug("Sample metadata: %s", chunks[0].metadata if chunks else "n/a")
            self.indexer.run(chunks)
        except Exception as e:
            log.error("Error loading PDF: %s", e, exc_info=True)

    def persist(self):
        """Force write to disk (newer Chroma versions auto-persist when persist_directory is set)."""
        pass  # langchain_chroma >= 0.1 auto-persists when persist_directory is set
