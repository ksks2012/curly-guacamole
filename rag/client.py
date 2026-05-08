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
from rag.reranker import RerankerFactory
from rag.retrieval.filters import SearchFilter

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

        self.ingester = DocumentIngester(embeddings=self.embed)

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _where(
        self,
        doc_id: str | None = None,
        search_filter: "SearchFilter | None" = None,
    ) -> dict | None:
        """Build a Chroma ``where`` dict from either a SearchFilter or a bare doc_id.

        search_filter takes precedence when both are supplied.
        Returns None when no constraint is active (no filtering).
        """
        if search_filter is not None and not search_filter.is_empty():
            return search_filter.to_chroma()
        if doc_id is not None:
            return {"doc_id": {"$eq": doc_id}}
        return None

    def similarity_search(self, query: str, k: int = 4, doc_id: str | None = None):
        """Returns a list of similar documents from Chroma (LangChain Document objects)."""
        kwargs: dict = {"k": k}
        where = self._where(doc_id=doc_id)
        if where:
            kwargs["filter"] = where
        return self.db.similarity_search(query, **kwargs)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        doc_id: str | None = None,
        search_filter: "SearchFilter | None" = None,
    ) -> list[tuple[Document, float]]:
        """Returns (Document, score) pairs sorted best-first (lower L2 = better).

        Chroma L2 distance is converted to a 0-1 relevance score:
        ``relevance = 1 / (1 + distance)``.

        Args:
            search_filter : takes precedence over *doc_id* when supplied.
            doc_id        : kept for backward compatibility.
        """
        log.debug(
            "similarity_search_with_scores: query=%r  k=%d  filter=%s",
            query, k, search_filter.summary() if search_filter else doc_id,
        )
        kwargs: dict = {"k": k}
        where = self._where(doc_id=doc_id, search_filter=search_filter)
        if where:
            kwargs["filter"] = where
        raw = self.db.similarity_search_with_score(query, **kwargs)
        log.debug("  raw results: %d  (L2 distances: %s)",
                  len(raw), [round(d, 4) for _, d in raw])
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

    def list_field_values(self, field: str) -> list[str]:
        """Return distinct non-empty values for *field* across all indexed chunks.

        For the 'tags' field the comma-joined values are split into individual
        tag strings before deduplication.
        """
        result = self.db.get(include=["metadatas"])
        values: set[str] = set()
        for m in (result.get("metadatas") or []):
            if not m:
                continue
            raw = m.get(field, "")
            if not raw:
                continue
            if field == "tags":
                for t in str(raw).split(","):
                    t = t.strip()
                    if t:
                        values.add(t)
            else:
                values.add(str(raw))
        return sorted(values)

    def list_workspaces(self) -> list[str]:
        return self.list_field_values("workspace")

    def list_document_types(self) -> list[str]:
        return self.list_field_values("document_type")

    def list_tags(self) -> list[str]:
        return self.list_field_values("tags")

    def search_for_debug(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        use_rerank: bool = False,
        search_filter: "SearchFilter | None" = None,
    ) -> dict:
        """Returns vector results and optionally reranked results for the debug dashboard.

        Args:
            search_filter : multi-dimension filter; takes precedence over *doc_id*.
            doc_id        : kept for backward compatibility.

        Returns a dict:
            "vector"   : list[tuple[Document, float]] — (doc, relevance_score), fetch_k items
            "reranked" : list[tuple[Document, float]] | None — (doc, rerank_score), k items
                         None when use_rerank is False or no reranker is configured.
        """
        filter_summary = search_filter.summary() if search_filter else doc_id
        log.info(
            "search_for_debug: query=%r  k=%d  fetch_k=%d  use_rerank=%s  filter=%s",
            query, k, fetch_k, use_rerank, filter_summary,
        )
        vector_results = self.similarity_search_with_scores(
            query, k=fetch_k, doc_id=doc_id, search_filter=search_filter,
        )
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

    def add_document(
        self,
        path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        doc_id: str | None = None,
    ) -> dict:
        """Ingest any supported document type (PDF, Markdown, plain text).

        Dispatches to the appropriate parser via DocumentIngester, then indexes
        the resulting chunks through Indexer.run().

        Args:
            path         : path to the document file.
            chunk_size   : maximum characters per chunk.
            chunk_overlap: overlap between consecutive chunks.
            doc_id       : document-level identifier; defaults to filename.

        Returns:
            Stats dict from Indexer.run() with keys:
            num_added, num_updated, num_skipped, num_deleted.
        """
        try:
            chunks = self.ingester.ingest(
                path, doc_id=doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            log.info(
                "add_document: %s  %d chunks  doc_id=%r",
                path, len(chunks), doc_id,
            )
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
        """Force write to disk (newer Chroma versions auto-persist when persist_directory is set)."""
        pass  # langchain_chroma >= 0.1 auto-persists when persist_directory is set
