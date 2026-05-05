import json
import os

from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config import AppConfig
from utils.file_processor import load_and_chunk_pdf, write_json
from rag.reranker import BaseReranker, CrossEncoderReranker, LLMReranker
from rag.prompt import RAG_PROMPT, QUERY_EXPANSION_PROMPT

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

        # Embedding: points to your embedding server (llama.cpp server)
        self.embed = OpenAIEmbeddings(
            openai_api_key=config.api_key,
            openai_api_base=config.embed_base,
            model=config.embed_model,
        )

        # Vector store (Chroma)
        self.persist_directory = config.persist_directory
        self.db = Chroma(
            persist_directory=config.persist_directory,
            embedding_function=self.embed,
        )

        # LLM (chat) - points to your LLM server (also OpenAI-compatible)
        self.llm = ChatOpenAI(
            base_url=config.llm_base,
            api_key=config.api_key,
            model=config.llm_model,
            **config.llm_kwargs,
        )

        self.setup_rag_collection(
            config.setup_rag_collection,
            db_url=config.db_url,
        )

        # Reranker: built once and reused for every answer_query call.
        # LLMReranker is handled after self.llm is available.
        if config.reranker_type == "llm":
            self.reranker: BaseReranker | None = LLMReranker(self.llm)
        else:
            self.reranker = self._build_reranker(config)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reranker(config: AppConfig) -> BaseReranker | None:
        """Instantiate the configured reranker, or return None when disabled."""
        kind = config.reranker_type
        if kind == "cross_encoder":
            return CrossEncoderReranker(model_name=config.reranker_model)
        if kind == "llm":
            # LLMReranker needs the ChatOpenAI instance; defer init to first use
            # by storing config and building lazily inside answer_query.
            return None  # replaced in __init__ after llm is available
        return None  # 'none' or unknown → reranking disabled

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

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _expand_query(self, query: str, n: int) -> list[str]:
        """Returns n alternative phrasings of query using QUERY_EXPANSION_PROMPT.

        Falls back to an empty list so the caller can always safely combine
        results with the original query.
        """
        prompt = QUERY_EXPANSION_PROMPT.format(question=query, n=n)
        try:
            response = self.llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            expanded = json.loads(raw.strip())
            if isinstance(expanded, list):
                return [str(q) for q in expanded]
        except Exception as e:
            print(f"[query_expansion] failed, using original query only: {e}")
        return []

    def answer_query(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        doc_id: str | None = None,
        expand_query: bool | None = None,
    ):
        """Retrieves documents and generates a citation-grounded response.

        Pipeline:
          [optional] query expansion  → N extra phrasings via LLM
           ↓
          vector search per phrasing  (fetch_k candidates each, via MMR)
           ↓
          de-duplicate by chunk_id
           ↓
          reranker                    (narrows down to k docs when enabled)
           ↓
          LLM generation

        expand_query : override config.query_expansion_enabled for this call.
        """
        # Resolve expansion flag: per-call override wins, else use config
        use_expansion = (
            expand_query
            if expand_query is not None
            else self.config.query_expansion_enabled
        )

        if use_expansion:
            extra_queries = self._expand_query(query, n=self.config.query_expansion_n)
        else:
            extra_queries = []

        all_queries = [query] + extra_queries

        # Retrieve candidates for every phrasing and de-duplicate by chunk_id
        seen_ids: set[int] = set()
        candidates: list[Document] = []
        retriever = self.get_retriever(k=fetch_k, fetch_k=fetch_k, doc_id=doc_id)
        for q in all_queries:
            for doc in retriever.invoke(q):
                chunk_id = doc.metadata.get("chunk_id", id(doc))
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    candidates.append(doc)

        # Re-rank then keep top k
        if self.reranker is not None:
            docs = self.reranker.rerank(query, candidates, top_k=k)
        else:
            docs = candidates[:k]

        # Build context blocks with source tags so the LLM can cite them
        context_blocks = []
        for doc in docs:
            page = doc.metadata.get("page", "?")  # 0-based page from PyPDFLoader
            filename = doc.metadata.get("filename", "unknown")
            tag = f"[page {page + 1}, {filename}]"  # convert to 1-based for display
            context_blocks.append(f"{tag}\n{doc.page_content}")
        context = "\n\n".join(context_blocks)

        prompt = RAG_PROMPT.format(context=context, question=query)
        return self.llm.invoke(prompt)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_texts(self, texts, metadatas=None, ids=None):
        """Adds raw texts to Chroma."""
        if hasattr(self.db, "add_texts"):
            return self.db.add_texts(texts, metadatas=metadatas, ids=ids)
        # fallback: use from_texts with persist (heavier operation)
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
        """Loads a PDF, splits it into chunks with metadata, and indexes them in Chroma.

        doc_id  : identifier stored in chunk metadata for retrieval-time filtering;
                  defaults to the PDF filename when not provided.
        Uses run_indexing (via LangChain index()) as the single mechanism for adding
        documents to the vector store. Chroma.from_documents must NOT be called here,
        because it bypasses the record manager and causes duplicate documents to
        accumulate on every run.
        """
        try:
            chunks = load_and_chunk_pdf(
                pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, doc_id=doc_id
            )
            print(f"Loaded PDF: {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
            print(f"Sample metadata: {chunks[0].metadata if chunks else 'n/a'}")

            self.run_indexing(chunks)
        except Exception as e:
            print(f"Error loading PDF: {e}")

    def setup_rag_collection(
        self,
        namespace: str = "rag_collection",
        db_url: str = "sqlite:///record_manager_cache.sql",
    ):
        try:
            self.record_manager = SQLRecordManager(namespace, db_url=db_url)
            if not os.path.isfile(str(namespace)):
                self.record_manager.create_schema()
        except Exception as e:
            print(f"Error setting up RAG collection: {e}")

    def run_indexing(self, docs: list[Document], batch_limit: int = 100):
        indexing_stats = {
            "num_added": 0,
            "num_updated": 0,
            "num_skipped": 0,
            "num_deleted": 0,
        }
        try:
            # Resolve batch_limit from argument or client config
            if batch_limit is None:
                batch_limit = getattr(self, "config", None) and getattr(self.config, "batch_limit", 100) or 100

            if len(docs) > batch_limit:
                cleanup_config = "scoped_ids"
                # use a conservative batch size when scoped cleanup is used
                batch_size_config = min(100, batch_limit)
            else:
                cleanup_config = "incremental"
                batch_size_config = max(1, len(docs))

            indexing_stats = index(
                docs_source=docs,
                record_manager=self.record_manager,
                vector_store=self.db,
                cleanup=cleanup_config,
                source_id_key="source_id",
                key_encoder="sha256",
                batch_size=batch_size_config,  # use the configured batch size
            )
            print(f"Indexing status: {indexing_stats}")
        except Exception as e:
            print(f"Error during indexing: {e}")

        return indexing_stats

    def persist(self):
        """Force write to disk (newer Chroma versions auto-persist when persist_directory is set)."""
        pass  # langchain_chroma >= 0.1 auto-persists when persist_directory is set
