import os
from time import perf_counter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config import AppConfig
from utils.file_processor import load_and_chunk_pdf, write_json
from utils.logger import AppLogger
from rag.engine import RAGEngine
from rag.indexer import Indexer
from rag.ingest.document_ingester import DocumentIngester
from rag.knowledge.extractor import KnowledgeExtractor
from rag.knowledge.qa_generator import QAGenerator
from rag.reranker import RerankerFactory
from rag.retrieval.bm25 import BM25Index, rrf_fuse
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
        log.info("Opening Chroma store → %s  collection=%s",
                 config.persist_directory, config.setup_rag_collection)
        self.persist_directory = config.persist_directory
        self.db = Chroma(
            persist_directory=config.persist_directory,
            embedding_function=self.embed,
            collection_name=config.setup_rag_collection or "rag_collection",
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

        # BM25 index — built lazily on first hybrid search request
        self.bm25_index = BM25Index()
        self._bm25_dirty = True  # rebuild needed before first use

        # Knowledge extractor — used for B.1 semantic enrichment
        self.extractor = KnowledgeExtractor(self.llm)

        # QA generator + dedicated QA index — Stage B.2
        _qa_collection = (config.setup_rag_collection or "rag_collection") + "_qa"
        self.qa_db = Chroma(
            persist_directory=config.persist_directory,
            embedding_function=self.embed,
            collection_name=_qa_collection,
        )
        self.qa_indexer = Indexer(
            db=self.qa_db,
            namespace=_qa_collection,
            db_url=config.db_url,
            batch_limit=config.batch_limit,
        )
        self.qa_generator = QAGenerator(self.llm)

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

    def list_doc_title_map(self) -> dict[str, str]:
        """Return a {doc_id: display_title} map for all indexed documents.

        *display_title* is the ``title`` metadata field when set, otherwise
        falls back to the raw ``doc_id``.  Used by filter dropdowns that show
        human-readable names while still filtering by doc_id.
        """
        result = self.db.get(include=["metadatas"])
        mapping: dict[str, str] = {}
        for m in (result.get("metadatas") or []):
            if not m:
                continue
            doc_id = m.get("doc_id")
            if not doc_id or doc_id in mapping:
                continue
            title = (m.get("title") or "").strip() or doc_id
            mapping[doc_id] = title
        return dict(sorted(mapping.items()))

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

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def rebuild_bm25(self) -> None:
        """Fetch all indexed documents from Chroma and rebuild the BM25 index.

        Called automatically before the first hybrid search after a dirty flag
        is set by ``invalidate_bm25()``.
        """
        log.info("rebuild_bm25: fetching all documents from Chroma …")
        result = self.db.get(include=["documents", "metadatas"])
        docs: list[Document] = []
        for text, meta in zip(
            result.get("documents") or [], result.get("metadatas") or []
        ):
            if text:
                docs.append(Document(page_content=text, metadata=meta or {}))
        self.bm25_index.build(docs)
        self._bm25_dirty = False
        log.info("rebuild_bm25 done: %d documents", len(docs))

    def invalidate_bm25(self) -> None:
        """Mark the BM25 index as stale so it is rebuilt before next hybrid search."""
        self._bm25_dirty = True
        log.debug("BM25 index invalidated")

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def hybrid_search_with_scores(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        search_filter: "SearchFilter | None" = None,
    ) -> tuple[
        list[tuple[Document, float]],
        list[tuple[Document, float]],
        list[tuple[Document, float]],
    ]:
        """Run vector search + BM25 and merge via RRF.

        Returns:
            (vector_results, bm25_results, fused_results)
            Each is a list of (Document, score) pairs sorted best-first.
            *vector_results* and *bm25_results* each contain up to *fetch_k*
            items; *fused_results* contains up to *fetch_k* items (candidates
            for downstream reranking or as final results when top_k < fetch_k).
        """
        if self._bm25_dirty:
            self.rebuild_bm25()

        where = self._where(search_filter=search_filter)

        vector = self.similarity_search_with_scores(
            query, k=fetch_k, search_filter=search_filter
        )
        bm25 = self.bm25_index.search(query, k=fetch_k, where=where)
        fused = rrf_fuse(vector, bm25, top_k=fetch_k)

        log.debug(
            "hybrid_search: vector=%d  bm25=%d  fused=%d",
            len(vector), len(bm25), len(fused),
        )
        return vector, bm25, fused

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
        """Returns retrieval results for the debug dashboard.

        Args:
            search_filter : multi-dimension filter; takes precedence over *doc_id*.
            doc_id        : kept for backward compatibility.
            use_hybrid    : when True, also runs BM25 and fuses via RRF.

        Returns a dict with keys:
            "vector"   : list[tuple[Document, float]] — raw vector results (fetch_k)
            "bm25"     : list[tuple[Document, float]] | None — BM25 results (fetch_k)
                         None when use_hybrid is False.
            "hybrid"   : list[tuple[Document, float]] | None — RRF-fused results (fetch_k)
                         None when use_hybrid is False.
            "reranked" : list[tuple[Document, float]] | None — top-k reranked candidates.
                         When hybrid is ON, reranking uses fused candidates as input.
                         None when use_rerank is False or no reranker is configured.
        """
        filter_summary = search_filter.summary() if search_filter else doc_id
        log.info(
            "search_for_debug: query=%r  k=%d  fetch_k=%d"
            "  use_rerank=%s  use_hybrid=%s  filter=%s",
            query, k, fetch_k, use_rerank, use_hybrid, filter_summary,
        )

        bm25_results: list[tuple[Document, float]] | None = None
        hybrid_results: list[tuple[Document, float]] | None = None

        if use_hybrid:
            vector_results, bm25_results, hybrid_results = self.hybrid_search_with_scores(
                query, k=k, fetch_k=fetch_k, search_filter=search_filter
            )
            rerank_pool = [doc for doc, _ in hybrid_results]
        else:
            vector_results = self.similarity_search_with_scores(
                query, k=fetch_k, doc_id=doc_id, search_filter=search_filter,
            )
            rerank_pool = [doc for doc, _ in vector_results]

        log.info(
            "  vector=%d  bm25=%s  hybrid=%s",
            len(vector_results),
            len(bm25_results) if bm25_results is not None else "off",
            len(hybrid_results) if hybrid_results is not None else "off",
        )

        reranked: list[tuple[Document, float]] | None = None
        if use_rerank:
            if self.reranker is not None:
                log.info("  reranking %d candidates → top %d", len(rerank_pool), k)
                reranked = self.reranker.rerank_with_scores(query, rerank_pool, top_k=k)
                log.info("  reranked results: %d", len(reranked))
            else:
                log.warning("  use_rerank=True but no reranker is configured — skipping")

        return {
            "vector": vector_results,
            "bm25": bm25_results,
            "hybrid": hybrid_results,
            "reranked": reranked,
        }

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
        """Like search_for_debug but also returns per-step timing in ``trace``.

        The ``trace`` key is a list of dicts with keys:
            stage       : str   — step name
            elapsed_ms  : float — wall time for this step
            in_count    : int   — input pool size from the previous step
            out_count   : int   — output count after this step
            docs        : list[tuple[Document, float]] — top-5 preview
            params      : dict  — stage-specific parameters
        """
        trace: list[dict] = []

        if use_hybrid:
            if self._bm25_dirty:
                self.rebuild_bm25()
            where = self._where(search_filter=search_filter)

            t0 = perf_counter()
            vector_results = self.similarity_search_with_scores(
                query, k=fetch_k, search_filter=search_filter
            )
            vector_ms = (perf_counter() - t0) * 1000
            trace.append({
                "stage": "Vector Search",
                "elapsed_ms": round(vector_ms, 1),
                "in_count": 0,
                "out_count": len(vector_results),
                "docs": vector_results[:5],
                "params": {"fetch_k": fetch_k},
            })

            t0 = perf_counter()
            bm25_results = self.bm25_index.search(query, k=fetch_k, where=where)
            bm25_ms = (perf_counter() - t0) * 1000
            trace.append({
                "stage": "BM25 Search",
                "elapsed_ms": round(bm25_ms, 1),
                "in_count": 0,
                "out_count": len(bm25_results),
                "docs": bm25_results[:5],
                "params": {"fetch_k": fetch_k},
            })

            vector_ids = {d.metadata.get("chunk_id") for d, _ in vector_results}
            bm25_ids   = {d.metadata.get("chunk_id") for d, _ in bm25_results}
            overlap    = len(vector_ids & bm25_ids)

            t0 = perf_counter()
            hybrid_results = rrf_fuse(vector_results, bm25_results, top_k=fetch_k)
            rrf_ms = (perf_counter() - t0) * 1000
            trace.append({
                "stage": "RRF Merge",
                "elapsed_ms": round(rrf_ms, 1),
                "in_count": len(vector_results) + len(bm25_results),
                "out_count": len(hybrid_results),
                "docs": hybrid_results[:5],
                "params": {
                    "overlap": overlap,
                    "total_unique": len(vector_ids | bm25_ids),
                },
            })
            rerank_pool = [doc for doc, _ in hybrid_results]
        else:
            t0 = perf_counter()
            vector_results = self.similarity_search_with_scores(
                query, k=fetch_k, doc_id=doc_id, search_filter=search_filter,
            )
            vector_ms = (perf_counter() - t0) * 1000
            trace.append({
                "stage": "Vector Search",
                "elapsed_ms": round(vector_ms, 1),
                "in_count": 0,
                "out_count": len(vector_results),
                "docs": vector_results[:5],
                "params": {"fetch_k": fetch_k},
            })
            bm25_results   = None
            hybrid_results = None
            rerank_pool    = [doc for doc, _ in vector_results]

        reranked: list[tuple[Document, float]] | None = None
        if use_rerank and self.reranker is not None:
            t0 = perf_counter()
            reranked  = self.reranker.rerank_with_scores(query, rerank_pool, top_k=k)
            rerank_ms = (perf_counter() - t0) * 1000
            trace.append({
                "stage": "Rerank",
                "elapsed_ms": round(rerank_ms, 1),
                "in_count": len(rerank_pool),
                "out_count": len(reranked),
                "docs": reranked[:5],
                "params": {"top_k": k},
            })

        final      = reranked if reranked else (hybrid_results if hybrid_results else vector_results)
        final_docs = final[:k]
        trace.append({
            "stage": "Final Context",
            "elapsed_ms": 0.0,
            "in_count": len(final),
            "out_count": len(final_docs),
            "docs": final_docs,
            "params": {"top_k": k},
        })

        log.debug(
            "search_for_trace: %d trace steps  total=%.1fms",
            len(trace),
            sum(s["elapsed_ms"] for s in trace),
        )
        return {
            "vector":   vector_results,
            "bm25":     bm25_results,
            "hybrid":   hybrid_results,
            "reranked": reranked,
            "trace":    trace,
        }

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
        extract_knowledge: bool = False,
    ) -> dict:
        """Ingest any supported document type (PDF, Markdown, plain text).

        Dispatches to the appropriate parser via DocumentIngester, then indexes
        the resulting chunks through Indexer.run().

        Args:
            path              : path to the document file.
            chunk_size        : maximum characters per chunk.
            chunk_overlap     : overlap between consecutive chunks.
            doc_id            : document-level identifier; defaults to filename.
            extract_knowledge : when True, call KnowledgeExtractor.enrich() before
                                indexing to stamp ka_summary / ka_keywords / ka_entities /
                                ka_topics / ka_questions onto every chunk.  Requires one
                                LLM call per chunk — opt-in to avoid cost on bulk loads.

        Returns:
            Stats dict from Indexer.run() with keys:
            num_added, num_updated, num_skipped, num_deleted.
        """
        try:
            chunks = self.ingester.ingest(
                path, doc_id=doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            log.info(
                "add_document: %s  %d chunks  doc_id=%r  extract_knowledge=%s",
                path, len(chunks), doc_id, extract_knowledge,
            )
            if extract_knowledge:
                chunks = self.extractor.enrich(chunks)
            return self.indexer.run(chunks)
        except Exception as e:
            log.error("add_document failed: %s", e, exc_info=True)
            raise

    def enrich_doc(
        self,
        doc_id: str,
        overwrite: bool = False,
    ) -> dict:
        """Run knowledge extraction on all chunks for *doc_id* already in Chroma.

        Fetches each chunk's text and existing metadata, calls KnowledgeExtractor,
        then updates the Chroma collection in-place (no re-embedding needed).

        This is the post-hoc enrichment path for documents that were indexed
        before B.1 was introduced, or when ``extract_knowledge=False`` was used.

        Args:
            doc_id    : The ``doc_id`` (= ``source_id`` / ``page_id``) to enrich.
            overwrite : When False (default), skip chunks that already have a
                        non-empty ``ka_summary``.  Set True to re-extract everything.

        Returns:
            dict with keys ``enriched``, ``skipped``, ``failed``.
        """
        log.info("enrich_doc: doc_id=%r  overwrite=%s", doc_id, overwrite)

        result = self.db.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["documents", "metadatas"],
        )
        ids       = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not ids:
            log.warning("enrich_doc: no chunks found for doc_id=%r", doc_id)
            return {"enriched": 0, "skipped": 0, "failed": 0}

        stats = {"enriched": 0, "skipped": 0, "failed": 0}
        new_ids, new_metas = [], []

        for chroma_id, text, meta in zip(ids, documents, metadatas):
            meta = meta or {}
            if not overwrite and KnowledgeExtractor.is_enriched(meta):
                stats["skipped"] += 1
                log.debug("  skip (already enriched): %s", chroma_id)
                continue

            artifact = self.extractor.extract_one(text or "")
            if not artifact.get("ka_summary"):
                stats["failed"] += 1
                log.warning("  extraction produced no summary: %s", chroma_id)
                continue

            new_ids.append(chroma_id)
            new_metas.append({**meta, **artifact})
            stats["enriched"] += 1

        if new_ids:
            self.db._collection.update(ids=new_ids, metadatas=new_metas)
            log.info(
                "enrich_doc done: enriched=%d  skipped=%d  failed=%d",
                stats["enriched"], stats["skipped"], stats["failed"],
            )

        return stats

    def generate_qa(self, doc_id: str, overwrite: bool = False) -> dict:
        """Generate and index QA pairs for all chunks of *doc_id*.

        Fetches chunks from the main collection, generates QA pairs via LLM,
        and writes them to the dedicated QA collection (``<collection>_qa``).

        Each QA pair is stored with the *question* as page_content so that
        semantic search on user queries matches questions directly.

        Args:
            doc_id    : The document identifier to generate QA pairs for.
            overwrite : When False (default), skip if any QA pairs already
                        exist for this doc_id.  When True, delete existing
                        pairs first and regenerate.

        Returns:
            dict with keys ``generated``, ``indexed``, ``skipped``, ``failed``.
        """
        log.info("generate_qa: doc_id=%r  overwrite=%s", doc_id, overwrite)

        existing     = self.qa_db.get(where={"doc_id": {"$eq": doc_id}}, include=["metadatas"])
        existing_ids = existing.get("ids") or []

        if existing_ids and not overwrite:
            log.info(
                "generate_qa: %d QA pairs already exist, skipping (overwrite=False)",
                len(existing_ids),
            )
            return {"generated": 0, "indexed": 0, "skipped": len(existing_ids), "failed": 0}

        if existing_ids:
            self.qa_db._collection.delete(where={"doc_id": {"$eq": doc_id}})
            log.info("generate_qa: deleted %d existing QA pairs", len(existing_ids))

        result    = self.db.get(where={"doc_id": {"$eq": doc_id}}, include=["documents", "metadatas"])
        ids       = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not ids:
            log.warning("generate_qa: no chunks found for doc_id=%r", doc_id)
            return {"generated": 0, "indexed": 0, "skipped": 0, "failed": 0}

        source_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(documents, metadatas)
            if text
        ]

        pairs = self.qa_generator.generate_for_docs(source_docs)
        if not pairs:
            return {"generated": 0, "indexed": 0, "skipped": 0, "failed": len(source_docs)}

        stats = self.qa_indexer.run([p.to_document() for p in pairs])
        indexed = stats.get("num_added", 0) + stats.get("num_updated", 0)
        log.info("generate_qa done: generated=%d  indexed=%d", len(pairs), indexed)
        return {"generated": len(pairs), "indexed": indexed, "skipped": 0, "failed": 0}

    def qa_search(self, query: str, k: int = 5) -> list[dict]:
        """Search the QA index for questions matching *query*.

        Returns a ranked list of QA pairs whose questions are semantically
        close to *query*.  Each entry contains the matched question, the
        generated answer, and back-references to the source chunk and document.

        Args:
            query : Natural-language query or question.
            k     : Maximum number of QA pairs to return.

        Returns:
            List of dicts with keys: ``question``, ``answer``, ``chunk_id``,
            ``doc_id``, ``score`` (float 0-1, higher = more similar).
        """
        raw = self.qa_db.similarity_search_with_score(query, k=k)
        return [
            {
                "question": doc.page_content,
                "answer":   doc.metadata.get("answer",   ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "doc_id":   doc.metadata.get("doc_id",   ""),
                "score":    round(1 / (1 + dist), 4),
            }
            for doc, dist in raw
        ]

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
