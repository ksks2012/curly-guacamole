"""Construction helpers for LocalLlamaClient subsystem wiring."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.embeddings import OpenRouterEmbeddings
from rag.engine import RAGEngine
from rag.indexer import Indexer
from rag.ingest.document_ingester import DocumentIngester
from rag.knowledge.clusterer import TopicClusterer
from rag.knowledge.extractor import KnowledgeExtractor
from rag.knowledge.linker import CrossDocLinker
from rag.knowledge.manager import KnowledgeManager
from rag.knowledge.qa_generator import QAGenerator
from rag.memory.manager import ConversationMemory
from rag.memory.research_session import ResearchSessionManager
from rag.memory.store import MemoryStore
from rag.memory.timeline import KnowledgeTimeline
from rag.memory.user_memory import UserMemoryManager
from rag.reranker import RerankerFactory
from rag.retrieval.document_retriever import DocumentRetriever
from rag.retrieval.pipeline import PipelineBuilder
from rag.retrieval.searcher import Searcher
from utils.config import AppConfig


@dataclass
class ClientComponents:
    """All constructed subsystems needed by LocalLlamaClient."""

    embed: object
    db: Chroma
    llm: ChatOpenAI
    indexer: Indexer
    ingester: DocumentIngester
    reranker: object
    searcher: Searcher
    doc_retriever: DocumentRetriever
    doc_pipeline: object
    engine: RAGEngine
    knowledge: KnowledgeManager
    user_memory: UserMemoryManager
    timeline: KnowledgeTimeline
    research: ResearchSessionManager
    memory: ConversationMemory


def build_client_components(config: AppConfig) -> ClientComponents:
    """Build all subsystems used by LocalLlamaClient."""
    if config.model_provider == "openrouter":
        embed = OpenRouterEmbeddings(
            model=config.embed_model,
            api_key=config.embed_api_key,
            base_url=config.embed_base,
            requests_per_minute=config.requests_rate_limit,
        )
    else:
        embed = OpenAIEmbeddings(
            openai_api_key=config.embed_api_key,
            openai_api_base=config.embed_base,
            model=config.embed_model,
        )

    db = Chroma(
        persist_directory=config.persist_directory,
        embedding_function=embed,
        collection_name=config.setup_rag_collection or "rag_collection",
    )

    llm = ChatOpenAI(
        base_url=config.llm_base,
        api_key=config.llm_api_key,
        model=config.llm_model,
        **config.llm_kwargs,
    )

    indexer = Indexer(
        db=db,
        namespace=config.setup_rag_collection,
        db_url=config.db_url,
        batch_limit=config.batch_limit,
    )
    ingester = DocumentIngester(embeddings=embed)
    reranker = RerankerFactory.build(config, llm=llm)
    searcher = Searcher(db=db, reranker=reranker)

    doc_retriever = DocumentRetriever(
        searcher,
        use_hybrid=False,
        reranker=reranker,
    )
    doc_pipeline = PipelineBuilder.document_pipeline(
        doc_retriever,
        reranker=reranker,
    )

    engine = RAGEngine(
        llm=llm,
        retriever=doc_pipeline,
        reranker=reranker,
        config=config,
    )

    qa_collection = (config.setup_rag_collection or "rag_collection") + "_qa"
    qa_db = Chroma(
        persist_directory=config.persist_directory,
        embedding_function=embed,
        collection_name=qa_collection,
    )
    knowledge = KnowledgeManager(
        db=db,
        qa_db=qa_db,
        qa_indexer=Indexer(
            db=qa_db,
            namespace=qa_collection,
            db_url=config.db_url,
            batch_limit=config.batch_limit,
        ),
        extractor=KnowledgeExtractor(llm),
        qa_generator=QAGenerator(llm),
        clusterer=TopicClusterer(llm=llm, db=db),
        linker=CrossDocLinker(db=db),
    )

    mem_store = MemoryStore(db_path=config.memory_db_path)
    user_memory = UserMemoryManager(store=mem_store)
    timeline = KnowledgeTimeline(store=mem_store)
    research = ResearchSessionManager(store=mem_store)
    memory = ConversationMemory(
        store=mem_store,
        llm=llm,
        session_id=config.memory_default_session,
        max_recent=config.memory_max_recent,
        max_topics=config.memory_max_topics,
        extract_topics=config.memory_extract_topics,
        auto_infer_project=config.memory_auto_infer_project,
        user_memory=user_memory,
        timeline=timeline,
        research=research,
    )
    memory.ensure_session()
    engine.memory = memory

    return ClientComponents(
        embed=embed,
        db=db,
        llm=llm,
        indexer=indexer,
        ingester=ingester,
        reranker=reranker,
        searcher=searcher,
        doc_retriever=doc_retriever,
        doc_pipeline=doc_pipeline,
        engine=engine,
        knowledge=knowledge,
        user_memory=user_memory,
        timeline=timeline,
        research=research,
        memory=memory,
    )
