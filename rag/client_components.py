"""Construction helpers for LocalLlamaClient subsystem wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
from rag.memory.gateway import MemoryGateway, build_memory_gateway
from rag.memory.manager import ConversationMemory
from rag.memory.research_session import ResearchSessionManager
from rag.memory.store import MemoryStore
from rag.memory.timeline import KnowledgeTimeline
from rag.memory.user_memory import UserMemoryManager
from rag.reranker import RerankerFactory
from rag.retrieval.collections import resolve_doc_collection_name, resolve_qa_collection_name
from rag.retrieval.code_result_filter import CodeResultFilter
from rag.retrieval.document_retriever import DocumentRetriever
from rag.retrieval.pipeline import PipelineBuilder
from rag.retrieval.searcher import Searcher
from utils.config import AppConfig


@dataclass
class ClientComponents:
    """All constructed subsystems needed by LocalLlamaClient."""

    code_result_filter: CodeResultFilter
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
    memory_manager: MemoryGateway
    user_memory: UserMemoryManager
    timeline: KnowledgeTimeline
    research: ResearchSessionManager
    memory: ConversationMemory


@dataclass
class ModelComponents:
    """Model-layer dependencies used across client subsystems."""

    embed: object
    llm: ChatOpenAI


@dataclass
class StorageComponents:
    """Storage-layer dependencies shared by retrieval and knowledge flows."""

    db: Chroma
    qa_db: Chroma
    indexer: Indexer
    qa_indexer: Indexer
    mem_store: MemoryStore


@dataclass
class RetrievalComponents:
    """Retrieval-layer dependencies used by the engine."""

    ingester: DocumentIngester
    reranker: object
    searcher: Searcher
    doc_retriever: DocumentRetriever
    doc_pipeline: object
    engine: RAGEngine


@dataclass
class KnowledgeComponents:
    """Knowledge-layer dependencies for enrichment and QA."""

    knowledge: KnowledgeManager


@dataclass
class MemoryComponents:
    """Memory-layer dependencies for conversation and research state."""

    memory_manager: MemoryGateway
    user_memory: UserMemoryManager
    timeline: KnowledgeTimeline
    research: ResearchSessionManager
    memory: ConversationMemory


@dataclass
class ClientComponentProviders:
    """Composable provider set used by build_client_components.

    Replacing one provider allows backend/pipeline swaps without editing the
    central assembly flow.
    """

    code_result_filter_factory: Callable[[], CodeResultFilter]
    model_builder: Callable[[AppConfig], ModelComponents]
    storage_builder: Callable[[AppConfig, object], StorageComponents]
    retrieval_builder: Callable[[AppConfig, object, ChatOpenAI, Chroma], RetrievalComponents]
    knowledge_builder: Callable[[ChatOpenAI, Chroma, Chroma, Indexer], KnowledgeComponents]
    memory_builder: Callable[[AppConfig, ChatOpenAI, MemoryStore], MemoryComponents]


def default_client_component_providers() -> ClientComponentProviders:
    """Return the default provider composition used in production."""
    return ClientComponentProviders(
        code_result_filter_factory=CodeResultFilter,
        model_builder=build_model_components,
        storage_builder=lambda config, embed: build_storage_components(config, embed=embed),
        retrieval_builder=lambda config, embed, llm, db: build_retrieval_components(
            config,
            embed=embed,
            llm=llm,
            db=db,
        ),
        knowledge_builder=lambda llm, db, qa_db, qa_indexer: build_knowledge_components(
            llm=llm,
            db=db,
            qa_db=qa_db,
            qa_indexer=qa_indexer,
        ),
        memory_builder=lambda config, llm, mem_store: build_memory_components(
            config,
            llm=llm,
            mem_store=mem_store,
        ),
    )


def build_model_components(config: AppConfig) -> ModelComponents:
    """Build embedding and chat model dependencies."""
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

    llm = ChatOpenAI(
        base_url=config.llm_base,
        api_key=config.llm_api_key,
        model=config.llm_model,
        **config.llm_kwargs,
    )
    return ModelComponents(embed=embed, llm=llm)


def build_storage_components(config: AppConfig, *, embed: object) -> StorageComponents:
    """Build vector and persistence dependencies shared across subsystems."""
    collection_name = resolve_doc_collection_name(config.setup_rag_collection)
    qa_collection = resolve_qa_collection_name(collection_name)

    db = Chroma(
        persist_directory=config.persist_directory,
        embedding_function=embed,
        collection_name=collection_name,
    )
    qa_db = Chroma(
        persist_directory=config.persist_directory,
        embedding_function=embed,
        collection_name=qa_collection,
    )
    indexer = Indexer(
        db=db,
        namespace=collection_name,
        db_url=config.db_url,
        batch_limit=config.batch_limit,
    )
    qa_indexer = Indexer(
        db=qa_db,
        namespace=qa_collection,
        db_url=config.db_url,
        batch_limit=config.batch_limit,
    )
    mem_store = MemoryStore(db_path=config.memory_db_path)

    return StorageComponents(
        db=db,
        qa_db=qa_db,
        indexer=indexer,
        qa_indexer=qa_indexer,
        mem_store=mem_store,
    )


def build_retrieval_components(
    config: AppConfig,
    *,
    embed: object,
    llm: ChatOpenAI,
    db: Chroma,
) -> RetrievalComponents:
    """Build retrieval-layer dependencies and the default document engine."""
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

    return RetrievalComponents(
        ingester=ingester,
        reranker=reranker,
        searcher=searcher,
        doc_retriever=doc_retriever,
        doc_pipeline=doc_pipeline,
        engine=engine,
    )


def build_knowledge_components(
    *,
    llm: ChatOpenAI,
    db: Chroma,
    qa_db: Chroma,
    qa_indexer: Indexer,
) -> KnowledgeComponents:
    """Build knowledge-enrichment and QA dependencies."""
    knowledge = KnowledgeManager(
        db=db,
        qa_db=qa_db,
        qa_indexer=qa_indexer,
        extractor=KnowledgeExtractor(llm),
        qa_generator=QAGenerator(llm),
        clusterer=TopicClusterer(llm=llm, db=db),
        linker=CrossDocLinker(db=db),
    )
    return KnowledgeComponents(knowledge=knowledge)


def build_memory_components(
    config: AppConfig,
    *,
    llm: ChatOpenAI,
    mem_store: MemoryStore,
) -> MemoryComponents:
    """Build memory, profile, timeline, and research-session dependencies."""
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
    memory_manager = build_memory_gateway(
        memory=memory,
        user_memory=user_memory,
        timeline=timeline,
        research=research,
    )

    return MemoryComponents(
        memory_manager=memory_manager,
        user_memory=user_memory,
        timeline=timeline,
        research=research,
        memory=memory,
    )


def build_client_components(
    config: AppConfig,
    *,
    providers: ClientComponentProviders | None = None,
) -> ClientComponents:
    """Build all subsystems used by LocalLlamaClient through layered factories.

    A custom provider set enables backend and pipeline replacement without
    changing this assembly function.
    """
    provider_set = providers or default_client_component_providers()
    code_result_filter = provider_set.code_result_filter_factory()
    models = provider_set.model_builder(config)
    storage = provider_set.storage_builder(config, models.embed)
    retrieval = provider_set.retrieval_builder(config, models.embed, models.llm, storage.db)
    knowledge = provider_set.knowledge_builder(
        models.llm,
        storage.db,
        storage.qa_db,
        storage.qa_indexer,
    )
    memory = provider_set.memory_builder(config, models.llm, storage.mem_store)
    retrieval.engine.memory = memory.memory

    return ClientComponents(
        code_result_filter=code_result_filter,
        embed=models.embed,
        db=storage.db,
        llm=models.llm,
        indexer=storage.indexer,
        ingester=retrieval.ingester,
        reranker=retrieval.reranker,
        searcher=retrieval.searcher,
        doc_retriever=retrieval.doc_retriever,
        doc_pipeline=retrieval.doc_pipeline,
        engine=retrieval.engine,
        knowledge=knowledge.knowledge,
        memory_manager=memory.memory_manager,
        user_memory=memory.user_memory,
        timeline=memory.timeline,
        research=memory.research,
        memory=memory.memory,
    )
