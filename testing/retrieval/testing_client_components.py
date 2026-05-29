from types import SimpleNamespace

import rag.client_components as mod


def _make_config(**overrides):
    data = {
        "model_provider": "openai",
        "embed_model": "embed-model",
        "embed_api_key": "embed-key",
        "embed_base": "https://embed.example",
        "requests_rate_limit": 12,
        "llm_base": "https://llm.example",
        "llm_api_key": "llm-key",
        "llm_model": "llm-model",
        "llm_kwargs": {"temperature": 0.1},
        "persist_directory": "/tmp/chroma",
        "setup_rag_collection": "docs",
        "db_url": "sqlite:///tmp.db",
        "batch_limit": 25,
        "memory_db_path": "/tmp/memory.db",
        "memory_default_session": "default",
        "memory_max_recent": 10,
        "memory_max_topics": 5,
        "memory_extract_topics": True,
        "memory_auto_infer_project": False,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_build_model_components_uses_openrouter(monkeypatch):
    config = _make_config(model_provider="openrouter")
    calls = {}

    class _FakeOpenRouterEmbeddings:
        def __init__(self, **kwargs):
            calls["embed"] = kwargs

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls["llm"] = kwargs

    monkeypatch.setattr(mod, "OpenRouterEmbeddings", _FakeOpenRouterEmbeddings)
    monkeypatch.setattr(mod, "ChatOpenAI", _FakeChatOpenAI)

    built = mod.build_model_components(config)

    assert isinstance(built.embed, _FakeOpenRouterEmbeddings)
    assert isinstance(built.llm, _FakeChatOpenAI)
    assert calls["embed"]["model"] == config.embed_model
    assert calls["embed"]["requests_per_minute"] == config.requests_rate_limit
    assert calls["llm"]["model"] == config.llm_model


def test_build_storage_components_creates_main_and_qa_storage(monkeypatch):
    config = _make_config(setup_rag_collection="knowledge")
    chroma_calls = []
    indexer_calls = []

    class _FakeChroma:
        def __init__(self, **kwargs):
            chroma_calls.append(kwargs)

    class _FakeIndexer:
        def __init__(self, **kwargs):
            indexer_calls.append(kwargs)

    class _FakeMemoryStore:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(mod, "Chroma", _FakeChroma)
    monkeypatch.setattr(mod, "Indexer", _FakeIndexer)
    monkeypatch.setattr(mod, "MemoryStore", _FakeMemoryStore)

    built = mod.build_storage_components(config, embed="embed")

    assert len(chroma_calls) == 2
    assert chroma_calls[0]["collection_name"] == "knowledge"
    assert chroma_calls[1]["collection_name"] == "knowledge_qa"
    assert indexer_calls[0]["namespace"] == "knowledge"
    assert indexer_calls[1]["namespace"] == "knowledge_qa"
    assert built.mem_store.kwargs["db_path"] == config.memory_db_path


def test_build_client_components_wires_layer_outputs(monkeypatch):
    config = _make_config()
    engine = SimpleNamespace(memory=None)
    models = mod.ModelComponents(embed="embed", llm="llm")
    storage = mod.StorageComponents(
        db="db",
        qa_db="qa_db",
        indexer="indexer",
        qa_indexer="qa_indexer",
        mem_store="mem_store",
    )
    retrieval = mod.RetrievalComponents(
        ingester="ingester",
        reranker="reranker",
        searcher="searcher",
        doc_retriever="doc_retriever",
        doc_pipeline="doc_pipeline",
        engine=engine,
    )
    knowledge = mod.KnowledgeComponents(knowledge="knowledge")
    memory = mod.MemoryComponents(
        memory_manager="memory_manager",
        user_memory="user_memory",
        timeline="timeline",
        research="research",
        memory="memory",
    )

    monkeypatch.setattr(mod, "build_model_components", lambda cfg: models)
    monkeypatch.setattr(mod, "build_storage_components", lambda cfg, *, embed: storage)
    monkeypatch.setattr(
        mod,
        "build_retrieval_components",
        lambda cfg, *, embed, llm, db: retrieval,
    )
    monkeypatch.setattr(
        mod,
        "build_knowledge_components",
        lambda *, llm, db, qa_db, qa_indexer: knowledge,
    )
    monkeypatch.setattr(
        mod,
        "build_memory_components",
        lambda cfg, *, llm, mem_store: memory,
    )

    built = mod.build_client_components(config)

    assert isinstance(built.code_result_filter, mod.CodeResultFilter)
    assert built.embed == "embed"
    assert built.db == "db"
    assert built.indexer == "indexer"
    assert built.engine is engine
    assert built.knowledge == "knowledge"
    assert built.memory_manager == "memory_manager"
    assert built.memory == "memory"
    assert engine.memory == "memory"