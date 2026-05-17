"""
Unit tests for Stage C.1 — Conversation Memory (testing/testing_memory.py).

Mock tests (no LLM, no live Chroma):
  1.  MemoryStore — create/read session
  2.  MemoryStore — save / get_recent_turns ordering and seq assignment
  3.  MemoryStore — delete_session cleans up turns
  4.  _merge_topics — dedup, max_topics, order preservation
  5.  ConversationMemory.ensure_session — creates and reloads session
  6.  ConversationMemory.add_turn — saves turn, updates rolling topics
  7.  ConversationMemory.add_turn — topic extraction called when llm present
  8.  ConversationMemory.add_turn — topic extraction failure is non-fatal
  9.  ConversationMemory.set_active_project — persists to store
 10.  ConversationMemory.build_context_block — correct format, empty when no state
 11.  ConversationMemory.clear_session — resets all state
 12.  ConversationMemory.max_recent — rolling window capped
 13.  RAGEngine.answer — memory injected into prompt, add_turn called
 14.  RAGEngine.answer — memory failure is non-fatal (fallback to plain prompt)

Live test (--no-live to skip):
 15.  Full round-trip: MemoryStore → add_turns → reload → build_context_block
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call
from typing import Any

LIVE = "--no-live" not in sys.argv


def _ok(name: str) -> None:
    print(f"{name}: OK")


def _fail(name: str, err: Exception) -> None:
    print(f"{name}: FAIL  {err}")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Test 1 — MemoryStore: create / read session
# ---------------------------------------------------------------------------

def test_store_session_create_read() -> None:
    from rag.memory.store import MemoryStore

    store = MemoryStore(db_path=":memory:")
    state = store.get_or_create_session("s1")
    assert state.session_id == "s1"
    assert state.active_project == ""
    assert state.current_topics == []
    assert state.created_at != ""

    # Second call returns same session
    state2 = store.get_or_create_session("s1")
    assert state2.session_id == "s1"
    assert state2.created_at == state.created_at
    _ok("MemoryStore.get_or_create_session")


# ---------------------------------------------------------------------------
# Test 2 — MemoryStore: save_turn + get_recent_turns ordering
# ---------------------------------------------------------------------------

def test_store_turns_ordering() -> None:
    from rag.memory.store import MemoryStore
    from rag.memory.models import ConversationTurn

    store = MemoryStore(db_path=":memory:")
    store.get_or_create_session("s1")

    for i in range(5):
        t = ConversationTurn(
            session_id="s1",
            question=f"Q{i}",
            answer_summary=f"A{i}",
            topics=[f"topic_{i}"],
        )
        saved = store.save_turn(t)
        assert saved.id is not None
        assert saved.seq == i + 1

    turns = store.get_recent_turns("s1", limit=3)
    # Should return last 3 in chronological order
    assert len(turns) == 3
    assert [t.seq for t in turns] == [3, 4, 5]
    _ok("MemoryStore.save_turn + get_recent_turns ordering")


# ---------------------------------------------------------------------------
# Test 3 — MemoryStore: delete_session removes turns
# ---------------------------------------------------------------------------

def test_store_delete_session() -> None:
    from rag.memory.store import MemoryStore
    from rag.memory.models import ConversationTurn

    store = MemoryStore(db_path=":memory:")
    store.get_or_create_session("s1")
    store.save_turn(ConversationTurn(session_id="s1", question="Q", answer_summary="A"))
    store.delete_session("s1")

    turns = store.get_recent_turns("s1")
    assert turns == []

    sessions = store.list_sessions()
    assert all(s["session_id"] != "s1" for s in sessions)
    _ok("MemoryStore.delete_session")


# ---------------------------------------------------------------------------
# Test 4 — _merge_topics
# ---------------------------------------------------------------------------

def test_merge_topics() -> None:
    from rag.memory.manager import _merge_topics

    result = _merge_topics(
        existing=["RAG", "Python", "LangChain"],
        new=["Memory", "RAG"],          # "RAG" is a duplicate
        max_topics=4,
    )
    assert result == ["Memory", "RAG", "Python", "LangChain"]

    # max_topics limits the result
    result2 = _merge_topics(["a", "b", "c", "d"], ["e"], max_topics=3)
    assert len(result2) == 3
    assert result2[0] == "e"
    _ok("_merge_topics")


# ---------------------------------------------------------------------------
# Test 5 — ConversationMemory.ensure_session
# ---------------------------------------------------------------------------

def test_memory_ensure_session() -> None:
    from rag.memory.store   import MemoryStore
    from rag.memory.manager import ConversationMemory

    store  = MemoryStore(db_path=":memory:")
    memory = ConversationMemory(store=store, llm=None, session_id="sess1")

    state = memory.ensure_session()
    assert state.session_id == "sess1"
    assert state.recent_questions == []

    # Calling again with a different id switches session
    memory.ensure_session("sess2")
    assert memory.session_id == "sess2"
    _ok("ConversationMemory.ensure_session")


# ---------------------------------------------------------------------------
# Test 6 — ConversationMemory.add_turn (no LLM → no topic extraction)
# ---------------------------------------------------------------------------

def test_memory_add_turn_no_llm() -> None:
    from rag.memory.store   import MemoryStore
    from rag.memory.manager import ConversationMemory

    store  = MemoryStore(db_path=":memory:")
    memory = ConversationMemory(
        store=store, llm=None, session_id="sess", extract_topics=False
    )
    memory.ensure_session()

    turn = memory.add_turn("What is RAG?", "RAG stands for Retrieval-Augmented Generation.")
    assert turn.id is not None
    assert turn.seq == 1
    assert turn.question == "What is RAG?"
    assert turn.answer_summary.startswith("RAG stands")
    assert turn.topics == []

    state = memory.get_state()
    assert len(state.recent_questions) == 1
    _ok("ConversationMemory.add_turn (no LLM)")


# ---------------------------------------------------------------------------
# Test 7 — ConversationMemory.add_turn: topic extraction called when llm set
# ---------------------------------------------------------------------------

def test_memory_add_turn_with_topic_extraction() -> None:
    from rag.memory.store   import MemoryStore
    from rag.memory.manager import ConversationMemory

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='["RAG", "Vector Search"]')

    store  = MemoryStore(db_path=":memory:")
    memory = ConversationMemory(
        store=store, llm=mock_llm, session_id="sess",
        extract_topics=True, auto_infer_project=False,
    )
    memory.ensure_session()

    turn = memory.add_turn("Explain RAG retrieval", "RAG uses vector search to find relevant docs.")
    assert "RAG" in turn.topics
    assert "Vector Search" in turn.topics

    state = memory.get_state()
    assert "RAG" in state.current_topics
    _ok("ConversationMemory.add_turn (topic extraction)")


# ---------------------------------------------------------------------------
# Test 8 — Topic extraction failure is non-fatal
# ---------------------------------------------------------------------------

def test_memory_topic_extraction_failure() -> None:
    from rag.memory.store   import MemoryStore
    from rag.memory.manager import ConversationMemory

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")

    store  = MemoryStore(db_path=":memory:")
    memory = ConversationMemory(
        store=store, llm=mock_llm, session_id="sess",
        extract_topics=True, auto_infer_project=False,
    )
    memory.ensure_session()

    # Should not raise
    turn = memory.add_turn("What is chunking?", "Chunking splits text into pieces.")
    assert turn.topics == []  # empty due to failure
    _ok("ConversationMemory.add_turn (topic extraction failure → non-fatal)")


# ---------------------------------------------------------------------------
# Test 9 — set_active_project persists
# ---------------------------------------------------------------------------

def test_memory_set_active_project() -> None:
    from rag.memory.store   import MemoryStore
    from rag.memory.manager import ConversationMemory

    store  = MemoryStore(db_path=":memory:")
    memory = ConversationMemory(store=store, llm=None, session_id="sess")
    memory.ensure_session()

    memory.set_active_project("Build Notion AI Knowledge System")

    # Reload to verify persistence
    memory2 = ConversationMemory(store=store, llm=None, session_id="sess")
    state = memory2.ensure_session()
    assert state.active_project == "Build Notion AI Knowledge System"
    _ok("ConversationMemory.set_active_project (persists)")


# ---------------------------------------------------------------------------
# Test 10 — build_context_block format
# ---------------------------------------------------------------------------

def test_memory_build_context_block() -> None:
    from rag.memory.store   import MemoryStore
    from rag.memory.manager import ConversationMemory

    store  = MemoryStore(db_path=":memory:")
    memory = ConversationMemory(store=store, llm=None, session_id="sess",
                                extract_topics=False)
    memory.ensure_session()

    # Empty session → empty block
    block = memory.build_context_block()
    assert block == ""

    memory.set_active_project("Notion RAG System")
    memory.add_turn("What is retrieval?", "Retrieval finds relevant chunks.")

    block = memory.build_context_block()
    assert "Active Project:" in block
    assert "Notion RAG System" in block
    assert "What is retrieval?" in block
    _ok("ConversationMemory.build_context_block")


# ---------------------------------------------------------------------------
# Test 11 — clear_session resets state
# ---------------------------------------------------------------------------

def test_memory_clear_session() -> None:
    from rag.memory.store   import MemoryStore
    from rag.memory.manager import ConversationMemory

    store  = MemoryStore(db_path=":memory:")
    memory = ConversationMemory(store=store, llm=None, session_id="sess",
                                extract_topics=False)
    memory.ensure_session()
    memory.add_turn("Q1", "A1")
    memory.set_active_project("Test Project")

    memory.clear_session()

    state = memory.get_state()
    assert state.recent_questions == []
    assert state.active_project == ""
    _ok("ConversationMemory.clear_session")


# ---------------------------------------------------------------------------
# Test 12 — max_recent rolling window
# ---------------------------------------------------------------------------

def test_memory_max_recent() -> None:
    from rag.memory.store   import MemoryStore
    from rag.memory.manager import ConversationMemory

    store  = MemoryStore(db_path=":memory:")
    memory = ConversationMemory(store=store, llm=None, session_id="sess",
                                max_recent=3, extract_topics=False)
    memory.ensure_session()

    for i in range(5):
        memory.add_turn(f"Q{i}", f"A{i}")

    state = memory.get_state()
    # In-memory list is capped at max_recent
    assert len(state.recent_questions) == 3
    assert state.recent_questions[-1].question == "Q4"
    _ok("ConversationMemory.max_recent rolling window")


# ---------------------------------------------------------------------------
# Test 13 — RAGEngine: memory injected into prompt, add_turn called
# ---------------------------------------------------------------------------

def test_engine_memory_integration() -> None:
    from rag.engine import RAGEngine

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="The answer is 42.")

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []  # no docs

    mock_config = MagicMock()
    mock_config.query_expansion_enabled = False

    mock_memory = MagicMock()
    mock_memory.build_context_block.return_value = "Active Project: Test\nCurrent Focus: Unit Testing"

    engine = RAGEngine(
        llm=mock_llm,
        get_retriever=lambda **kw: mock_retriever,
        reranker=None,
        config=mock_config,
        memory=mock_memory,
    )

    engine.answer("What is 42?")

    # Memory block was fetched
    mock_memory.build_context_block.assert_called_once()
    # add_turn was called with question and answer
    mock_memory.add_turn.assert_called_once()
    call_args = mock_memory.add_turn.call_args
    assert call_args[0][0] == "What is 42?"
    assert "42" in call_args[0][1]

    # The prompt passed to LLM contained memory context
    prompt_arg = str(mock_llm.invoke.call_args[0][0])
    assert "Active Project: Test" in prompt_arg
    _ok("RAGEngine.answer (memory injection + add_turn)")


# ---------------------------------------------------------------------------
# Test 14 — RAGEngine: memory failure is non-fatal
# ---------------------------------------------------------------------------

def test_engine_memory_failure_nonfatal() -> None:
    from rag.engine import RAGEngine

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Answer")

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []

    mock_config = MagicMock()
    mock_config.query_expansion_enabled = False

    mock_memory = MagicMock()
    mock_memory.build_context_block.side_effect = RuntimeError("memory error")
    mock_memory.add_turn.side_effect = RuntimeError("memory error")

    engine = RAGEngine(
        llm=mock_llm,
        get_retriever=lambda **kw: mock_retriever,
        reranker=None,
        config=mock_config,
        memory=mock_memory,
    )

    # Should not raise
    result = engine.answer("Any question?")
    assert result is not None
    _ok("RAGEngine.answer (memory failure → non-fatal)")


# ---------------------------------------------------------------------------
# Test 15 (live) — full round-trip
# ---------------------------------------------------------------------------

def test_live_round_trip() -> None:
    if not LIVE:
        print("Live test skipped (--no-live)")
        return

    import os, tempfile
    from rag.memory.store   import MemoryStore
    from rag.memory.manager import ConversationMemory

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    try:
        store  = MemoryStore(db_path=db_path)
        memory = ConversationMemory(store=store, llm=None, session_id="live-test",
                                    extract_topics=False)
        memory.ensure_session()

        for i in range(3):
            memory.add_turn(f"Question {i}", f"Answer for {i}.")

        memory.set_active_project("Live Test Project")

        # Reload from disk
        store2  = MemoryStore(db_path=db_path)
        memory2 = ConversationMemory(store=store2, llm=None, session_id="live-test",
                                     extract_topics=False)
        state = memory2.ensure_session()

        assert len(state.recent_questions) == 3
        assert state.active_project == "Live Test Project"

        block = memory2.build_context_block()
        assert "Live Test Project" in block
        print(f"Context block:\n{block}")

        _ok("ConversationMemory live round-trip")
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_store_session_create_read()
    test_store_turns_ordering()
    test_store_delete_session()
    test_merge_topics()
    test_memory_ensure_session()
    test_memory_add_turn_no_llm()
    test_memory_add_turn_with_topic_extraction()
    test_memory_topic_extraction_failure()
    test_memory_set_active_project()
    test_memory_build_context_block()
    test_memory_clear_session()
    test_memory_max_recent()
    test_engine_memory_integration()
    test_engine_memory_failure_nonfatal()
    test_live_round_trip()
    print("\nAll conversation memory tests passed.")
