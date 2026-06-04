"""
Tests for Stage C.2 (Semantic User Memory) and Stage C.3 (Knowledge Timeline).

All tests use an in-memory SQLite database — no file system side effects.

Run:
    python testing/testing_user_memory.py          # mock tests only
    python testing/testing_user_memory.py --live   # includes live-LLM tests
    python testing/testing_user_memory.py --no-live  # explicit mock-only mode
"""

from __future__ import annotations

import pytest
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store():
    """Return a MemoryStore backed by a fresh in-memory SQLite database."""
    from rag.memory.store import MemoryStore
    return MemoryStore(db_path=":memory:")


def _make_conv_memory(store=None, user_memory=None, timeline=None):
    """Return a ConversationMemory with no LLM wired (mock-safe)."""
    from rag.memory.manager import ConversationMemory
    s = store or _make_store()
    cm = ConversationMemory(
        store=s,
        llm=None,                 # no LLM → topic extraction disabled
        extract_topics=False,
        auto_infer_project=False,
        user_memory=user_memory,
        timeline=timeline,
    )
    cm.ensure_session()
    return cm


# ===========================================================================
# MemoryStore — C.2 user_interests table
# ===========================================================================

class TestStoreUserInterests(unittest.TestCase):

    def test_upsert_creates_new_row(self):
        store = _make_store()
        store.upsert_interest("RAG Architecture")
        rows = store.get_top_interests(10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["topic"], "RAG Architecture")
        self.assertEqual(rows[0]["count"], 1)
        self.assertAlmostEqual(rows[0]["weight"], 1.0, places=3)

    def test_upsert_ema_update(self):
        store = _make_store()
        store.upsert_interest("Vector Search")   # weight = 1.0
        store.upsert_interest("Vector Search")   # weight = 1.0*0.9 + 1.0 = 1.9
        rows = store.get_top_interests(1)
        self.assertEqual(rows[0]["count"], 2)
        self.assertAlmostEqual(rows[0]["weight"], 1.9, places=2)

    def test_upsert_ema_three_times(self):
        store = _make_store()
        for _ in range(3):
            store.upsert_interest("Memory")
        # 1st: 1.0 → 2nd: 1.9 → 3rd: 1.9*0.9 + 1.0 = 2.71
        rows = store.get_top_interests(1)
        self.assertAlmostEqual(rows[0]["weight"], 2.71, places=1)

    def test_get_top_interests_sorted_by_weight(self):
        store = _make_store()
        store.upsert_interest("A")   # weight 1.0
        store.upsert_interest("B")
        store.upsert_interest("B")   # weight 1.9
        store.upsert_interest("C")
        store.upsert_interest("C")
        store.upsert_interest("C")  # weight 2.71

        rows = store.get_top_interests(3)
        self.assertEqual([r["topic"] for r in rows], ["C", "B", "A"])

    def test_get_top_interests_limit(self):
        store = _make_store()
        for topic in ["A", "B", "C", "D", "E"]:
            store.upsert_interest(topic)
        self.assertEqual(len(store.get_top_interests(3)), 3)

    def test_clear_interests(self):
        store = _make_store()
        store.upsert_interest("Cleanup")
        store.clear_interests()
        self.assertEqual(store.get_top_interests(), [])


# ===========================================================================
# MemoryStore — C.3 timeline_entries table
# ===========================================================================

class TestStoreTimeline(unittest.TestCase):

    def test_upsert_creates_entry(self):
        store = _make_store()
        store.upsert_timeline_entry("2026-05-01", ["RAG", "Memory"], ["doc1"])
        rows = store.get_timeline("2026-05-01", "2026-05-01")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["date"], "2026-05-01")
        self.assertEqual(rows[0]["topics"], ["RAG", "Memory"])
        self.assertEqual(rows[0]["doc_ids"], ["doc1"])
        self.assertEqual(rows[0]["question_count"], 1)

    def test_upsert_merges_same_day(self):
        store = _make_store()
        store.upsert_timeline_entry("2026-05-01", ["RAG"],    ["doc1"])
        store.upsert_timeline_entry("2026-05-01", ["Memory"], ["doc1", "doc2"])

        rows = store.get_timeline("2026-05-01", "2026-05-01")
        self.assertEqual(len(rows), 1)
        self.assertIn("RAG", rows[0]["topics"])
        self.assertIn("Memory", rows[0]["topics"])
        # doc1 should appear only once (deduped)
        self.assertEqual(rows[0]["doc_ids"].count("doc1"), 1)
        self.assertIn("doc2", rows[0]["doc_ids"])
        self.assertEqual(rows[0]["question_count"], 2)

    def test_upsert_preserves_topic_order(self):
        store = _make_store()
        store.upsert_timeline_entry("2026-06-01", ["A", "B"], [])
        store.upsert_timeline_entry("2026-06-01", ["C", "A"], [])   # A duplicate
        rows = store.get_timeline("2026-06-01", "2026-06-01")
        # A should appear once, preserved in insertion order: A, B, C
        self.assertEqual(rows[0]["topics"], ["A", "B", "C"])

    def test_get_timeline_date_range(self):
        store = _make_store()
        for day in ["2026-04-30", "2026-05-01", "2026-05-15", "2026-05-31", "2026-06-01"]:
            store.upsert_timeline_entry(day, ["T"], [])

        rows = store.get_timeline("2026-05-01", "2026-05-31")
        dates = [r["date"] for r in rows]
        self.assertIn("2026-05-01", dates)
        self.assertIn("2026-05-15", dates)
        self.assertIn("2026-05-31", dates)
        self.assertNotIn("2026-04-30", dates)
        self.assertNotIn("2026-06-01", dates)

    def test_get_timeline_ordered_ascending(self):
        store = _make_store()
        for day in ["2026-05-03", "2026-05-01", "2026-05-02"]:
            store.upsert_timeline_entry(day, ["T"], [])
        rows = store.get_timeline("2026-05-01", "2026-05-03")
        self.assertEqual([r["date"] for r in rows], ["2026-05-01", "2026-05-02", "2026-05-03"])

    def test_get_timeline_by_year(self):
        store = _make_store()
        store.upsert_timeline_entry("2025-12-31", ["Old"], [])
        store.upsert_timeline_entry("2026-01-01", ["New"], [])
        rows = store.get_timeline_by_year(2026)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["date"], "2026-01-01")


# ===========================================================================
# UserMemoryManager (C.2)
# ===========================================================================

class TestUserMemoryManager(unittest.TestCase):

    def test_update_from_topics(self):
        from rag.memory.user_memory import UserMemoryManager
        store = _make_store()
        um = UserMemoryManager(store)
        um.update_from_topics(["RAG", "Vector Search", "RAG"])   # RAG twice
        top = um.get_top_interests(10)
        topics = [r["topic"] for r in top]
        self.assertIn("RAG", topics)
        self.assertIn("Vector Search", topics)

    def test_update_skips_blank_topics(self):
        from rag.memory.user_memory import UserMemoryManager
        store = _make_store()
        um = UserMemoryManager(store)
        um.update_from_topics(["  ", "", "Valid"])
        top = um.get_top_interests()
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0]["topic"], "Valid")

    def test_build_profile_block_format(self):
        from rag.memory.user_memory import UserMemoryManager
        store = _make_store()
        um = UserMemoryManager(store)
        um.update_from_topics(["RAG", "Memory", "Embeddings"])
        block = um.build_profile_block(n=3)
        self.assertTrue(block.startswith("Frequent Research Areas:"))
        self.assertIn("RAG", block)

    def test_build_profile_block_empty(self):
        from rag.memory.user_memory import UserMemoryManager
        store = _make_store()
        um = UserMemoryManager(store)
        self.assertEqual(um.build_profile_block(), "")

    def test_get_profile(self):
        from rag.memory.user_memory import UserMemoryManager, UserProfile
        store = _make_store()
        um = UserMemoryManager(store)
        um.update_from_topics(["A", "B"])
        profile = um.get_profile()
        self.assertIsInstance(profile, UserProfile)
        self.assertEqual(len(profile.top_interests), 2)

    def test_reset_clears_all(self):
        from rag.memory.user_memory import UserMemoryManager
        store = _make_store()
        um = UserMemoryManager(store)
        um.update_from_topics(["A", "B"])
        um.reset()
        self.assertEqual(um.get_top_interests(), [])


# ===========================================================================
# KnowledgeTimeline (C.3)
# ===========================================================================

class TestKnowledgeTimeline(unittest.TestCase):

    def test_record_activity_today(self):
        from rag.memory.timeline import KnowledgeTimeline
        store = _make_store()
        tl = KnowledgeTimeline(store)
        tl.record_activity(["RAG"], doc_ids=["doc1"])
        entries = tl.get_recent(days=1)
        self.assertEqual(len(entries), 1)
        self.assertIn("RAG", entries[0]["topics"])

    def test_record_activity_specific_date(self):
        from rag.memory.timeline import KnowledgeTimeline
        store = _make_store()
        tl = KnowledgeTimeline(store)
        tl.record_activity(["A", "B"], date_str="2026-03-15")
        entries = tl.get_period(2026, 3)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["date"], "2026-03-15")

    def test_record_activity_merges_same_day(self):
        from rag.memory.timeline import KnowledgeTimeline
        store = _make_store()
        tl = KnowledgeTimeline(store)
        tl.record_activity(["A"], date_str="2026-05-01")
        tl.record_activity(["B"], date_str="2026-05-01")
        entries = tl.get_period(2026, 5)
        self.assertEqual(entries[0]["question_count"], 2)
        self.assertIn("A", entries[0]["topics"])
        self.assertIn("B", entries[0]["topics"])

    def test_record_activity_skips_all_blank(self):
        from rag.memory.timeline import KnowledgeTimeline
        store = _make_store()
        tl = KnowledgeTimeline(store)
        tl.record_activity(["  ", ""], doc_ids=[""], date_str="2026-05-01")
        entries = tl.get_period(2026, 5)
        self.assertEqual(len(entries), 0)

    def test_build_timeline_block_format(self):
        from rag.memory.timeline import KnowledgeTimeline
        store = _make_store()
        tl = KnowledgeTimeline(store)
        tl.record_activity(["RAG", "Chunking"], date_str="2026-05-17")
        tl.record_activity(["Memory"],          date_str="2026-05-16")
        block = tl.build_timeline_block(days=30)
        self.assertTrue(block.startswith("Recent Activity:"))
        self.assertIn("2026-05-17", block)
        self.assertIn("RAG", block)
        self.assertIn("question", block)

    def test_build_timeline_block_empty(self):
        from rag.memory.timeline import KnowledgeTimeline
        store = _make_store()
        tl = KnowledgeTimeline(store)
        self.assertEqual(tl.build_timeline_block(days=7), "")

    def test_get_yearly_summary_counts(self):
        from rag.memory.timeline import KnowledgeTimeline
        store = _make_store()
        tl = KnowledgeTimeline(store)
        # Same topic on 3 different days → appears in 3 entries
        tl.record_activity(["RAG", "Memory"], date_str="2026-01-01")
        tl.record_activity(["RAG"],           date_str="2026-01-02")
        tl.record_activity(["Memory"],        date_str="2026-01-03")
        summary = tl.get_yearly_summary(2026)
        self.assertEqual(summary["RAG"], 2)      # 2 days
        self.assertEqual(summary["Memory"], 2)   # 2 days

    def test_get_yearly_summary_sorted(self):
        from rag.memory.timeline import KnowledgeTimeline
        store = _make_store()
        tl = KnowledgeTimeline(store)
        tl.record_activity(["B", "A"], date_str="2026-02-01")
        tl.record_activity(["B"],      date_str="2026-02-02")
        summary = tl.get_yearly_summary(2026)
        topics = list(summary.keys())
        self.assertEqual(topics[0], "B")   # B on 2 days → highest

    def test_get_period_month_filter(self):
        from rag.memory.timeline import KnowledgeTimeline
        store = _make_store()
        tl = KnowledgeTimeline(store)
        tl.record_activity(["A"], date_str="2026-04-30")
        tl.record_activity(["B"], date_str="2026-05-01")
        rows = tl.get_period(2026, month=4)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["date"], "2026-04-30")


# ===========================================================================
# ConversationMemory integration with C.2 + C.3
# ===========================================================================

class TestConversationMemoryC2C3(unittest.TestCase):

    def test_add_turn_calls_user_memory(self):
        from rag.memory.user_memory import UserMemoryManager
        um = UserMemoryManager(store=_make_store())
        um_spy = MagicMock(wraps=um)

        cm = _make_conv_memory(user_memory=um_spy)
        cm.add_turn("What is RAG?", "RAG is ...")

        # With extract_topics=False and no LLM, topics list is [] — but the
        # call should still be made (with an empty list — harmless).
        um_spy.update_from_topics.assert_called_once()

    def test_add_turn_calls_timeline(self):
        from rag.memory.timeline import KnowledgeTimeline
        tl = KnowledgeTimeline(store=_make_store())
        tl_spy = MagicMock(wraps=tl)

        cm = _make_conv_memory(timeline=tl_spy)
        cm.add_turn("What is vector search?", "Vector search is ...")

        tl_spy.record_activity.assert_called_once()

    def test_add_turn_passes_doc_ids_to_timeline(self):
        from rag.memory.timeline import KnowledgeTimeline
        tl = KnowledgeTimeline(store=_make_store())
        tl_spy = MagicMock(wraps=tl)

        cm = _make_conv_memory(timeline=tl_spy)
        cm.add_turn("Q", "A", doc_ids=["d1", "d2"])

        call_kwargs = tl_spy.record_activity.call_args
        # doc_ids passed as kwarg
        self.assertEqual(call_kwargs.kwargs.get("doc_ids"), ["d1", "d2"])

    def test_build_context_block_includes_profile(self):
        from rag.memory.user_memory import UserMemoryManager
        store = _make_store()
        um = UserMemoryManager(store=store)
        um.update_from_topics(["RAG", "Memory"])

        cm = _make_conv_memory(store=store, user_memory=um)
        block = cm.build_context_block()
        self.assertIn("Frequent Research Areas", block)

    def test_build_context_block_includes_timeline(self):
        from rag.memory.timeline import KnowledgeTimeline
        store = _make_store()
        tl = KnowledgeTimeline(store=store)
        tl.record_activity(["RAG"], date_str="2026-05-17")

        # Patch 'today' inside timeline so our fixed date is within the window
        from datetime import date
        with patch("rag.memory.timeline.date") as mock_date:
            mock_date.today.return_value = date(2026, 5, 17)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)

            cm = _make_conv_memory(store=store, timeline=tl)
            block = cm.build_context_block()

        self.assertIn("Recent Activity", block)

    def test_build_context_block_no_user_memory(self):
        """When user_memory and timeline are None, block still works fine."""
        cm = _make_conv_memory()
        block = cm.build_context_block()   # should not raise
        self.assertIsInstance(block, str)


# ===========================================================================
# RAGEngine — doc_ids propagation
# ===========================================================================

class TestRAGEngineDocIds(unittest.TestCase):

    def test_engine_passes_doc_ids_to_memory(self):
        """Engine should collect doc_ids from retrieved chunks and pass to add_turn."""
        from rag.engine import RAGEngine

        mock_llm      = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test answer")
        mock_config   = MagicMock()
        mock_config.query_expansion_enabled = False

        # Set up two docs with doc_id metadata
        from rag.retrieval.base import RetrievalResult
        result1 = RetrievalResult(content="chunk1", score=0.9, source="document",
                                   metadata={"doc_id": "doc-abc", "chunk_id": "c1", "title": "t1"})
        result2 = RetrievalResult(content="chunk2", score=0.8, source="document",
                                   metadata={"doc_id": "doc-xyz", "chunk_id": "c2", "title": "t2"})

        mock_retriever = MagicMock()
        mock_retriever.search = MagicMock(return_value=[result1, result2])

        engine = RAGEngine(
            llm=mock_llm,
            retriever=mock_retriever,
            reranker=None,
            config=mock_config,
        )

        mock_mem = MagicMock()
        mock_mem.build_context_block.return_value = ""
        engine.memory = mock_mem

        engine.answer("What is RAG?")

        add_turn_call = mock_mem.add_turn.call_args
        passed_doc_ids = add_turn_call.kwargs.get("doc_ids", [])
        self.assertIn("doc-abc", passed_doc_ids)
        self.assertIn("doc-xyz", passed_doc_ids)


# ===========================================================================
# Live test (skipped by default)
# ===========================================================================

@pytest.mark.integration
class TestUserMemoryLive(unittest.TestCase):

    def test_full_pipeline_live(self):
        """Smoke-test: client wires user_memory + timeline and they are updated."""
        from rag.client import LocalLlamaClient
        client = LocalLlamaClient()

        topics = ["Live Integration Test", "RAG Pipeline", "Python"]
        client.user_memory.update_from_topics(topics)
        top = client.user_memory.get_top_interests(10)
        topic_names = [r["topic"] for r in top]
        for t in topics:
            self.assertIn(t, topic_names)

        profile = client.user_memory.get_profile()
        self.assertGreaterEqual(profile.total_topics_seen, 3)

        client.timeline.record_activity(["Live Test Topic"], doc_ids=["live-doc"])
        recent = client.timeline.get_recent(days=1)
        self.assertTrue(any("Live Test Topic" in e["topics"] for e in recent))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


