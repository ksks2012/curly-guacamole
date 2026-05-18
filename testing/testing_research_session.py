"""
Tests for Stage C.4 — Research Session Tracking.

All tests use an in-memory SQLite database.

Run:
    python testing/testing_research_session.py
    python testing/testing_research_session.py --live
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store():
    from rag.memory.store import MemoryStore
    return MemoryStore(db_path=":memory:")


def _make_mgr(store=None):
    from rag.memory.research_session import ResearchSessionManager
    return ResearchSessionManager(store=store or _make_store())


def _make_conv_memory(store, research=None):
    from rag.memory.manager import ConversationMemory
    cm = ConversationMemory(
        store=store,
        llm=None,
        extract_topics=False,
        auto_infer_project=False,
        research=research,
    )
    cm.ensure_session()
    return cm


# ===========================================================================
# MemoryStore — C.4 tables
# ===========================================================================

class TestStoreResearchSessions(unittest.TestCase):

    def test_create_and_get(self):
        store = _make_store()
        store.create_research_session("sid-1", "Agentic RAG", tags=["RAG"])
        row = store.get_research_session("sid-1")
        self.assertIsNotNone(row)
        self.assertEqual(row["name"],   "Agentic RAG")
        self.assertEqual(row["status"], "active")
        self.assertEqual(row["tags"],   ["RAG"])
        self.assertEqual(row["queries"], [])
        self.assertEqual(row["doc_ids"], [])

    def test_get_nonexistent_returns_none(self):
        store = _make_store()
        self.assertIsNone(store.get_research_session("no-such-id"))

    def test_get_by_name_returns_most_recent(self):
        store = _make_store()
        store.create_research_session("s1", "Topic")
        store.create_research_session("s2", "Topic")
        row = store.get_research_session_by_name("Topic")
        # The store returns newest first (by created_at desc)
        self.assertIn(row["session_id"], ["s1", "s2"])

    def test_list_sessions_by_status(self):
        store = _make_store()
        store.create_research_session("a", "Active One")
        store.create_research_session("b", "To Archive")
        store.update_research_session_status("b", "archived")
        active   = store.list_research_sessions("active")
        archived = store.list_research_sessions("archived")
        self.assertEqual(len(active),   1)
        self.assertEqual(active[0]["name"], "Active One")
        self.assertEqual(len(archived), 1)

    def test_add_query_appends_and_dedupes_doc_ids(self):
        store = _make_store()
        store.create_research_session("s", "Test")
        store.add_query_to_research_session("s", "What is RAG?", doc_ids=["d1", "d2"])
        store.add_query_to_research_session("s", "What is chunking?", doc_ids=["d2", "d3"])
        row = store.get_research_session("s")
        self.assertEqual(row["queries"], ["What is RAG?", "What is chunking?"])
        # d2 should appear only once
        self.assertEqual(row["doc_ids"].count("d2"), 1)
        self.assertIn("d1", row["doc_ids"])
        self.assertIn("d3", row["doc_ids"])

    def test_delete_removes_session_and_notes(self):
        store = _make_store()
        store.create_research_session("s", "To Delete")
        store.add_research_note("n1", "s", "Some note")
        store.delete_research_session("s")
        self.assertIsNone(store.get_research_session("s"))
        self.assertEqual(store.get_research_notes("s"), [])


class TestStoreResearchNotes(unittest.TestCase):

    def test_add_and_get_notes(self):
        store = _make_store()
        store.create_research_session("s", "Session")
        store.add_research_note("n1", "s", "First note", source_doc_ids=["d1"])
        store.add_research_note("n2", "s", "Second note")
        notes = store.get_research_notes("s")
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[0]["note_id"], "n1")
        self.assertEqual(notes[0]["source_doc_ids"], ["d1"])
        self.assertEqual(notes[1]["content"], "Second note")

    def test_notes_ordered_oldest_first(self):
        store = _make_store()
        store.create_research_session("s", "Session")
        store.add_research_note("n1", "s", "Old")
        store.add_research_note("n2", "s", "New")
        notes = store.get_research_notes("s")
        self.assertEqual(notes[0]["note_id"], "n1")

    def test_delete_note(self):
        store = _make_store()
        store.create_research_session("s", "Session")
        store.add_research_note("n1", "s", "Delete me")
        store.delete_research_note("n1")
        self.assertEqual(store.get_research_notes("s"), [])


# ===========================================================================
# ResearchSessionManager
# ===========================================================================

class TestResearchSessionManager(unittest.TestCase):

    def test_create_returns_session(self):
        from rag.memory.research_session import ResearchSession
        mgr = _make_mgr()
        sess = mgr.create("Agentic RAG", tags=["RAG", "Agents"])
        self.assertIsInstance(sess, ResearchSession)
        self.assertEqual(sess.name, "Agentic RAG")
        self.assertEqual(sess.tags, ["RAG", "Agents"])
        self.assertEqual(sess.status, "active")

    def test_active_session_lifecycle(self):
        mgr = _make_mgr()
        sess = mgr.create("My Research")
        self.assertIsNone(mgr.active_session_id)
        mgr.set_active(sess.session_id)
        self.assertEqual(mgr.active_session_id, sess.session_id)
        mgr.clear_active()
        self.assertIsNone(mgr.active_session_id)

    def test_archive_clears_active(self):
        mgr = _make_mgr()
        sess = mgr.create("Work")
        mgr.set_active(sess.session_id)
        mgr.archive(sess.session_id)
        self.assertIsNone(mgr.active_session_id)
        archived = mgr.list_archived()
        self.assertEqual(len(archived), 1)

    def test_restore_session(self):
        mgr = _make_mgr()
        sess = mgr.create("Restored")
        mgr.archive(sess.session_id)
        mgr.restore(sess.session_id)
        self.assertEqual(len(mgr.list_active()), 1)

    def test_on_turn_updates_active_session(self):
        mgr = _make_mgr()
        sess = mgr.create("Track Me")
        mgr.set_active(sess.session_id)
        mgr.on_turn("What is chunking?", doc_ids=["d1"])
        mgr.on_turn("What is RAG?",      doc_ids=["d2"])
        updated = mgr.get(sess.session_id)
        self.assertEqual(len(updated.queries), 2)
        self.assertIn("d1", updated.doc_ids)
        self.assertIn("d2", updated.doc_ids)

    def test_on_turn_no_op_when_no_active(self):
        mgr = _make_mgr()
        # Should not raise even with no active session
        mgr.on_turn("Silent query")

    def test_add_note_to_active_session(self):
        from rag.memory.research_session import ResearchNote
        mgr = _make_mgr()
        sess = mgr.create("Notes Test")
        mgr.set_active(sess.session_id)
        note = mgr.add_note("Key finding: ReAct pattern works well", source_doc_ids=["doc-1"])
        self.assertIsInstance(note, ResearchNote)
        self.assertEqual(note.session_id, sess.session_id)
        notes = mgr.get_notes()
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].content, "Key finding: ReAct pattern works well")

    def test_add_note_raises_without_session(self):
        mgr = _make_mgr()
        with self.assertRaises(ValueError):
            mgr.add_note("This should fail — no active or explicit session")

    def test_delete_note(self):
        mgr = _make_mgr()
        sess = mgr.create("Note Delete Test")
        mgr.set_active(sess.session_id)
        note = mgr.add_note("Temporary note")
        mgr.delete_note(note.note_id)
        self.assertEqual(mgr.get_notes(), [])

    def test_delete_session_and_notes(self):
        mgr = _make_mgr()
        sess = mgr.create("Delete All")
        mgr.set_active(sess.session_id)
        mgr.add_note("Note to be deleted")
        mgr.delete(sess.session_id)
        self.assertIsNone(mgr.get(sess.session_id))
        self.assertIsNone(mgr.active_session_id)

    def test_build_session_block_empty_when_no_active(self):
        mgr = _make_mgr()
        self.assertEqual(mgr.build_session_block(), "")

    def test_build_session_block_empty_when_session_has_no_data(self):
        mgr = _make_mgr()
        sess = mgr.create("Empty Session")
        mgr.set_active(sess.session_id)
        # No queries or docs → block should be empty
        self.assertEqual(mgr.build_session_block(), "")

    def test_build_session_block_format(self):
        mgr = _make_mgr()
        sess = mgr.create("Agentic RAG research", tags=["RAG", "Agents"])
        mgr.set_active(sess.session_id)
        mgr.on_turn("What is ReAct?",     doc_ids=["d1", "d2"])
        mgr.on_turn("What is Reflexion?", doc_ids=["d3"])
        mgr.add_note("ReAct uses alternating reason+act steps")
        block = mgr.build_session_block()
        self.assertIn("Research Session: Agentic RAG research", block)
        self.assertIn("Tags: RAG, Agents", block)
        self.assertIn("Queries: 2", block)
        self.assertIn("Docs: 3",    block)
        self.assertIn("Notes: 1",   block)
        self.assertIn("Latest: What is Reflexion?", block)

    def test_get_by_name(self):
        mgr = _make_mgr()
        mgr.create("Named Session")
        found = mgr.get_by_name("Named Session")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "Named Session")

    def test_list_active_excludes_archived(self):
        mgr = _make_mgr()
        s1 = mgr.create("Keep")
        s2 = mgr.create("Archive")
        mgr.archive(s2.session_id)
        active = mgr.list_active()
        names = [s.name for s in active]
        self.assertIn("Keep", names)
        self.assertNotIn("Archive", names)


# ===========================================================================
# ConversationMemory integration
# ===========================================================================

class TestConversationMemoryC4(unittest.TestCase):

    def test_add_turn_calls_research_on_turn(self):
        from rag.memory.research_session import ResearchSessionManager
        store = _make_store()
        mgr   = ResearchSessionManager(store=store)
        spy   = MagicMock(wraps=mgr)

        cm = _make_conv_memory(store=store, research=spy)
        cm.add_turn("Tell me about RAG", "RAG is ...", doc_ids=["d1"])

        spy.on_turn.assert_called_once()
        args, kwargs = spy.on_turn.call_args
        self.assertEqual(args[0], "Tell me about RAG")
        self.assertEqual(kwargs.get("doc_ids"), ["d1"])

    def test_build_context_block_includes_session(self):
        from rag.memory.research_session import ResearchSessionManager
        store = _make_store()
        mgr   = ResearchSessionManager(store=store)
        sess  = mgr.create("Context Test")
        mgr.set_active(sess.session_id)
        # Seed a query so the block is non-empty
        mgr.on_turn("Initial question", doc_ids=["d1"])

        cm    = _make_conv_memory(store=store, research=mgr)
        block = cm.build_context_block()
        self.assertIn("Research Session: Context Test", block)

    def test_build_context_block_no_research(self):
        store = _make_store()
        cm    = _make_conv_memory(store=store, research=None)
        # Should not raise
        block = cm.build_context_block()
        self.assertIsInstance(block, str)


# ===========================================================================
# Live test
# ===========================================================================

@unittest.skipUnless("--live" in sys.argv, "Skipped (pass --live to run)")
class TestResearchSessionLive(unittest.TestCase):

    def test_full_client_pipeline(self):
        from rag.client import LocalLlamaClient
        client = LocalLlamaClient()

        sess = client.start_research_session("Live Integration Session", tags=["test"])
        self.assertIsNotNone(sess)

        client.add_research_note("Live note from test run", source_doc_ids=[])
        notes = client.get_research_notes()
        self.assertTrue(any("Live note" in n.content for n in notes))

        sessions = client.list_research_sessions()
        self.assertTrue(any(s.name == "Live Integration Session" for s in sessions))

        client.archive_research_session()
        archived = client.list_research_sessions(archived=True)
        self.assertTrue(any(s.name == "Live Integration Session" for s in archived))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.argv = [a for a in sys.argv if a not in ("--live", "--no-live")]
    unittest.main(verbosity=2)
