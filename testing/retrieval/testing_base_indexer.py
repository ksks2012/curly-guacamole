"""
Smoke tests for Phase 2 Step 2.1 — BaseIndexer unified lifecycle.

Tests:
  - IndexStats arithmetic (+, from_dict, aggregate)
  - BaseIndexer is an ABC (cannot instantiate directly)
  - DocumentIndexer satisfies BaseIndexer protocol (isinstance check)
  - CodeIndexer satisfies BaseIndexer (isinstance check)
  - DocumentIndexer.ingest delegates to DocumentIngester + Indexer.run
  - DocumentIndexer.update aliases ingest
  - DocumentIndexer.delete removes by doc_id
  - DocumentIndexer.reindex = delete + ingest
  - CodeIndexer.ingest delegates to index_all
  - CodeIndexer.update aliases ingest
  - CodeIndexer.delete delegates to delete_repo
  - CodeIndexer.reindex = delete_repo + index_all
  - IndexStats.from_dict handles both LangChain (num_added) and raw (added) keys
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = []
FAIL = []

def ok(name: str) -> None:
    print(f"{name}: OK")
    PASS.append(name)

def fail(name: str, exc: Exception) -> None:
    print(f"{name}: FAIL — {exc}")
    FAIL.append(name)

# ---------------------------------------------------------------------------
# IndexStats tests
# ---------------------------------------------------------------------------

def test_index_stats_defaults():
    from rag.indexer import IndexStats
    s = IndexStats()
    assert s.added == 0 and s.updated == 0 and s.skipped == 0 and s.deleted == 0
    ok("test_index_stats_defaults")

def test_index_stats_add():
    from rag.indexer import IndexStats
    a = IndexStats(added=3, updated=1)
    b = IndexStats(skipped=2, deleted=4)
    c = a + b
    assert c.added == 3 and c.updated == 1 and c.skipped == 2 and c.deleted == 4
    ok("test_index_stats_add")

def test_index_stats_from_dict_langchain():
    from rag.indexer import IndexStats
    d = {"num_added": 5, "num_updated": 2, "num_skipped": 1, "num_deleted": 0}
    s = IndexStats.from_dict(d)
    assert s.added == 5 and s.updated == 2 and s.skipped == 1 and s.deleted == 0
    ok("test_index_stats_from_dict_langchain")

def test_index_stats_from_dict_raw():
    from rag.indexer import IndexStats
    d = {"added": 3, "updated": 0, "skipped": 7, "deleted": 1}
    s = IndexStats.from_dict(d)
    assert s.added == 3 and s.skipped == 7
    ok("test_index_stats_from_dict_raw")

def test_index_stats_aggregate():
    from rag.indexer import IndexStats
    stats = [IndexStats(added=1), IndexStats(updated=2), IndexStats(deleted=3)]
    total = IndexStats.aggregate(stats)
    assert total.added == 1 and total.updated == 2 and total.deleted == 3
    ok("test_index_stats_aggregate")

# ---------------------------------------------------------------------------
# BaseIndexer ABC tests
# ---------------------------------------------------------------------------

def test_base_indexer_is_abstract():
    from rag.indexer import BaseIndexer
    try:
        BaseIndexer()  # type: ignore[abstract]
        assert False, "Should have raised TypeError"
    except TypeError:
        ok("test_base_indexer_is_abstract")

def test_document_indexer_is_base_indexer():
    from rag.indexer import BaseIndexer
    from rag.ingest.document_indexer import DocumentIndexer
    assert issubclass(DocumentIndexer, BaseIndexer)
    ok("test_document_indexer_is_base_indexer")

def test_code_indexer_is_base_indexer():
    from rag.indexer import BaseIndexer
    from rag.code.indexer import CodeIndexer
    assert issubclass(CodeIndexer, BaseIndexer)
    ok("test_code_indexer_is_base_indexer")

# ---------------------------------------------------------------------------
# DocumentIndexer tests (all IO mocked)
# ---------------------------------------------------------------------------

def _make_doc_indexer():
    """Build a DocumentIndexer with fully mocked internals."""
    from rag.ingest.document_indexer import DocumentIndexer
    idx = DocumentIndexer.__new__(DocumentIndexer)
    idx._ingester = MagicMock()
    idx._indexer  = MagicMock()
    idx._indexer.db = MagicMock()
    return idx

def test_document_indexer_ingest_calls_ingester_and_indexer():
    idx = _make_doc_indexer()
    fake_docs = [MagicMock()]
    idx._ingester.ingest.return_value = fake_docs
    idx._indexer.run.return_value = {"num_added": 2, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

    stats = idx.ingest("docs/foo.md", doc_id="foo")

    idx._ingester.ingest.assert_called_once_with(path="docs/foo.md", doc_id="foo")
    idx._indexer.run.assert_called_once_with(fake_docs)
    assert stats.added == 2
    ok("test_document_indexer_ingest_calls_ingester_and_indexer")

def test_document_indexer_ingest_defaults_doc_id_to_stem():
    idx = _make_doc_indexer()
    idx._ingester.ingest.return_value = []
    idx._indexer.run.return_value = {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

    idx.ingest("/some/path/readme.md")

    call_kwargs = idx._ingester.ingest.call_args
    assert call_kwargs.kwargs["doc_id"] == "readme"
    ok("test_document_indexer_ingest_defaults_doc_id_to_stem")

def test_document_indexer_update_aliases_ingest():
    idx = _make_doc_indexer()
    idx._ingester.ingest.return_value = []
    idx._indexer.run.return_value = {"num_added": 1, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

    stats = idx.update("docs/foo.md", doc_id="foo")
    assert stats.added == 1
    idx._ingester.ingest.assert_called_once()
    ok("test_document_indexer_update_aliases_ingest")

def test_document_indexer_delete_removes_by_doc_id():
    idx = _make_doc_indexer()
    idx._indexer.db.get.return_value = {"ids": ["id1", "id2"]}

    stats = idx.delete("foo")

    idx._indexer.db.get.assert_called_once_with(where={"doc_id": "foo"}, include=[])
    idx._indexer.db.delete.assert_called_once_with(["id1", "id2"])
    assert stats.deleted == 2
    ok("test_document_indexer_delete_removes_by_doc_id")

def test_document_indexer_delete_no_docs_is_noop():
    idx = _make_doc_indexer()
    idx._indexer.db.get.return_value = {"ids": []}

    stats = idx.delete("nonexistent")
    idx._indexer.db.delete.assert_not_called()
    assert stats.deleted == 0
    ok("test_document_indexer_delete_no_docs_is_noop")

def test_document_indexer_reindex_delete_then_ingest():
    idx = _make_doc_indexer()
    idx._indexer.db.get.return_value = {"ids": ["old1"]}
    idx._ingester.ingest.return_value = [MagicMock()]
    idx._indexer.run.return_value = {"num_added": 3, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}

    stats = idx.reindex("docs/foo.md", doc_id="foo")
    assert stats.deleted == 1   # from delete step
    assert stats.added   == 3   # from ingest step
    ok("test_document_indexer_reindex_delete_then_ingest")

# ---------------------------------------------------------------------------
# CodeIndexer tests (all IO mocked)
# ---------------------------------------------------------------------------

def _make_code_indexer():
    """Build a CodeIndexer with mocked internals."""
    from rag.code.indexer import CodeIndexer
    idx = CodeIndexer.__new__(CodeIndexer)
    idx._persist_dir = "/fake"
    idx._embed = MagicMock()
    idx._prefix = "code"
    idx._dbs = {}
    return idx

def test_code_indexer_ingest_delegates_to_index_all():
    idx = _make_code_indexer()
    manifest = MagicMock(); manifest.repo_id = "myrepo"
    chunks = [MagicMock()]
    idx.index_all = MagicMock(return_value={
        "repo":   {"added": 1, "updated": 0, "skipped": 0, "deleted": 0},
        "file":   {"added": 2, "updated": 0, "skipped": 0, "deleted": 0},
        "symbol": {"added": 5, "updated": 1, "skipped": 0, "deleted": 0},
        "block":  {"added": 3, "updated": 0, "skipped": 0, "deleted": 0},
    })

    stats = idx.ingest((manifest, chunks))
    idx.index_all.assert_called_once_with(manifest, chunks, store=None)
    assert stats.added == 11 and stats.updated == 1
    ok("test_code_indexer_ingest_delegates_to_index_all")

def test_code_indexer_update_aliases_ingest():
    idx = _make_code_indexer()
    manifest = MagicMock(); manifest.repo_id = "myrepo"
    chunks = []
    idx.index_all = MagicMock(return_value={
        "repo": {"added": 0, "updated": 1, "skipped": 0, "deleted": 0},
        "file": {"added": 0, "updated": 0, "skipped": 0, "deleted": 0},
        "symbol": {"added": 0, "updated": 0, "skipped": 0, "deleted": 0},
        "block": {"added": 0, "updated": 0, "skipped": 0, "deleted": 0},
    })

    stats = idx.update((manifest, chunks))
    idx.index_all.assert_called_once()
    assert stats.updated == 1
    ok("test_code_indexer_update_aliases_ingest")

def test_code_indexer_delete_delegates_to_delete_repo():
    idx = _make_code_indexer()
    from rag.indexer import IndexStats
    idx.delete_repo = MagicMock(return_value=IndexStats(deleted=7))

    stats = idx.delete("myrepo")
    idx.delete_repo.assert_called_once_with("myrepo")
    assert stats.deleted == 7
    ok("test_code_indexer_delete_delegates_to_delete_repo")

def test_code_indexer_reindex_delete_then_ingest():
    idx = _make_code_indexer()
    from rag.indexer import IndexStats
    manifest = MagicMock(); manifest.repo_id = "myrepo"
    chunks = []
    idx.delete_repo = MagicMock(return_value=IndexStats(deleted=10))
    idx.index_all = MagicMock(return_value={
        "repo":   {"added": 1, "updated": 0, "skipped": 0, "deleted": 0},
        "file":   {"added": 2, "updated": 0, "skipped": 0, "deleted": 0},
        "symbol": {"added": 4, "updated": 0, "skipped": 0, "deleted": 0},
        "block":  {"added": 3, "updated": 0, "skipped": 0, "deleted": 0},
    })

    stats = idx.reindex((manifest, chunks))
    idx.delete_repo.assert_called_once_with("myrepo")
    idx.index_all.assert_called_once_with(manifest, chunks, store=None)
    assert stats.deleted == 10 and stats.added == 10
    ok("test_code_indexer_reindex_delete_then_ingest")

def test_delete_repo_returns_index_stats():
    """delete_repo() now returns IndexStats instead of None."""
    idx = _make_code_indexer()
    from rag.indexer import IndexStats
    # Mock _db to return a fake Chroma db
    fake_db = MagicMock()
    fake_db.get.return_value = {"ids": ["a", "b"]}
    idx._db = MagicMock(return_value=fake_db)

    result = idx.delete_repo("myrepo")
    assert isinstance(result, IndexStats)
    # 4 levels × 2 ids each = 8 total
    assert result.deleted == 8
    ok("test_delete_repo_returns_index_stats")

# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_index_stats_defaults,
        test_index_stats_add,
        test_index_stats_from_dict_langchain,
        test_index_stats_from_dict_raw,
        test_index_stats_aggregate,
        test_base_indexer_is_abstract,
        test_document_indexer_is_base_indexer,
        test_code_indexer_is_base_indexer,
        test_document_indexer_ingest_calls_ingester_and_indexer,
        test_document_indexer_ingest_defaults_doc_id_to_stem,
        test_document_indexer_update_aliases_ingest,
        test_document_indexer_delete_removes_by_doc_id,
        test_document_indexer_delete_no_docs_is_noop,
        test_document_indexer_reindex_delete_then_ingest,
        test_code_indexer_ingest_delegates_to_index_all,
        test_code_indexer_update_aliases_ingest,
        test_code_indexer_delete_delegates_to_delete_repo,
        test_code_indexer_reindex_delete_then_ingest,
        test_delete_repo_returns_index_stats,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            fail(t.__name__, e)

    print()
    if FAIL:
        print(f"FAIL  ({len(FAIL)} failed, {len(PASS)} passed)")
        sys.exit(1)
    else:
        print(f"PASS")
