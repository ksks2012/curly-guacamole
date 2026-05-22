"""
Tests for Phase 2 Step 2.4 — Shared Incremental Indexing.

Covers:
  - diff_by_content_hash: all four change categories
  - ChangeSet field types and defaults
  - diff_by_content_hash: empty inputs
  - diff_by_content_hash: identical hashes → all skipped
  - diff_by_content_hash: all new → all added
  - diff_by_content_hash: nothing incoming → all deleted
  - diff_by_content_hash: mixed case
  - CodeIndexer._upsert() calls diff_by_content_hash (integration via mock)
  - _upsert() skipped count propagates correctly
  - _upsert() prune_missing=False suppresses deletes
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# ChangeSet dataclass
# ---------------------------------------------------------------------------

def test_changeset_fields():
    from rag.indexer import ChangeSet
    field_names = {f.name for f in fields(ChangeSet)}
    assert field_names == {"added", "modified", "skipped", "deleted"}


def test_changeset_defaults_empty_lists():
    from rag.indexer import ChangeSet
    cs = ChangeSet()
    assert cs.added == []
    assert cs.modified == []
    assert cs.skipped == []
    assert cs.deleted == []


# ---------------------------------------------------------------------------
# diff_by_content_hash — boundary cases
# ---------------------------------------------------------------------------

def test_diff_both_empty():
    from rag.indexer import diff_by_content_hash
    cs = diff_by_content_hash({}, {})
    assert cs.added == [] and cs.modified == [] and cs.skipped == [] and cs.deleted == []


def test_diff_all_new():
    from rag.indexer import diff_by_content_hash
    incoming = {"a": "h1", "b": "h2", "c": "h3"}
    cs = diff_by_content_hash({}, incoming)
    assert set(cs.added) == {"a", "b", "c"}
    assert cs.modified == [] and cs.skipped == [] and cs.deleted == []


def test_diff_all_deleted():
    from rag.indexer import diff_by_content_hash
    existing = {"a": "h1", "b": "h2"}
    cs = diff_by_content_hash(existing, {})
    assert set(cs.deleted) == {"a", "b"}
    assert cs.added == [] and cs.modified == [] and cs.skipped == []


def test_diff_all_skipped():
    from rag.indexer import diff_by_content_hash
    hashes = {"a": "h1", "b": "h2", "c": "h3"}
    cs = diff_by_content_hash(hashes, dict(hashes))
    assert set(cs.skipped) == {"a", "b", "c"}
    assert cs.added == [] and cs.modified == [] and cs.deleted == []


def test_diff_all_modified():
    from rag.indexer import diff_by_content_hash
    existing = {"a": "old1", "b": "old2"}
    incoming = {"a": "new1", "b": "new2"}
    cs = diff_by_content_hash(existing, incoming)
    assert set(cs.modified) == {"a", "b"}
    assert cs.added == [] and cs.skipped == [] and cs.deleted == []


def test_diff_mixed():
    from rag.indexer import diff_by_content_hash
    existing = {
        "keep":    "same",
        "change":  "old",
        "gone":    "h",
    }
    incoming = {
        "keep":    "same",    # skipped
        "change":  "new",     # modified
        "fresh":   "h2",      # added
        # "gone" absent → deleted
    }
    cs = diff_by_content_hash(existing, incoming)
    assert "keep"   in cs.skipped
    assert "change" in cs.modified
    assert "fresh"  in cs.added
    assert "gone"   in cs.deleted
    # mutually exclusive
    all_ids = cs.added + cs.modified + cs.skipped + cs.deleted
    assert len(all_ids) == len(set(all_ids)), "IDs should not appear in multiple categories"


def test_diff_empty_hash_treated_as_distinct():
    """Empty string hash should NOT match a non-empty hash."""
    from rag.indexer import diff_by_content_hash
    cs = diff_by_content_hash({"a": ""}, {"a": "h1"})
    assert "a" in cs.modified


def test_diff_existing_not_mutated():
    """diff_by_content_hash must not modify its inputs."""
    from rag.indexer import diff_by_content_hash
    existing = {"a": "h1"}
    incoming = {"b": "h2"}
    existing_copy = dict(existing)
    incoming_copy = dict(incoming)
    diff_by_content_hash(existing, incoming)
    assert existing == existing_copy
    assert incoming == incoming_copy


# ---------------------------------------------------------------------------
# CodeIndexer._upsert() integration (store calls mocked)
# ---------------------------------------------------------------------------

def _make_indexer():
    from rag.code.indexer import CodeIndexer
    idx = CodeIndexer.__new__(CodeIndexer)
    idx._prefix = "code"
    idx._collection_names = {**CodeIndexer._DEFAULT_COLLECTION_NAMES}
    idx._dbs = {}
    idx._persist_dir = "/fake"
    idx._embed = MagicMock()
    return idx


def _fake_doc(doc_id: str, hash_: str):
    from langchain_core.documents import Document
    return Document(page_content="x", metadata={"content_hash": hash_})


def test_upsert_add_calls_add_documents():
    idx = _make_indexer()
    fake_db = MagicMock()
    fake_db.get.return_value = {"ids": [], "metadatas": []}
    idx._dbs["file"] = fake_db

    docs = [_fake_doc("id1", "h1"), _fake_doc("id2", "h2")]
    ids  = ["id1", "id2"]
    result = idx._upsert("file", docs, ids)

    fake_db.add_documents.assert_called_once()
    assert result["added"] == 2
    assert result["updated"] == 0 and result["skipped"] == 0 and result["deleted"] == 0


def test_upsert_skip_no_db_write():
    idx = _make_indexer()
    fake_db = MagicMock()
    fake_db.get.return_value = {
        "ids": ["id1"],
        "metadatas": [{"content_hash": "same"}],
    }
    idx._dbs["file"] = fake_db

    docs = [_fake_doc("id1", "same")]
    result = idx._upsert("file", docs, ["id1"])

    fake_db.add_documents.assert_not_called()
    fake_db.update_documents.assert_not_called()
    assert result["skipped"] == 1


def test_upsert_modified_calls_update_documents():
    idx = _make_indexer()
    fake_db = MagicMock()
    fake_db.get.return_value = {
        "ids": ["id1"],
        "metadatas": [{"content_hash": "old"}],
    }
    idx._dbs["symbol"] = fake_db

    docs = [_fake_doc("id1", "new")]
    result = idx._upsert("symbol", docs, ["id1"])

    fake_db.update_documents.assert_called_once()
    assert result["updated"] == 1


def test_upsert_prune_deletes_stale():
    idx = _make_indexer()
    fake_db = MagicMock()
    fake_db.get.return_value = {
        "ids": ["old_id"],
        "metadatas": [{"content_hash": "h"}],
    }
    idx._dbs["file"] = fake_db

    # incoming has no "old_id" → should be pruned
    docs = [_fake_doc("new_id", "h2")]
    result = idx._upsert("file", docs, ["new_id"], prune_missing=True)

    fake_db.delete.assert_called_once_with(["old_id"])
    assert result["deleted"] == 1


def test_upsert_no_prune_suppresses_delete():
    idx = _make_indexer()
    fake_db = MagicMock()
    fake_db.get.return_value = {
        "ids": ["old_id"],
        "metadatas": [{"content_hash": "h"}],
    }
    idx._dbs["file"] = fake_db

    docs = [_fake_doc("new_id", "h2")]
    result = idx._upsert("file", docs, ["new_id"], prune_missing=False)

    fake_db.delete.assert_not_called()
    assert result["deleted"] == 0


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------


