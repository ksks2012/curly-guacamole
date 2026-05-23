"""
Smoke tests for RepoIndex (GCR1.4 Phase A — repo-level index).

Tests:
  - RepoIndex is a BaseIndexer subclass
  - COLLECTION_NAME default is "code_repo"
  - collection_name override is respected in __init__
  - index_manifest() builds the correct Document and doc_id
  - index_manifest() content_hash is stable (same manifest → same hash)
  - index_manifest() content_hash changes when manifest files change
  - ingest() wraps index_manifest() and returns IndexStats
  - update() aliases ingest()
  - delete() removes by repo_id; noop when absent
  - reindex() = delete + ingest; combined stats returned
  - CodeIndexer.index_manifest() delegates to RepoIndex (compat shim)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo_index():
    from rag.code.repo_index import RepoIndex
    ri = RepoIndex.__new__(RepoIndex)
    ri._persist_dir  = "/fake"
    ri._embed        = MagicMock()
    ri._collection_name = "code_repo"
    ri._db_instance  = None
    return ri


def _make_manifest(repo_id: str = "testrepo", content_hash: str = "abc123"):
    from rag.code.schema import RepoFile, RepoManifest
    rf = RepoFile(
        repo_id=repo_id,
        branch="main",
        file_path="mod.py",
        language="Python",
        size=100,
        is_test=False,
        is_generated=False,
        content_hash=content_hash,
        mtime="2026-01-01T00:00:00+00:00",
    )
    return RepoManifest(
        repo_id=repo_id,
        repo_root="/repo",
        branch="main",
        scanned_at="2026-01-01T00:00:00+00:00",
        files={"mod.py": rf},
    )


# ---------------------------------------------------------------------------
# Class contract
# ---------------------------------------------------------------------------

def test_repo_index_is_base_indexer():
    from rag.indexer import BaseIndexer
    from rag.code.repo_index import RepoIndex
    assert issubclass(RepoIndex, BaseIndexer)


def test_repo_index_collection_name_default():
    from rag.code.repo_index import RepoIndex
    assert RepoIndex.COLLECTION_NAME == "code_repo"
    ri = _make_repo_index()
    assert ri._collection_name == "code_repo"


def test_repo_index_collection_name_override():
    from rag.code.repo_index import RepoIndex
    ri = RepoIndex("/fake", MagicMock(), collection_name="my_repo")
    assert ri._collection_name == "my_repo"


# ---------------------------------------------------------------------------
# index_manifest()
# ---------------------------------------------------------------------------

def test_index_manifest_builds_correct_doc_and_id():
    ri = _make_repo_index()
    manifest = _make_manifest()

    captured_docs: list = []
    captured_ids:  list = []

    def fake_upsert(docs, ids):
        captured_docs.extend(docs)
        captured_ids.extend(ids)
        return {"added": 1, "updated": 0, "skipped": 0, "deleted": 0}

    ri._upsert = fake_upsert
    ri.index_manifest(manifest)

    assert len(captured_docs) == 1
    assert captured_ids[0] == "testrepo::repo"
    meta = captured_docs[0].metadata
    assert meta["repo_id"]    == "testrepo"
    assert meta["branch"]     == "main"
    assert "content_hash"     in meta
    assert "file_count"       in meta
    assert "source_count"     in meta


def test_index_manifest_content_hash_stable():
    """Same manifest must produce the same content_hash on repeated calls."""
    ri = _make_repo_index()
    manifest = _make_manifest()

    hashes: list[str] = []

    def fake_upsert(docs, ids):
        hashes.append(docs[0].metadata["content_hash"])
        return {"added": 1, "updated": 0, "skipped": 0, "deleted": 0}

    ri._upsert = fake_upsert
    ri.index_manifest(manifest)
    ri.index_manifest(manifest)

    assert hashes[0] == hashes[1], "content_hash must be deterministic"


def test_index_manifest_content_hash_changes_with_file_hash():
    """Changing a file's content_hash must produce a different repo hash."""
    ri = _make_repo_index()

    hashes: list[str] = []

    def fake_upsert(docs, ids):
        hashes.append(docs[0].metadata["content_hash"])
        return {"added": 1, "updated": 0, "skipped": 0, "deleted": 0}

    ri._upsert = fake_upsert
    ri.index_manifest(_make_manifest(content_hash="aaa"))
    ri.index_manifest(_make_manifest(content_hash="bbb"))

    assert hashes[0] != hashes[1], "Different file hashes must yield different repo hash"


def test_index_manifest_with_chunks_extracts_docstrings():
    """When chunks are provided, module docstrings appear in the document text."""
    from rag.code.schema import CodeChunk
    ri = _make_repo_index()
    manifest = _make_manifest()

    chunk = CodeChunk(
        chunk_id="testrepo::mod.py::module::x",
        content="",
        repo_id="testrepo",
        file_path="mod.py",
        language="python",
        chunk_type="module",
        name="<module>",
        start_line=1,
        end_line=1,
        content_hash="abc",
        docstring="Handles authentication logic.",
    )

    captured: list = []

    def fake_upsert(docs, ids):
        captured.extend(docs)
        return {"added": 1, "updated": 0, "skipped": 0, "deleted": 0}

    ri._upsert = fake_upsert
    ri.index_manifest(manifest, chunks=[chunk])

    assert "authentication" in captured[0].page_content


# ---------------------------------------------------------------------------
# ingest() / update()
# ---------------------------------------------------------------------------

def test_repo_index_ingest_returns_index_stats():
    from rag.indexer import IndexStats
    ri = _make_repo_index()
    ri._upsert = MagicMock(return_value={"added": 1, "updated": 0, "skipped": 0, "deleted": 0})

    stats = ri.ingest(_make_manifest())

    assert isinstance(stats, IndexStats)
    assert stats.added == 1


def test_repo_index_update_aliases_ingest():
    from rag.indexer import IndexStats
    ri = _make_repo_index()
    ri._upsert = MagicMock(return_value={"added": 0, "updated": 1, "skipped": 0, "deleted": 0})

    stats = ri.update(_make_manifest())

    assert isinstance(stats, IndexStats)
    assert stats.updated == 1
    ri._upsert.assert_called_once()


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

def test_repo_index_delete_removes_by_repo_id():
    from rag.indexer import IndexStats
    ri = _make_repo_index()
    fake_db = MagicMock()
    fake_db.get.return_value = {"ids": ["testrepo::repo"]}
    ri._db = MagicMock(return_value=fake_db)

    stats = ri.delete("testrepo")

    fake_db.get.assert_called_once_with(where={"repo_id": "testrepo"}, include=[])
    fake_db.delete.assert_called_once_with(["testrepo::repo"])
    assert isinstance(stats, IndexStats)
    assert stats.deleted == 1


def test_repo_index_delete_no_docs_is_noop():
    ri = _make_repo_index()
    fake_db = MagicMock()
    fake_db.get.return_value = {"ids": []}
    ri._db = MagicMock(return_value=fake_db)

    stats = ri.delete("nonexistent")
    fake_db.delete.assert_not_called()
    assert stats.deleted == 0


# ---------------------------------------------------------------------------
# reindex()
# ---------------------------------------------------------------------------

def test_repo_index_reindex_combines_stats():
    from rag.indexer import IndexStats
    ri = _make_repo_index()
    manifest = _make_manifest()
    ri.delete = MagicMock(return_value=IndexStats(deleted=1))
    ri.ingest  = MagicMock(return_value=IndexStats(added=1))

    stats = ri.reindex(manifest)

    ri.delete.assert_called_once_with("testrepo")
    ri.ingest.assert_called_once_with(manifest, chunks=None)
    assert stats.deleted == 1 and stats.added == 1


def test_repo_index_reindex_passes_chunks():
    from rag.indexer import IndexStats
    ri = _make_repo_index()
    manifest = _make_manifest()
    fake_chunks = [MagicMock()]
    ri.delete = MagicMock(return_value=IndexStats(deleted=1))
    ri.ingest  = MagicMock(return_value=IndexStats(added=1))

    ri.reindex(manifest, chunks=fake_chunks)

    ri.ingest.assert_called_once_with(manifest, chunks=fake_chunks)


# ---------------------------------------------------------------------------
# collection_stats()
# ---------------------------------------------------------------------------

def test_repo_index_collection_stats_returns_int():
    ri = _make_repo_index()
    fake_db = MagicMock()
    fake_db.get.return_value = {"ids": ["testrepo::repo", "other::repo"]}
    ri._db = MagicMock(return_value=fake_db)

    count = ri.collection_stats()
    assert count == 2


# ---------------------------------------------------------------------------
# CodeIndexer.index_manifest() backward-compat shim
# ---------------------------------------------------------------------------

def test_code_indexer_index_manifest_delegates_to_repo_index():
    """CodeIndexer.index_manifest() must forward to RepoIndex.index_manifest()."""
    from rag.code.indexer import CodeIndexer
    idx = CodeIndexer.__new__(CodeIndexer)
    idx._persist_dir = "/fake"
    idx._embed       = MagicMock()
    idx._prefix      = "code"
    idx._dbs         = {}
    idx._collection_names = {**CodeIndexer._DEFAULT_COLLECTION_NAMES}

    manifest = _make_manifest()
    expected = {"added": 1, "updated": 0, "skipped": 0, "deleted": 0}

    with patch("rag.code.repo_index.RepoIndex") as MockRI:
        MockRI.return_value.index_manifest.return_value = expected
        result = idx.index_manifest(manifest)

    MockRI.assert_called_once_with("/fake", idx._embed)
    MockRI.return_value.index_manifest.assert_called_once_with(manifest, None)
    assert result == expected


def test_code_indexer_index_manifest_passes_chunks():
    from rag.code.indexer import CodeIndexer
    idx = CodeIndexer.__new__(CodeIndexer)
    idx._persist_dir = "/fake"
    idx._embed       = MagicMock()
    idx._prefix      = "code"
    idx._dbs         = {}
    idx._collection_names = {**CodeIndexer._DEFAULT_COLLECTION_NAMES}

    manifest    = _make_manifest()
    fake_chunks = [MagicMock()]

    with patch("rag.code.repo_index.RepoIndex") as MockRI:
        MockRI.return_value.index_manifest.return_value = {}
        idx.index_manifest(manifest, chunks=fake_chunks)

    MockRI.return_value.index_manifest.assert_called_once_with(manifest, fake_chunks)
