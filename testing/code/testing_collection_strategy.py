"""
Smoke tests for Phase 2 Step 2.3a — Collection Consolidation.

Tests:
  - CodeIndexer.LEVELS no longer contains "repo"
  - LEVELS contains exactly ("file", "symbol", "block")
  - Default collection names: file→documents, symbol→symbols, block→code_block
  - collection_names override works per-level
  - collection_name() falls back to {prefix}_{level} for unknown levels
  - index_files() injects source_type="code" and repo-level metadata
  - index_files() without manifest omits repo metadata gracefully
  - index_all() returns file/symbol/block keys (no "repo" key)
  - index_all() does NOT call index_manifest()
  - CodeRetriever rejects level="repo"
  - CodeRetriever accepts level="file", "symbol", "block"
  - CodeIndexer._agg handles 3-level stats dict (no repo)
  - BaseIndexer.reindex stats aggregated correctly
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# CodeIndexer collection strategy tests
# ---------------------------------------------------------------------------

def test_levels_no_repo():
    from rag.code.indexer import CodeIndexer
    assert "repo" not in CodeIndexer.LEVELS

def test_levels_exactly_file_symbol_block():
    from rag.code.indexer import CodeIndexer
    assert set(CodeIndexer.LEVELS) == {"file", "symbol", "block"}

def test_default_collection_names():
    from rag.code.indexer import CodeIndexer
    idx = CodeIndexer.__new__(CodeIndexer)
    idx._prefix = "code"
    idx._collection_names = {**CodeIndexer._DEFAULT_COLLECTION_NAMES}
    idx._dbs = {}
    assert idx.collection_name("file")   == "documents"
    assert idx.collection_name("symbol") == "symbols"
    assert idx.collection_name("block")  == "code_block"

def test_collection_names_override():
    from rag.code.indexer import CodeIndexer
    idx = CodeIndexer.__new__(CodeIndexer)
    idx._prefix = "code"
    idx._collection_names = {**CodeIndexer._DEFAULT_COLLECTION_NAMES, "file": "rag_collection"}
    idx._dbs = {}
    assert idx.collection_name("file") == "rag_collection"
    assert idx.collection_name("symbol") == "symbols"

def test_collection_name_fallback_for_unknown_level():
    from rag.code.indexer import CodeIndexer
    idx = CodeIndexer.__new__(CodeIndexer)
    idx._prefix = "code"
    idx._collection_names = {**CodeIndexer._DEFAULT_COLLECTION_NAMES}
    idx._dbs = {}
    # repo is no longer in LEVELS but collection_name() should still work
    # as a fallback for external callers that still reference it
    assert idx.collection_name("repo") == "code_repo"

def test_init_merges_collection_names():
    """Constructor should merge caller overrides on top of defaults."""
    from rag.code.indexer import CodeIndexer
    fake_embed = MagicMock()
    idx = CodeIndexer(
        persist_directory="/fake",
        embedding_function=fake_embed,
        collection_names={"file": "my_docs"},
    )
    assert idx.collection_name("file")   == "my_docs"
    assert idx.collection_name("symbol") == "symbols"    # default preserved

# ---------------------------------------------------------------------------
# index_files() metadata tests
# ---------------------------------------------------------------------------

def _make_idx():
    from rag.code.indexer import CodeIndexer
    idx = CodeIndexer.__new__(CodeIndexer)
    idx._prefix = "code"
    idx._collection_names = {**CodeIndexer._DEFAULT_COLLECTION_NAMES}
    idx._dbs = {}
    idx._persist_dir = "/fake"
    idx._embed = MagicMock()
    return idx

def _make_chunk(repo_id="r", file_path="f.py", chunk_type="module"):
    from rag.code.schema import CodeChunk
    return CodeChunk(
        chunk_id=f"{repo_id}::{file_path}::{chunk_type}::x",
        content="def f(): pass",
        repo_id=repo_id,
        file_path=file_path,
        language="python",
        chunk_type=chunk_type,
        name="<module>" if chunk_type == "module" else "f",
        start_line=1,
        end_line=1,
        content_hash="abc",
    )

def test_index_files_injects_source_type_code():
    idx = _make_idx()
    captured_docs: list = []

    def fake_upsert(level, docs, ids, **kwargs):
        captured_docs.extend(docs)
        return {"added": len(docs), "updated": 0, "skipped": 0, "deleted": 0}

    idx._upsert = fake_upsert
    chunk = _make_chunk()
    idx.index_files([chunk])

    assert len(captured_docs) == 1
    assert captured_docs[0].metadata["source_type"] == "code"

def test_index_files_injects_repo_metadata_when_manifest_provided():
    idx = _make_idx()
    captured_docs: list = []

    def fake_upsert(level, docs, ids, **kwargs):
        captured_docs.extend(docs)
        return {"added": 1, "updated": 0, "skipped": 0, "deleted": 0}

    idx._upsert = fake_upsert

    chunk = _make_chunk()

    manifest = MagicMock()
    manifest.repo_root  = "/home/user/myrepo"
    manifest.branch     = "main"
    manifest.scanned_at = "2026-05-21T10:00:00+00:00"
    manifest.files      = {}   # no RepoFile for this path, just blanks

    idx.index_files([chunk], manifest=manifest)

    meta = captured_docs[0].metadata
    assert meta["repo_root"]   == "/home/user/myrepo"
    assert meta["branch"]      == "main"
    assert meta["scanned_at"]  == "2026-05-21T10:00:00+00:00"

def test_index_files_no_manifest_uses_empty_strings():
    idx = _make_idx()
    captured_docs: list = []

    def fake_upsert(level, docs, ids, **kwargs):
        captured_docs.extend(docs)
        return {"added": 1, "updated": 0, "skipped": 0, "deleted": 0}

    idx._upsert = fake_upsert
    chunk = _make_chunk()
    idx.index_files([chunk])

    meta = captured_docs[0].metadata
    assert meta["repo_root"]  == ""
    assert meta["branch"]     == ""
    assert meta["scanned_at"] == ""

# ---------------------------------------------------------------------------
# index_all() no longer calls index_manifest()
# ---------------------------------------------------------------------------

def test_index_all_has_no_repo_key():
    idx = _make_idx()
    manifest = MagicMock()
    manifest.repo_root  = ""
    manifest.branch     = ""
    manifest.scanned_at = ""
    manifest.files = {}
    manifest.repo_id = "r"

    chunk = _make_chunk()
    idx._upsert = MagicMock(return_value={"added": 0, "updated": 0, "skipped": 0, "deleted": 0})
    idx.index_manifest = MagicMock()

    result = idx.index_all(manifest, [chunk])

    assert "repo" not in result, f"'repo' key should not be in index_all() result: {result.keys()}"
    assert "file"   in result
    assert "symbol" in result
    assert "block"  in result

def test_index_all_does_not_call_index_manifest():
    idx = _make_idx()
    manifest = MagicMock()
    manifest.repo_root  = ""
    manifest.branch     = ""
    manifest.scanned_at = ""
    manifest.files = {}
    manifest.repo_id = "r"

    chunk = _make_chunk()
    idx._upsert = MagicMock(return_value={"added": 0, "updated": 0, "skipped": 0, "deleted": 0})
    idx.index_manifest = MagicMock()

    idx.index_all(manifest, [chunk])
    idx.index_manifest.assert_not_called()

# ---------------------------------------------------------------------------
# CodeRetriever level validation
# ---------------------------------------------------------------------------

def test_code_retriever_rejects_repo_level():
    from rag.retrieval.code_retriever import CodeRetriever
    try:
        CodeRetriever(MagicMock(), level="repo")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "repo" in str(e).lower() or "invalid" in str(e).lower()

def test_code_retriever_accepts_file_symbol_block():
    from rag.retrieval.code_retriever import CodeRetriever
    for level in ("file", "symbol", "block"):
        CodeRetriever(MagicMock(), level=level)

# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------


