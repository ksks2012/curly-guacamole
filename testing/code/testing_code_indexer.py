"""Smoke test for GCR1.4 — Multi-resolution CodeIndexer."""

from __future__ import annotations

import hashlib
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import NamedTuple

import pytest

from rag.code.ast_parser import PythonASTParser
from rag.code.indexer import CodeIndexer
from rag.code.scanner import RepoScanner
from rag.code.schema import CodeChunk, RepoManifest
from rag.code.symbol_store import SymbolStore


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ID   = "langchain-test"


# ---------------------------------------------------------------------------
# Fake embeddings
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.0] * 16 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 16


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

class _Fixture(NamedTuple):
    manifest:   RepoManifest
    all_chunks: list[CodeChunk]
    store:      SymbolStore


def _build_fixture() -> _Fixture:
    scanner  = RepoScanner()
    manifest = scanner.scan(str(REPO_ROOT), repo_id=REPO_ID)
    source   = manifest.source_files()
    parser   = PythonASTParser()
    all_chunks: list[CodeChunk] = []
    for rf in source:
        if rf.language == "Python":
            path = Path(manifest.repo_root) / rf.file_path
            if path.exists():
                all_chunks.extend(parser.parse_file(path, REPO_ROOT, REPO_ID))
    assert all_chunks, "No chunks parsed"
    store = SymbolStore.from_chunks(all_chunks)
    return _Fixture(manifest=manifest, all_chunks=all_chunks, store=store)


def _make_indexer(tmpdir: str) -> CodeIndexer:
    return CodeIndexer(
        persist_directory=tmpdir,
        embedding_function=_FakeEmbeddings(),
        collection_prefix="test",
    )


@pytest.fixture(scope="module")
def code_index_state(tmp_path_factory):
    """Build the repo fixture, run the first index, then yield shared state."""
    tmpdir = str(tmp_path_factory.mktemp("code_index"))
    fx = _build_fixture()
    indexer = _make_indexer(tmpdir)
    counts = indexer.collection_stats()
    # Run first index
    results = indexer.index_all(fx.manifest, fx.all_chunks, fx.store)
    counts = indexer.collection_stats()
    return {"tmpdir": tmpdir, "fx": fx, "counts": counts}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_first_index(code_index_state):
    counts = code_index_state["counts"]
    assert counts["repo"]   == 1,  f"repo count: {counts['repo']}"
    assert counts["file"]   > 0,   f"file count: {counts['file']}"
    assert counts["symbol"] > 0,   f"symbol count: {counts['symbol']}"
    fx = code_index_state["fx"]
    assert counts["block"] == len(fx.all_chunks), \
        f"block count: {counts['block']} vs {len(fx.all_chunks)}"


@pytest.mark.integration
def test_reindex_skipped(code_index_state):
    tmpdir  = code_index_state["tmpdir"]
    fx      = code_index_state["fx"]
    indexer = _make_indexer(tmpdir)
    results = indexer.index_all(fx.manifest, fx.all_chunks, fx.store)
    for level, stats in results.items():
        assert stats["added"]   == 0, f"[{level}] unexpected adds on re-index"
        assert stats["updated"] == 0, f"[{level}] unexpected updates on re-index"
        assert stats["deleted"] == 0, f"[{level}] unexpected deletes on re-index"


@pytest.mark.integration
def test_update_one_chunk(code_index_state):
    tmpdir  = code_index_state["tmpdir"]
    fx      = code_index_state["fx"]
    indexer = _make_indexer(tmpdir)
    target   = fx.all_chunks[5]
    modified = dc_replace(target, code=target.code + "\n# modified")
    modified = dc_replace(
        modified,
        content_hash=hashlib.sha256(modified.code.encode()).hexdigest(),
    )
    patched = fx.all_chunks[:5] + [modified] + fx.all_chunks[6:]
    stats = indexer.index_blocks(patched)
    assert stats["updated"] == 1, f"expected 1 updated, got {stats['updated']}"
    assert stats["added"]   == 0
    assert stats["deleted"] == 0


@pytest.mark.integration
def test_delete_one_chunk(code_index_state):
    tmpdir  = code_index_state["tmpdir"]
    fx      = code_index_state["fx"]
    indexer = _make_indexer(tmpdir)
    trimmed = fx.all_chunks[:5] + fx.all_chunks[6:]
    stats   = indexer.index_blocks(trimmed)
    assert stats["deleted"] == 1, f"expected 1 deleted, got {stats['deleted']}"
    # Restore
    indexer.index_blocks(fx.all_chunks)


@pytest.mark.integration
def test_search(code_index_state):
    tmpdir  = code_index_state["tmpdir"]
    indexer = _make_indexer(tmpdir)
    assert len(indexer.search("class that handles configuration", level="symbol", k=3)) > 0
    assert len(indexer.search("def __init__", level="block", k=3)) > 0
    assert len(indexer.search("indexer", level="file", k=3)) > 0


@pytest.mark.integration
def test_delete_repo(code_index_state):
    tmpdir  = code_index_state["tmpdir"]
    indexer = _make_indexer(tmpdir)
    indexer.delete_repo(REPO_ID)
    stats = indexer.collection_stats()
    for level in ("file", "symbol", "block"):
        assert stats[level] == 0, \
            f"[{level}] still has {stats[level]} docs after delete_repo"
