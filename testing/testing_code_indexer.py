"""Smoke test for GCR1.4 — Multi-resolution CodeIndexer.

Uses a deterministic fake embeddings implementation (no server required)
and a temporary Chroma directory so the test is fully self-contained.

Assertions
----------
1. index_all() → correct document counts in all 4 collections
2. Re-index with same data → all docs skipped (incremental deduplication)
3. index_blocks() with one modified chunk → 1 updated, rest skipped
4. index_blocks() with one chunk removed → 1 deleted (per-repo prune)
5. search() returns results from symbol and block collections
6. delete_repo() clears all collections
"""

import hashlib
import sys
import tempfile
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.code.ast_parser import PythonASTParser
from rag.code.indexer import CodeIndexer
from rag.code.scanner import RepoScanner
from rag.code.schema import CodeChunk, RepoManifest
from rag.code.symbol_store import SymbolStore


# ---------------------------------------------------------------------------
# Fake embeddings (no LLM server required)
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    """Deterministic fake embeddings for testing (16-dim zero vectors)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 16 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 16


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ID   = "langchain-test"


class _Fixture(NamedTuple):
    manifest:   RepoManifest
    all_chunks: list[CodeChunk]
    store:      SymbolStore


def build_fixture() -> _Fixture:
    print("Scanning repository …")
    scanner  = RepoScanner()
    manifest = scanner.scan(str(REPO_ROOT), repo_id=REPO_ID)
    source   = manifest.source_files()
    print(f"  manifest: {len(manifest.files)} files  source: {len(source)}")

    print("Parsing Python source files …")
    parser     = PythonASTParser()
    all_chunks: list[CodeChunk] = []
    for rf in source:
        if rf.language == "Python":
            path = Path(manifest.repo_root) / rf.file_path
            if path.exists():
                all_chunks.extend(parser.parse_file(path, REPO_ROOT, REPO_ID))

    print(f"  chunks: {len(all_chunks)}")
    assert all_chunks, "No chunks parsed — check Python source files exist"

    store = SymbolStore.from_chunks(all_chunks)
    print(f"  symbols: {store.summary()}")

    return _Fixture(manifest=manifest, all_chunks=all_chunks, store=store)


def _make_indexer(tmpdir: str) -> CodeIndexer:
    return CodeIndexer(
        persist_directory=tmpdir,
        embedding_function=_FakeEmbeddings(),
        collection_prefix="test",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_first_index(
    tmpdir: str,
    fx: _Fixture,
) -> dict[str, int]:
    """First-time index_all: every doc should be added, none skipped."""
    print("\n[Test 1] First-time index_all …")
    indexer = _make_indexer(tmpdir)
    results = indexer.index_all(fx.manifest, fx.all_chunks, fx.store)

    for level, stats in results.items():
        print(
            f"  [{level:6s}]  +{stats['added']} added  "
            f"~{stats['updated']} updated  "
            f"={stats['skipped']} skipped  "
            f"-{stats['deleted']} deleted"
        )
        assert stats["updated"] == 0, f"[{level}] unexpected updates: {stats['updated']}"
        assert stats["skipped"] == 0, f"[{level}] unexpected skips: {stats['skipped']}"
        assert stats["deleted"] == 0, f"[{level}] unexpected deletes: {stats['deleted']}"
        assert stats["added"]   > 0,  f"[{level}] nothing was added"

    counts = indexer.collection_stats()
    print(f"  collection_stats: {counts}")
    assert counts["repo"]   == 1,                  f"repo count: {counts['repo']}"
    assert counts["file"]   > 0,                   f"file count: {counts['file']}"
    assert counts["symbol"] > 0,                   f"symbol count: {counts['symbol']}"
    assert counts["block"]  == len(fx.all_chunks), \
        f"block count: {counts['block']} vs {len(fx.all_chunks)}"

    return counts


def test_reindex_skipped(tmpdir: str, fx: _Fixture) -> None:
    """Re-indexing unchanged data: every doc should be skipped."""
    print("\n[Test 2] Re-index (no changes) → all skipped …")
    indexer = _make_indexer(tmpdir)
    results = indexer.index_all(fx.manifest, fx.all_chunks, fx.store)

    for level, stats in results.items():
        assert stats["added"]   == 0, f"[{level}] unexpected adds on re-index"
        assert stats["updated"] == 0, f"[{level}] unexpected updates on re-index"
        assert stats["deleted"] == 0, f"[{level}] unexpected deletes on re-index"
    print("  OK — all skipped")


def test_update_one_chunk(tmpdir: str, fx: _Fixture) -> None:
    """Modifying one chunk's code: exactly 1 block update expected."""
    print("\n[Test 3] Modify one block chunk → 1 block updated …")
    indexer = _make_indexer(tmpdir)

    target   = fx.all_chunks[5]
    modified = dc_replace(target, code=target.code + "\n# modified")
    modified = dc_replace(
        modified,
        content_hash=hashlib.sha256(modified.code.encode()).hexdigest(),
    )
    patched = fx.all_chunks[:5] + [modified] + fx.all_chunks[6:]

    stats = indexer.index_blocks(patched)
    print(f"  block_stats: {stats}")
    assert stats["updated"] == 1, f"expected 1 updated, got {stats['updated']}"
    assert stats["added"]   == 0
    assert stats["deleted"] == 0


def test_delete_one_chunk(tmpdir: str, fx: _Fixture) -> None:
    """Removing one chunk from the list: exactly 1 block deletion expected."""
    print("\n[Test 4] Remove one block chunk → 1 block deleted …")
    indexer  = _make_indexer(tmpdir)
    trimmed  = fx.all_chunks[:5] + fx.all_chunks[6:]  # drop index 5

    stats = indexer.index_blocks(trimmed)
    print(f"  block_stats: {stats}")
    assert stats["deleted"] == 1, f"expected 1 deleted, got {stats['deleted']}"

    # Restore full state for subsequent tests
    indexer.index_blocks(fx.all_chunks)


def test_search(tmpdir: str) -> None:
    """search() should return non-empty results from each collection."""
    print("\n[Test 5] search() returns results …")
    indexer = _make_indexer(tmpdir)

    sym_results = indexer.search("class that handles configuration", level="symbol", k=3)
    blk_results = indexer.search("def __init__", level="block", k=3)
    fil_results = indexer.search("indexer", level="file", k=3)

    print(f"  symbol search: {len(sym_results)} results")
    print(f"  block  search: {len(blk_results)} results")
    print(f"  file   search: {len(fil_results)} results")
    assert len(sym_results) > 0, "symbol search returned nothing"
    assert len(blk_results) > 0, "block search returned nothing"
    assert len(fil_results) > 0, "file search returned nothing"


def test_delete_repo(tmpdir: str) -> None:
    """delete_repo() should clear all per-repo documents."""
    print("\n[Test 6] delete_repo() clears all collections …")
    indexer = _make_indexer(tmpdir)
    indexer.delete_repo(REPO_ID)

    stats = indexer.collection_stats()
    print(f"  stats after delete: {stats}")
    for level in ("file", "symbol", "block"):
        assert stats[level] == 0, \
            f"[{level}] still has {stats[level]} docs after delete_repo"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fx = build_fixture()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Tests 3–6 all share the same tmpdir so state carries forward
        # after the initial index created by test_first_index.
        counts = test_first_index(tmpdir, fx)
        test_reindex_skipped(tmpdir, fx)
        test_update_one_chunk(tmpdir, fx)
        test_delete_one_chunk(tmpdir, fx)
        test_search(tmpdir)
        test_delete_repo(tmpdir)

    print(f"""
Summary
-------
  manifest files  : {len(fx.manifest.files)}
  source files    : {len(fx.manifest.source_files())}
  chunks parsed   : {len(fx.all_chunks)}
  symbols indexed : {len(fx.store)}
  collections     : repo={counts['repo']}  file={counts['file']}  symbol={counts['symbol']}  block={counts['block']}

PASS""")


if __name__ == "__main__":
    main()
