"""
Tests for GCR2.1 — Dependency Graph Extraction.

Covers:
  _EdgeCollector / PythonASTParser.parse_edges
  - IMPORTS edges for 'import foo' and 'from foo import Bar'
  - IMPORTS with 'as' alias resolves to qualified name
  - relative imports are skipped
  - EXTENDS edge for non-Protocol base class
  - IMPLEMENTS edge for Protocol base
  - IMPLEMENTS edge for ABC base
  - CALLS edge for direct call to imported name
  - CALLS edge for module.attribute call pattern
  - duplicate calls produce only one edge (dedup by edge_id)
  - syntax error returns empty list

  GraphStore
  - upsert and query by edge_type
  - upsert is idempotent (same edges twice → count stays same)
  - delete_repo_edges removes only the target repo
  - delete_file_edges removes only the target file
  - stats() returns correct total count
  - get_edges filter combinations (src_id, repo_id)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

# Synthetic source covering all edge types.
SOURCE = '''\
"""Module with various dependency patterns."""

import os
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Protocol
from rag.code.indexer import CodeIndexer
from rag.retrieval.bm25 import BM25Index as BM25


class BaseEngine(ABC):
    """Inherits ABC -> IMPLEMENTS."""
    pass


class CodeEngine(CodeIndexer):
    """Inherits CodeIndexer -> EXTENDS."""

    def run(self) -> None:
        result = BM25()
        path = Path("/tmp")


class Queryable(Protocol):
    """Protocol subclass -> IMPLEMENTS."""
    def query(self) -> str: ...


def standalone() -> None:
    CodeIndexer()
    BM25()
    os.getcwd()
'''

REPO_ID   = "test-repo"
FILE_PATH = "rag/engine.py"


def _parse_edges():
    from rag.code.ast_parser import PythonASTParser
    parser = PythonASTParser()
    return parser.parse_edges(SOURCE, FILE_PATH, REPO_ID)


# ---------------------------------------------------------------------------
# IMPORTS edges
# ---------------------------------------------------------------------------

def test_imports_edge_bare_import():
    edges = _parse_edges()
    types  = {e.edge_type for e in edges}
    assert "IMPORTS" in types
    dst_ids = {e.dst_id for e in edges if e.edge_type == "IMPORTS"}
    assert "import::os" in dst_ids
    assert "import::sys" in dst_ids


def test_imports_edge_from_import():
    edges = _parse_edges()
    dst_ids = {e.dst_id for e in edges if e.edge_type == "IMPORTS"}
    assert "import::pathlib.Path" in dst_ids
    assert "import::abc.ABC" in dst_ids
    assert "import::abc.abstractmethod" in dst_ids
    assert "import::typing.Protocol" in dst_ids


def test_imports_edge_qualified_module():
    edges = _parse_edges()
    dst_ids = {e.dst_id for e in edges if e.edge_type == "IMPORTS"}
    assert "import::rag.code.indexer.CodeIndexer" in dst_ids


def test_imports_as_alias_uses_original_qualified_name():
    # 'from rag.retrieval.bm25 import BM25Index as BM25'
    # -> dst should be the original name, not the alias
    edges = _parse_edges()
    dst_ids = {e.dst_id for e in edges if e.edge_type == "IMPORTS"}
    assert "import::rag.retrieval.bm25.BM25Index" in dst_ids
    # alias 'BM25' must NOT appear as dst
    assert "import::rag.retrieval.bm25.BM25" not in dst_ids


def test_relative_import_skipped():
    from rag.code.ast_parser import PythonASTParser
    source = "from . import utils\nfrom .helpers import parse\n"
    edges = PythonASTParser().parse_edges(source, "rag/engine.py", "r")
    assert edges == []


def test_imports_src_is_module_symbol():
    edges = _parse_edges()
    mod_id = f"{REPO_ID}::{FILE_PATH}::module::<module>"
    import_edges = [e for e in edges if e.edge_type == "IMPORTS"]
    assert all(e.src_id == mod_id for e in import_edges)


# ---------------------------------------------------------------------------
# EXTENDS / IMPLEMENTS edges
# ---------------------------------------------------------------------------

def test_extends_edge_for_non_protocol_base():
    edges = _parse_edges()
    extends = [e for e in edges if e.edge_type == "EXTENDS"]
    dst_ids = {e.dst_id for e in extends}
    assert "import::rag.code.indexer.CodeIndexer" in dst_ids


def test_implements_edge_for_abc():
    edges = _parse_edges()
    impls = [e for e in edges if e.edge_type == "IMPLEMENTS"]
    dst_ids = {e.dst_id for e in impls}
    assert "import::abc.ABC" in dst_ids


def test_implements_edge_for_protocol():
    edges = _parse_edges()
    impls = [e for e in edges if e.edge_type == "IMPLEMENTS"]
    dst_ids = {e.dst_id for e in impls}
    assert "import::typing.Protocol" in dst_ids


def test_extends_src_is_class_symbol():
    edges = _parse_edges()
    extends = [e for e in edges if e.edge_type == "EXTENDS"]
    src_ids = {e.src_id for e in extends}
    expected = f"{REPO_ID}::{FILE_PATH}::class::CodeEngine"
    assert expected in src_ids


def test_implements_src_is_class_symbol():
    edges = _parse_edges()
    impls = [e for e in edges if e.edge_type == "IMPLEMENTS"]
    src_ids = {e.src_id for e in impls}
    base_engine_id = f"{REPO_ID}::{FILE_PATH}::class::BaseEngine"
    queryable_id   = f"{REPO_ID}::{FILE_PATH}::class::Queryable"
    assert base_engine_id in src_ids
    assert queryable_id   in src_ids


# ---------------------------------------------------------------------------
# CALLS edges
# ---------------------------------------------------------------------------

def test_calls_edge_for_imported_constructor():
    edges = _parse_edges()
    calls = [e for e in edges if e.edge_type == "CALLS"]
    dst_ids = {e.dst_id for e in calls}
    # CodeIndexer() in standalone() -> CALLS
    assert "import::rag.code.indexer.CodeIndexer" in dst_ids


def test_calls_edge_alias_resolves_to_original():
    edges = _parse_edges()
    calls = [e for e in edges if e.edge_type == "CALLS"]
    dst_ids = {e.dst_id for e in calls}
    # BM25() in standalone() — alias of BM25Index
    assert "import::rag.retrieval.bm25.BM25Index" in dst_ids


def test_calls_edge_module_attribute_pattern():
    edges = _parse_edges()
    calls = [e for e in edges if e.edge_type == "CALLS"]
    dst_ids = {e.dst_id for e in calls}
    # os.getcwd() -> CALLS to import::os.getcwd
    assert "import::os.getcwd" in dst_ids


def test_calls_no_duplicate_edges():
    # BM25() is called twice (once inside CodeEngine.run, once in standalone).
    # But edge_id is keyed on (src, type, dst, file) so only one CALLS edge
    # for BM25Index should exist.
    edges = _parse_edges()
    bm25_calls = [
        e for e in edges
        if e.edge_type == "CALLS"
        and e.dst_id == "import::rag.retrieval.bm25.BM25Index"
    ]
    assert len(bm25_calls) == 1


def test_non_imported_calls_not_tracked():
    from rag.code.ast_parser import PythonASTParser
    source = "def foo():\n    bar()\n    len([1, 2, 3])\n"
    edges = PythonASTParser().parse_edges(source, "x.py", "r")
    calls = [e for e in edges if e.edge_type == "CALLS"]
    assert calls == []   # bar and len are not imported


# ---------------------------------------------------------------------------
# Syntax error
# ---------------------------------------------------------------------------

def test_syntax_error_returns_empty():
    from rag.code.ast_parser import PythonASTParser
    edges = PythonASTParser().parse_edges("def broken(:\n    pass\n", "x.py", "r")
    assert edges == []


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

def _make_store(tmp_path):
    from rag.code.graph_store import GraphStore
    return GraphStore(str(tmp_path / "graph.db"))


def test_graph_store_upsert_and_query(tmp_path):
    store = _make_store(tmp_path)
    edges = _parse_edges()
    store.upsert_edges(edges)
    imports = store.get_edges(edge_type="IMPORTS")
    assert len(imports) > 0
    assert all(e.edge_type == "IMPORTS" for e in imports)


def test_graph_store_upsert_idempotent(tmp_path):
    store = _make_store(tmp_path)
    edges = _parse_edges()
    first  = store.upsert_edges(edges)
    second = store.upsert_edges(edges)   # same edges again
    assert first  == len(edges)
    assert second == 0                   # all duplicates, none inserted


def test_graph_store_stats(tmp_path):
    store = _make_store(tmp_path)
    edges = _parse_edges()
    store.upsert_edges(edges)
    s = store.stats()
    assert s["edges"] == len(edges)


def test_graph_store_filter_by_edge_type(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_edges(_parse_edges())
    extends  = store.get_edges(edge_type="EXTENDS")
    impls    = store.get_edges(edge_type="IMPLEMENTS")
    calls    = store.get_edges(edge_type="CALLS")
    imports  = store.get_edges(edge_type="IMPORTS")
    # Each filter returns only the requested type
    assert all(e.edge_type == "EXTENDS"    for e in extends)
    assert all(e.edge_type == "IMPLEMENTS" for e in impls)
    assert all(e.edge_type == "CALLS"      for e in calls)
    assert all(e.edge_type == "IMPORTS"    for e in imports)
    # Total must equal all edges
    total = len(extends) + len(impls) + len(calls) + len(imports)
    assert total == store.stats()["edges"]


def test_graph_store_filter_by_repo_id(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_edges(_parse_edges())
    # Index a second repo
    from rag.code.ast_parser import PythonASTParser
    other_edges = PythonASTParser().parse_edges(
        "import os\n", "other.py", "other-repo"
    )
    store.upsert_edges(other_edges)
    repo_edges = store.get_edges(repo_id=REPO_ID)
    assert all(e.repo_id == REPO_ID for e in repo_edges)
    other_repo_edges = store.get_edges(repo_id="other-repo")
    assert all(e.repo_id == "other-repo" for e in other_repo_edges)


def test_graph_store_delete_repo_edges(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_edges(_parse_edges())
    from rag.code.ast_parser import PythonASTParser
    other = PythonASTParser().parse_edges("import os\n", "other.py", "other-repo")
    store.upsert_edges(other)

    deleted = store.delete_repo_edges(REPO_ID)
    assert deleted > 0
    assert store.get_edges(repo_id=REPO_ID) == []
    # other-repo edges must be intact
    assert len(store.get_edges(repo_id="other-repo")) == len(other)


def test_graph_store_delete_file_edges(tmp_path):
    store = _make_store(tmp_path)
    edges = _parse_edges()
    store.upsert_edges(edges)
    deleted = store.delete_file_edges(REPO_ID, FILE_PATH)
    assert deleted == len(edges)
    assert store.get_edges(file_path=FILE_PATH) == []


def test_graph_store_filter_by_src_id(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_edges(_parse_edges())
    mod_id = f"{REPO_ID}::{FILE_PATH}::module::<module>"
    edges = store.get_edges(src_id=mod_id)
    assert len(edges) > 0
    assert all(e.src_id == mod_id for e in edges)


# ---------------------------------------------------------------------------
# DependencyEdge schema
# ---------------------------------------------------------------------------

def test_dependency_edge_roundtrip():
    from rag.code.schema import DependencyEdge
    e = DependencyEdge(
        edge_id="abc123",
        src_id="repo::file.py::module::<module>",
        dst_id="import::os",
        edge_type="IMPORTS",
        repo_id="repo",
        file_path="file.py",
        line_no=1,
    )
    assert DependencyEdge.from_dict(e.to_dict()) == e


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


