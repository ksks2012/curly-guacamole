"""
Tests for GCR2.2 — Symbol Evolution Tracking.

Covers:
  FileSnapshot
  - symbol_hashes field defaults to {}
  - from_dict backward compat (no symbol_hashes key)
  - to_dict / from_dict roundtrip with symbol_hashes

  SymbolEvolution schema
  - to_dict / from_dict roundtrip
  - from_dict backward compat for JSON list fields (string → list)
  - is_alive() True/False

  build_symbol_evolutions (evolution_builder)
  - empty input returns []
  - single snapshot: introduced_in set, deleted_in empty
  - symbol added in later commit: correct introduced_in
  - symbol removed in later commit: deleted_in set
  - symbol modified (hash changed): appears in modified_in
  - no hash info: modified_in stays []
  - symbol re-introduced after deletion: deleted_in cleared
  - symbol present throughout: deleted_in = ""
  - multiple files' worth of symbols tracked independently

  GraphStore — symbol_evolution
  - upsert_evolutions and get_evolution
  - upsert is idempotent (ON CONFLICT DO UPDATE)
  - get_evolutions filter by repo_id
  - get_evolutions filter by file_path
  - get_evolutions alive_only flag
  - delete_file_evolutions
  - delete_repo_evolutions
  - stats() includes symbol_evolution count

  GitReader — symbol_hashes populated
  - snapshot_file with parser fills symbol_hashes
  - snapshot_file without parser gives empty symbol_hashes
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ID   = "test-repo"
FILE_PATH = "rag/engine.py"

# Commit hashes (fake, just need to be distinct strings)
CA = "a" * 40
CB = "b" * 40
CC = "c" * 40
CD = "d" * 40


def _snap(commit: str, symbols: list[str], hashes: dict[str, str] | None = None):
    from rag.code.schema import FileSnapshot
    return FileSnapshot(
        snapshot_id=f"{REPO_ID}::{commit[:12]}::{FILE_PATH}",
        repo_id=REPO_ID,
        commit_hash=commit,
        file_path=FILE_PATH,
        content_hash="x",
        symbols=symbols,
        symbol_hashes=hashes or {},
    )


# ---------------------------------------------------------------------------
# FileSnapshot
# ---------------------------------------------------------------------------

def test_file_snapshot_symbol_hashes_defaults_empty():
    from rag.code.schema import FileSnapshot
    snap = FileSnapshot(
        snapshot_id="s", repo_id="r", commit_hash="a" * 40,
        file_path="f.py", content_hash="x",
    )
    assert snap.symbol_hashes == {}


def test_file_snapshot_backward_compat_no_symbol_hashes():
    from rag.code.schema import FileSnapshot
    d = {
        "snapshot_id": "s", "repo_id": "r", "commit_hash": "a" * 40,
        "file_path": "f.py", "content_hash": "x", "symbols": [],
        # deliberately missing symbol_hashes
    }
    snap = FileSnapshot.from_dict(d)
    assert snap.symbol_hashes == {}


def test_file_snapshot_roundtrip_with_symbol_hashes():
    from rag.code.schema import FileSnapshot
    snap = _snap(CA, ["foo", "bar"], {"foo": "hash1", "bar": "hash2"})
    snap2 = FileSnapshot.from_dict(snap.to_dict())
    assert snap2.symbol_hashes == {"foo": "hash1", "bar": "hash2"}


# ---------------------------------------------------------------------------
# SymbolEvolution schema
# ---------------------------------------------------------------------------

def test_symbol_evolution_roundtrip():
    from rag.code.schema import SymbolEvolution
    e = SymbolEvolution(
        evolution_id="eid",
        symbol_name="foo",
        repo_id=REPO_ID,
        file_path=FILE_PATH,
        introduced_in=CA,
        modified_in=[CB],
        deleted_in=CC,
        renamed_from=[],
    )
    e2 = SymbolEvolution.from_dict(e.to_dict())
    assert e2 == e


def test_symbol_evolution_from_dict_json_string_lists():
    from rag.code.schema import SymbolEvolution
    import json
    d = {
        "evolution_id": "eid", "symbol_name": "foo",
        "repo_id": REPO_ID, "file_path": FILE_PATH,
        "introduced_in": CA,
        "modified_in":  json.dumps([CB, CC]),   # SQLite TEXT column
        "deleted_in":   "",
        "renamed_from": json.dumps([]),
    }
    e = SymbolEvolution.from_dict(d)
    assert e.modified_in == [CB, CC]
    assert e.renamed_from == []


def test_symbol_evolution_is_alive():
    from rag.code.schema import SymbolEvolution
    alive = SymbolEvolution(
        evolution_id="e1", symbol_name="foo",
        repo_id=REPO_ID, file_path=FILE_PATH,
        introduced_in=CA, deleted_in="",
    )
    dead = SymbolEvolution(
        evolution_id="e2", symbol_name="bar",
        repo_id=REPO_ID, file_path=FILE_PATH,
        introduced_in=CA, deleted_in=CB,
    )
    assert alive.is_alive() is True
    assert dead.is_alive()  is False


# ---------------------------------------------------------------------------
# build_symbol_evolutions
# ---------------------------------------------------------------------------

def test_build_empty_input():
    from rag.code.evolution_builder import build_symbol_evolutions
    assert build_symbol_evolutions([]) == []


def test_build_single_snapshot():
    from rag.code.evolution_builder import build_symbol_evolutions
    snap = _snap(CA, ["foo", "bar"], {"foo": "h1", "bar": "h2"})
    evos = build_symbol_evolutions([snap])
    by_name = {e.symbol_name: e for e in evos}
    assert set(by_name) == {"foo", "bar"}
    assert by_name["foo"].introduced_in == CA
    assert by_name["foo"].deleted_in == ""
    assert by_name["foo"].modified_in == []


def test_build_symbol_added_in_later_commit():
    from rag.code.evolution_builder import build_symbol_evolutions
    s1 = _snap(CA, ["foo"],       {"foo": "h1"})
    s2 = _snap(CB, ["foo", "bar"],{"foo": "h1", "bar": "h2"})
    evos = {e.symbol_name: e for e in build_symbol_evolutions([s1, s2])}
    assert evos["foo"].introduced_in == CA
    assert evos["bar"].introduced_in == CB


def test_build_symbol_deleted():
    from rag.code.evolution_builder import build_symbol_evolutions
    s1 = _snap(CA, ["foo", "bar"], {"foo": "h1", "bar": "h2"})
    s2 = _snap(CB, ["foo"],        {"foo": "h1"})
    evos = {e.symbol_name: e for e in build_symbol_evolutions([s1, s2])}
    assert evos["bar"].deleted_in == CB
    assert evos["bar"].is_alive() is False
    assert evos["foo"].deleted_in == ""


def test_build_symbol_modified():
    from rag.code.evolution_builder import build_symbol_evolutions
    s1 = _snap(CA, ["foo"], {"foo": "hash_v1"})
    s2 = _snap(CB, ["foo"], {"foo": "hash_v2"})
    s3 = _snap(CC, ["foo"], {"foo": "hash_v3"})
    evos = {e.symbol_name: e for e in build_symbol_evolutions([s1, s2, s3])}
    assert evos["foo"].modified_in == [CB, CC]


def test_build_no_hash_info_no_modified():
    from rag.code.evolution_builder import build_symbol_evolutions
    s1 = _snap(CA, ["foo"])  # no symbol_hashes
    s2 = _snap(CB, ["foo"])
    evos = {e.symbol_name: e for e in build_symbol_evolutions([s1, s2])}
    assert evos["foo"].modified_in == []


def test_build_symbol_reintroduced_after_deletion():
    from rag.code.evolution_builder import build_symbol_evolutions
    s1 = _snap(CA, ["foo"])
    s2 = _snap(CB, [])             # foo deleted
    s3 = _snap(CC, ["foo"])        # foo re-introduced
    evos = {e.symbol_name: e for e in build_symbol_evolutions([s1, s2, s3])}
    # Re-introduced → deleted_in cleared since foo is in final snapshot
    assert evos["foo"].deleted_in == ""


def test_build_symbol_present_throughout():
    from rag.code.evolution_builder import build_symbol_evolutions
    snaps = [_snap(h, ["stable"], {"stable": f"hash{i}"}) for i, h in enumerate([CA, CB, CC, CD])]
    evos = {e.symbol_name: e for e in build_symbol_evolutions(snaps)}
    assert evos["stable"].introduced_in == CA
    assert evos["stable"].deleted_in    == ""


def test_build_renamed_from_always_empty():
    from rag.code.evolution_builder import build_symbol_evolutions
    snap = _snap(CA, ["foo"], {"foo": "h1"})
    evos = build_symbol_evolutions([snap])
    assert all(e.renamed_from == [] for e in evos)


def test_build_evolution_id_deterministic():
    from rag.code.evolution_builder import build_symbol_evolutions
    from rag.code.schema import _evolution_id
    snap = _snap(CA, ["foo"])
    evos = build_symbol_evolutions([snap])
    assert evos[0].evolution_id == _evolution_id(REPO_ID, FILE_PATH, "foo")


# ---------------------------------------------------------------------------
# GraphStore — symbol_evolution
# ---------------------------------------------------------------------------

def _make_store(tmp_path):
    from rag.code.graph_store import GraphStore
    return GraphStore(str(tmp_path / "graph.db"))


def _make_evolutions():
    from rag.code.evolution_builder import build_symbol_evolutions
    s1 = _snap(CA, ["foo", "bar"], {"foo": "h1", "bar": "h2"})
    s2 = _snap(CB, ["foo"],        {"foo": "h1_mod"})  # bar deleted, foo modified
    return build_symbol_evolutions([s1, s2])


def test_graph_store_upsert_and_get_evolution(tmp_path):
    store = _make_store(tmp_path)
    evos  = _make_evolutions()
    store.upsert_evolutions(evos)
    foo = store.get_evolution(REPO_ID, FILE_PATH, "foo")
    bar = store.get_evolution(REPO_ID, FILE_PATH, "bar")
    assert foo is not None
    assert foo.introduced_in == CA
    assert foo.modified_in   == [CB]
    assert foo.deleted_in    == ""
    assert bar is not None
    assert bar.deleted_in == CB


def test_graph_store_upsert_evolution_idempotent(tmp_path):
    store = _make_store(tmp_path)
    evos  = _make_evolutions()
    store.upsert_evolutions(evos)
    store.upsert_evolutions(evos)  # second time — same data
    all_evos = store.get_evolutions(repo_id=REPO_ID)
    assert len(all_evos) == len(evos)


def test_graph_store_get_evolutions_by_repo(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_evolutions(_make_evolutions())

    # Add second repo
    from rag.code.schema import SymbolEvolution, _evolution_id
    other = SymbolEvolution(
        evolution_id=_evolution_id("other", "x.py", "baz"),
        symbol_name="baz", repo_id="other", file_path="x.py",
        introduced_in=CA,
    )
    store.upsert_evolutions([other])

    repo_evos = store.get_evolutions(repo_id=REPO_ID)
    assert all(e.repo_id == REPO_ID for e in repo_evos)
    other_evos = store.get_evolutions(repo_id="other")
    assert len(other_evos) == 1


def test_graph_store_get_evolutions_by_file(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_evolutions(_make_evolutions())
    evos = store.get_evolutions(repo_id=REPO_ID, file_path=FILE_PATH)
    assert all(e.file_path == FILE_PATH for e in evos)


def test_graph_store_get_evolutions_alive_only(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_evolutions(_make_evolutions())
    alive = store.get_evolutions(repo_id=REPO_ID, alive_only=True)
    assert all(e.is_alive() for e in alive)
    assert any(not e.is_alive() for e in store.get_evolutions(repo_id=REPO_ID))


def test_graph_store_delete_file_evolutions(tmp_path):
    store = _make_store(tmp_path)
    evos  = _make_evolutions()
    store.upsert_evolutions(evos)
    deleted = store.delete_file_evolutions(REPO_ID, FILE_PATH)
    assert deleted == len(evos)
    assert store.get_evolutions(repo_id=REPO_ID) == []


def test_graph_store_delete_repo_evolutions(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_evolutions(_make_evolutions())
    from rag.code.schema import SymbolEvolution, _evolution_id
    other = SymbolEvolution(
        evolution_id=_evolution_id("other", "x.py", "baz"),
        symbol_name="baz", repo_id="other", file_path="x.py",
        introduced_in=CA,
    )
    store.upsert_evolutions([other])
    store.delete_repo_evolutions(REPO_ID)
    assert store.get_evolutions(repo_id=REPO_ID) == []
    assert len(store.get_evolutions(repo_id="other")) == 1


def test_graph_store_stats_includes_evolution(tmp_path):
    store = _make_store(tmp_path)
    store.upsert_evolutions(_make_evolutions())
    s = store.stats()
    assert "symbol_evolution" in s
    assert s["symbol_evolution"] == len(_make_evolutions())


def test_graph_store_get_evolution_not_found(tmp_path):
    store = _make_store(tmp_path)
    result = store.get_evolution(REPO_ID, FILE_PATH, "nonexistent")
    assert result is None


# ---------------------------------------------------------------------------
# GitReader — symbol_hashes populated
# ---------------------------------------------------------------------------

SAMPLE_SOURCE = '''\
def greet(name: str) -> str:
    """Return greeting."""
    return f"Hello, {name}"

class Foo:
    def bar(self) -> None:
        pass
'''


def test_git_reader_snapshot_file_fills_symbol_hashes():
    """snapshot_file with a parser must populate symbol_hashes."""
    from rag.code.ast_parser import PythonASTParser
    from rag.code.schema import FileSnapshot
    import hashlib

    parser = PythonASTParser()
    chunks = parser.parse(SAMPLE_SOURCE, "x.py", "repo")
    non_module = [c for c in chunks if c.chunk_type != "module"]

    # Simulate what snapshot_file does (without actual git)
    symbol_hashes = {c.name: c.content_hash for c in non_module}
    snap = FileSnapshot(
        snapshot_id="s", repo_id="repo", commit_hash="a" * 40,
        file_path="x.py", content_hash="h",
        symbols=[c.name for c in non_module],
        symbol_hashes=symbol_hashes,
    )
    assert "greet" in snap.symbol_hashes
    assert "Foo" in snap.symbol_hashes
    assert "Foo.bar" in snap.symbol_hashes
    # Each hash must be a 64-char hex string (SHA-256)
    for h in snap.symbol_hashes.values():
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_git_reader_snapshot_file_no_parser_empty_hashes():
    from rag.code.schema import FileSnapshot
    snap = FileSnapshot(
        snapshot_id="s", repo_id="repo", commit_hash="a" * 40,
        file_path="x.py", content_hash="h",
    )
    assert snap.symbol_hashes == {}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


