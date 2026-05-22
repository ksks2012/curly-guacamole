"""Smoke test for GCR1.5 — Git-aware Snapshot System (GitReader + SnapshotStore)."""

from pathlib import Path

import pytest

from rag.code.ast_parser import PythonASTParser
from rag.code.git_reader import GitReader
from rag.code.schema import CommitInfo, FileSnapshot
from rag.code.snapshot_store import SnapshotStore, SymbolDiff


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ID   = "langchain-test"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def git_state():
    reader = GitReader(REPO_ROOT)
    head    = reader.head_commit()
    commits = reader.commits(max_count=10)
    parser  = PythonASTParser()
    snap    = reader.snapshot_file(REPO_ID, head, "rag/code/schema.py", parser=parser)
    return {
        "reader":  reader,
        "head":    head,
        "commits": commits,
        "snap":    snap,
        "parser":  parser,
    }


# ---------------------------------------------------------------------------
# Tests: GitReader
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_head_commit(git_state):
    head = git_state["head"]
    assert len(head) == 40, f"head_commit should be 40 chars: {head!r}"
    assert head.isalnum(), f"head_commit should be hex: {head!r}"


@pytest.mark.integration
def test_current_branch(git_state):
    branch = git_state["reader"].current_branch()
    assert isinstance(branch, str) and branch, f"current_branch should be non-empty: {branch!r}"


@pytest.mark.integration
def test_commits_count(git_state):
    commits = git_state["commits"]
    assert len(commits) > 0, "commits() should return at least one commit"
    assert len(commits) <= 10


@pytest.mark.integration
def test_commits_fields(git_state):
    for ci in git_state["commits"]:
        assert len(ci.commit_hash) == 40, f"bad commit_hash: {ci.commit_hash!r}"
        assert ci.author, f"commit author is empty: {ci.commit_hash[:8]}"
        assert ci.date,   f"commit date is empty: {ci.commit_hash[:8]}"


@pytest.mark.integration
def test_files_changed(git_state):
    reader  = git_state["reader"]
    commits = git_state["commits"]
    for ci in commits[:5]:
        files = reader.files_changed_at(ci.commit_hash)
        assert isinstance(files, list)
        assert files == ci.files_changed, (
            f"files_changed mismatch for {ci.short_hash}: "
            f"direct={files}  stored={ci.files_changed}"
        )


@pytest.mark.integration
def test_file_content_at(git_state):
    content = git_state["reader"].file_content_at(git_state["head"], "rag/code/schema.py")
    assert content is not None, "file_content_at returned None for rag/code/schema.py"
    assert "FileSnapshot" in content or "RepoFile" in content


@pytest.mark.integration
def test_snapshot_file(git_state):
    snap = git_state["snap"]
    head = git_state["head"]
    assert snap is not None, "snapshot_file returned None"
    assert snap.repo_id == REPO_ID
    assert snap.commit_hash == head
    assert snap.file_path == "rag/code/schema.py"
    assert len(snap.content_hash) == 64
    assert len(snap.symbols) > 0, f"no symbols extracted: {snap.symbols}"
    assert snap.snapshot_id == f"{REPO_ID}::{head[:12]}::rag/code/schema.py"


@pytest.mark.integration
def test_snapshot_nonexistent_file(git_state):
    snap = git_state["reader"].snapshot_file(REPO_ID, git_state["head"], "does/not/exist.py")
    assert snap is None


@pytest.mark.integration
def test_snapshot_commit(git_state):
    reader  = git_state["reader"]
    parser  = git_state["parser"]
    commits = git_state["commits"]
    found = False
    for ci in commits:
        snaps = reader.snapshot_commit(REPO_ID, ci.commit_hash, parser=parser)
        if snaps:
            found = True
            break
    assert found, "no commit returned any Python file snapshots"


# ---------------------------------------------------------------------------
# Tests: SnapshotStore
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_store_add_query(git_state):
    snap  = git_state["snap"]
    store = SnapshotStore(repo_id=REPO_ID)
    store.add(snap)

    assert len(store) == 1
    assert store.get(snap.snapshot_id) == snap
    history = store.file_history(snap.file_path)
    assert len(history) == 1 and history[0] == snap
    assert len(store.by_commit(snap.commit_hash)) == 1
    assert snap.file_path in store.tracked_files()


def test_symbol_diff():
    def _make(sid, symbols):
        return FileSnapshot(
            snapshot_id=sid, repo_id="r", commit_hash="a" * 40,
            file_path="f.py", content_hash="h", symbols=symbols,
        )

    old  = _make("r::aaa::f.py", ["Foo", "Foo.bar", "Foo.baz"])
    new  = _make("r::bbb::f.py", ["Foo", "Foo.bar", "Foo.qux"])
    store = SnapshotStore()
    diff  = store.symbol_diff(old, new)
    assert diff.added   == ["Foo.qux"]
    assert diff.removed == ["Foo.baz"]
    assert not diff.is_empty()
    assert store.symbol_diff(old, old).is_empty()


@pytest.mark.integration
def test_churn(git_state):
    reader  = git_state["reader"]
    parser  = git_state["parser"]
    commits = git_state["commits"]
    store   = SnapshotStore(repo_id=REPO_ID)
    for ci in commits[:5]:
        store.add_many(reader.snapshot_commit(REPO_ID, ci.commit_hash, parser=parser))
    churn = store.churn()
    if churn:
        assert churn[0][1] >= churn[-1][1], "churn should be sorted descending"


@pytest.mark.integration
def test_round_trip(git_state, tmp_path):
    snap  = git_state["snap"]
    store = SnapshotStore(repo_id=REPO_ID)
    store.add(snap)
    p = tmp_path / "snapshots.json"
    store.save(p)
    loaded = SnapshotStore.load(p)
    assert len(loaded) == len(store)
    assert loaded.get(snap.snapshot_id) == snap


def test_file_history_ordering():
    store = SnapshotStore()
    snaps = [
        FileSnapshot(
            snapshot_id=f"r::{'a' * 12}::{i}::f.py",
            repo_id="r", commit_hash="a" * 40,
            file_path="f.py", content_hash=str(i), symbols=[],
        )
        for i in range(3)
    ]
    for s in snaps:
        store.add(s)
    history = store.file_history("f.py")
    assert [h.content_hash for h in history] == ["0", "1", "2"]
