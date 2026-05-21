"""Smoke test for GCR1.5 — Git-aware Snapshot System (GitReader + SnapshotStore)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.code.ast_parser import PythonASTParser
from rag.code.git_reader import GitReader
from rag.code.schema import CommitInfo, FileSnapshot
from rag.code.snapshot_store import SnapshotStore, SymbolDiff


REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ID   = "langchain-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reader() -> GitReader:
    return GitReader(REPO_ROOT)


def _parser() -> PythonASTParser:
    return PythonASTParser()


# ---------------------------------------------------------------------------
# Tests: GitReader
# ---------------------------------------------------------------------------

def test_head_commit() -> str:
    reader = _reader()
    head = reader.head_commit()
    assert len(head) == 40,   f"head_commit should be 40 chars, got {len(head)!r}: {head!r}"
    assert head.isalnum(),    f"head_commit should be hex: {head!r}"
    print(f"head_commit: {head[:12]}")
    return head


def test_current_branch() -> None:
    reader  = _reader()
    branch  = reader.current_branch()
    assert isinstance(branch, str) and branch, f"current_branch should be non-empty: {branch!r}"
    print(f"current_branch: {branch!r}")


def test_commits() -> list[CommitInfo]:
    reader  = _reader()
    commits = reader.commits(max_count=10)
    assert len(commits) > 0,  "commits() should return at least one commit"
    assert len(commits) <= 10

    for ci in commits:
        assert len(ci.commit_hash) == 40, f"bad commit_hash: {ci.commit_hash!r}"
        assert ci.author,                 f"commit author is empty: {ci.commit_hash[:8]}"
        assert ci.date,                   f"commit date is empty: {ci.commit_hash[:8]}"

    print(f"commits (max 10): {len(commits)}")
    for ci in commits[:3]:
        print(f"  {ci.short_hash}  {ci.date[:10]}  {ci.author[:20]:<20}  {ci.message.splitlines()[0][:60]}")
    return commits


def test_files_changed(commits: list[CommitInfo]) -> None:
    reader = _reader()
    # Check a handful of commits
    for ci in commits[:5]:
        files = reader.files_changed_at(ci.commit_hash)
        assert isinstance(files, list)
        # files_changed on CommitInfo should match
        assert files == ci.files_changed, (
            f"files_changed mismatch for {ci.short_hash}: "
            f"direct={files}  stored={ci.files_changed}"
        )
    print(f"files_changed_at: consistent across first {min(5, len(commits))} commits")


def test_file_content_at(head: str) -> None:
    reader  = _reader()
    # Use rag/code/schema.py — committed early in GCR1.1
    content = reader.file_content_at(head, "rag/code/schema.py")
    assert content is not None, "file_content_at returned None for rag/code/schema.py at HEAD"
    assert "FileSnapshot" in content or "RepoFile" in content, \
        "expected schema definitions in content"
    print(f"file_content_at HEAD:rag/code/schema.py  → {len(content)} chars")


def test_snapshot_file(head: str) -> FileSnapshot:
    reader = _reader()
    parser = _parser()
    # Use rag/code/schema.py — guaranteed committed at HEAD
    snap   = reader.snapshot_file(REPO_ID, head, "rag/code/schema.py", parser=parser)

    assert snap is not None,                         "snapshot_file returned None"
    assert snap.repo_id == REPO_ID
    assert snap.commit_hash == head
    assert snap.file_path == "rag/code/schema.py"
    assert len(snap.content_hash) == 64,             "content_hash should be 64-char SHA-256"
    assert len(snap.symbols) > 0,                    f"no symbols extracted: {snap.symbols}"
    assert snap.snapshot_id == f"{REPO_ID}::{head[:12]}::rag/code/schema.py"

    print(f"snapshot_file: {snap.snapshot_id}  symbols={snap.symbols[:5]}")
    return snap


def test_snapshot_nonexistent_file(head: str) -> None:
    reader = _reader()
    snap   = reader.snapshot_file(REPO_ID, head, "does/not/exist.py")
    assert snap is None, f"expected None for non-existent file, got {snap}"
    print("snapshot_file (nonexistent) → None  OK")


def test_snapshot_commit(commits: list[CommitInfo]) -> list[FileSnapshot]:
    reader  = _reader()
    parser  = _parser()
    # Find a recent commit that touched at least one Python file
    snaps: list[FileSnapshot] = []
    for ci in commits:
        snaps = reader.snapshot_commit(REPO_ID, ci.commit_hash, parser=parser)
        if snaps:
            print(f"snapshot_commit {ci.short_hash}: {len(snaps)} Python file snapshots")
            for s in snaps[:3]:
                print(f"  {s.file_path}  symbols={len(s.symbols)}")
            break
    return snaps


# ---------------------------------------------------------------------------
# Tests: SnapshotStore
# ---------------------------------------------------------------------------

def test_store_add_query(snap: FileSnapshot) -> None:
    store = SnapshotStore(repo_id=REPO_ID)
    store.add(snap)

    assert len(store) == 1
    assert store.get(snap.snapshot_id) == snap

    history = store.file_history(snap.file_path)
    assert len(history) == 1 and history[0] == snap

    by_commit = store.by_commit(snap.commit_hash)
    assert len(by_commit) == 1

    assert snap.file_path in store.tracked_files()
    print(f"store add/query: {store.summary()}")


def test_symbol_diff() -> None:
    def _make(sid: str, symbols: list[str]) -> FileSnapshot:
        return FileSnapshot(
            snapshot_id=sid, repo_id="r", commit_hash="a" * 40,
            file_path="f.py", content_hash="h", symbols=symbols,
        )

    old  = _make("r::aaa::f.py", ["Foo", "Foo.bar", "Foo.baz"])
    new  = _make("r::bbb::f.py", ["Foo", "Foo.bar", "Foo.qux"])

    store = SnapshotStore()
    diff  = store.symbol_diff(old, new)

    assert diff.added   == ["Foo.qux"],  f"added:   {diff.added}"
    assert diff.removed == ["Foo.baz"],  f"removed: {diff.removed}"
    assert not diff.is_empty()

    same_diff = store.symbol_diff(old, old)
    assert same_diff.is_empty()

    print(f"symbol_diff: {diff.summary()}")


def test_churn(commits: list[CommitInfo]) -> None:
    reader = _reader()
    parser = _parser()
    store  = SnapshotStore(repo_id=REPO_ID)

    for ci in commits[:5]:
        store.add_many(reader.snapshot_commit(REPO_ID, ci.commit_hash, parser=parser))

    churn = store.churn()
    print(f"churn (top 3 of {len(churn)} tracked files):")
    for fp, cnt in churn[:3]:
        print(f"  {cnt:3d}×  {fp}")

    if churn:
        assert churn[0][1] >= churn[-1][1], "churn should be sorted descending"


def test_round_trip(snap: FileSnapshot) -> None:
    store = SnapshotStore(repo_id=REPO_ID)
    store.add(snap)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "snapshots.json"
        store.save(p)
        loaded = SnapshotStore.load(p)

    assert len(loaded) == len(store), \
        f"round-trip count mismatch: {len(loaded)} vs {len(store)}"
    reloaded = loaded.get(snap.snapshot_id)
    assert reloaded == snap, f"round-trip mismatch: {reloaded}"
    print("round-trip save/load: OK")


def test_file_history_ordering() -> None:
    """Snapshots should appear in insertion order (oldest first when added that way)."""
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
    assert [h.content_hash for h in history] == ["0", "1", "2"], \
        f"unexpected order: {[h.content_hash for h in history]}"
    print("file_history ordering: OK")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    head    = test_head_commit()
    test_current_branch()
    commits = test_commits()
    test_files_changed(commits)
    test_file_content_at(head)
    snap    = test_snapshot_file(head)
    test_snapshot_nonexistent_file(head)
    test_snapshot_commit(commits)

    test_store_add_query(snap)
    test_symbol_diff()
    test_churn(commits)
    test_round_trip(snap)
    test_file_history_ordering()

    print("\nPASS")


if __name__ == "__main__":
    main()
