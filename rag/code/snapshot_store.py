"""
GCR1.5 — Git-aware Snapshot System: SnapshotStore.

SnapshotStore is an in-memory store for FileSnapshot objects.  It enables:

- Building a temporal history of a file: which symbols existed at each commit.
- Computing symbol diffs between two snapshots of the same file.
- Detecting high-churn files (potential instability signals for GCR4).
- Persistence to / loading from a JSON file for offline analysis.

Typical usage
-------------
>>> from rag.code.git_reader import GitReader
>>> from rag.code.ast_parser import PythonASTParser
>>> from rag.code.snapshot_store import SnapshotStore
>>>
>>> reader = GitReader("/path/to/repo")
>>> parser = PythonASTParser()
>>> store  = SnapshotStore()
>>>
>>> for ci in reader.commits(max_count=50):
...     snaps = reader.snapshot_commit("my-repo", ci.commit_hash, parser=parser)
...     store.add_many(snaps)
>>>
>>> history = store.file_history("rag/engine.py")
>>> for snap in history:
...     print(snap.short_hash, len(snap.symbols), "symbols")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rag.code.schema import FileSnapshot


# ---------------------------------------------------------------------------
# SymbolDiff
# ---------------------------------------------------------------------------

class SymbolDiff:
    """Symbol-level change between two FileSnapshots of the same file.

    Attributes
    ----------
    added   : Symbol names present in *new* but absent in *old*.
    removed : Symbol names present in *old* but absent in *new*.
    """

    def __init__(self, added: list[str], removed: list[str]) -> None:
        self.added   = added
        self.removed = removed

    def is_empty(self) -> bool:
        return not (self.added or self.removed)

    def summary(self) -> str:
        return f"+{len(self.added)} added  -{len(self.removed)} removed"

    def __repr__(self) -> str:
        return f"SymbolDiff({self.summary()})"


# ---------------------------------------------------------------------------
# SnapshotStore
# ---------------------------------------------------------------------------

class SnapshotStore:
    """In-memory store for FileSnapshot objects.

    Primary indexes:
    - ``snapshot_id`` → FileSnapshot   (unique key)
    - ``file_path``   → [snapshot_id]  (file timeline, in insertion order)
    - ``commit_hash`` → [snapshot_id]  (all files touched by a commit)
    """

    def __init__(self, repo_id: str = "") -> None:
        self._repo_id   = repo_id
        self._by_id:     dict[str, FileSnapshot]    = {}
        self._by_file:   dict[str, list[str]]        = {}  # file_path → [snapshot_id]
        self._by_commit: dict[str, list[str]]        = {}  # commit_hash → [snapshot_id]

    # ── Mutators ──────────────────────────────────────────────────────────

    def add(self, snap: FileSnapshot) -> None:
        """Register a FileSnapshot.  Duplicate snapshot_ids are overwritten."""
        sid = snap.snapshot_id
        self._by_id[sid] = snap

        self._by_file.setdefault(snap.file_path, [])
        if sid not in self._by_file[snap.file_path]:
            self._by_file[snap.file_path].append(sid)

        self._by_commit.setdefault(snap.commit_hash, [])
        if sid not in self._by_commit[snap.commit_hash]:
            self._by_commit[snap.commit_hash].append(sid)

    def add_many(self, snaps: Iterable[FileSnapshot]) -> None:
        for s in snaps:
            self.add(s)

    # ── Queries ───────────────────────────────────────────────────────────

    def get(self, snapshot_id: str) -> FileSnapshot | None:
        return self._by_id.get(snapshot_id)

    def file_history(self, file_path: str) -> list[FileSnapshot]:
        """All snapshots for *file_path* in insertion order (newest-last).

        Each snapshot corresponds to a commit where the file was changed.
        """
        ids = self._by_file.get(file_path, [])
        return [self._by_id[i] for i in ids]

    def by_commit(self, commit_hash: str) -> list[FileSnapshot]:
        """All file snapshots captured for *commit_hash*."""
        ids = self._by_commit.get(commit_hash, [])
        return [self._by_id[i] for i in ids]

    def tracked_files(self) -> list[str]:
        """All file paths for which at least one snapshot exists."""
        return sorted(self._by_file.keys())

    def commit_hashes(self) -> list[str]:
        """All commit hashes for which at least one snapshot exists."""
        return list(self._by_commit.keys())

    # ── Analysis helpers ──────────────────────────────────────────────────

    def symbol_diff(self, old: FileSnapshot, new: FileSnapshot) -> SymbolDiff:
        """Compute which symbols were added or removed between two snapshots.

        Parameters
        ----------
        old : Earlier snapshot (e.g. parent commit).
        new : Later snapshot (e.g. current commit).
        """
        old_set = set(old.symbols)
        new_set = set(new.symbols)
        return SymbolDiff(
            added=sorted(new_set - old_set),
            removed=sorted(old_set - new_set),
        )

    def churn(self) -> list[tuple[str, int]]:
        """Return (file_path, change_count) sorted by change_count descending.

        High churn = many commits touched this file = potential instability.
        """
        counts = [(fp, len(ids)) for fp, ids in self._by_file.items()]
        return sorted(counts, key=lambda x: -x[1])

    # ── Statistics ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def summary(self) -> str:
        return (
            f"snapshots={len(self)}  "
            f"files={len(self._by_file)}  "
            f"commits={len(self._by_commit)}"
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "repo_id":   self._repo_id,
            "snapshots": [s.to_dict() for s in self._by_id.values()],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SnapshotStore":
        """Deserialise a store previously saved by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(repo_id=data.get("repo_id", ""))
        for d in data.get("snapshots", []):
            store.add(FileSnapshot.from_dict(d))
        return store
