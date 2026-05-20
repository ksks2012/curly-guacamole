"""
GCR1 — Repository Intelligence Foundation: domain schema.

RepoFile       — metadata for a single file inside a repository.
RepoManifest   — snapshot of an entire repository scan.
ManifestDiff   — change set between two consecutive manifests.

Design notes
------------
- All fields are Chroma-safe scalars (str / int / float / bool) so
  RepoFile.to_meta() can be attached directly to a Chroma Document.
- content_hash (SHA-256) + mtime together enable fast incremental indexing:
    • hash changed   → content was modified, must re-embed
    • mtime changed but hash same → metadata-only edit (e.g. chmod), skip
    • absent from new manifest → file was deleted
- is_test / is_generated are heuristic flags derived from path patterns;
  callers may override them after construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# RepoFile
# ---------------------------------------------------------------------------

@dataclass
class RepoFile:
    """Metadata for one file in a repository.

    Attributes
    ----------
    repo_id       : Logical identifier for the repository (e.g. "my-project").
    branch        : Git branch name at scan time ("" if not a git repo).
    file_path     : POSIX path relative to the repository root.
    language      : Programming language inferred from file extension.
                    "" when the extension is unknown / not a source file.
    size          : File size in bytes.
    is_test       : True when the file is likely a test file (heuristic).
    is_generated  : True when the file is machine-generated (heuristic).
    content_hash  : SHA-256 hex digest of the file content.
    mtime         : ISO-8601 UTC last-modified timestamp.
    """

    repo_id:       str
    branch:        str
    file_path:     str    # relative POSIX path from repo root
    language:      str
    size:          int
    is_test:       bool
    is_generated:  bool
    content_hash:  str
    mtime:         str

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_meta(self) -> dict:
        """Return a flat dict of Chroma-safe scalar values."""
        return {
            "repo_id":      self.repo_id,
            "branch":       self.branch,
            "file_path":    self.file_path,
            "language":     self.language,
            "size":         self.size,
            "is_test":      self.is_test,
            "is_generated": self.is_generated,
            "content_hash": self.content_hash,
            "mtime":        self.mtime,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RepoFile":
        return cls(**d)


# ---------------------------------------------------------------------------
# RepoManifest
# ---------------------------------------------------------------------------

@dataclass
class RepoManifest:
    """Snapshot of a full repository scan.

    Attributes
    ----------
    repo_id    : Logical identifier for the repository.
    repo_root  : Absolute path to the repository root on disk.
    branch     : Git branch name at scan time.
    scanned_at : ISO-8601 UTC timestamp of when the scan was performed.
    files      : Mapping of relative POSIX path → RepoFile.
    """

    repo_id:    str
    repo_root:  str
    branch:     str
    scanned_at: str = field(default_factory=lambda: _now_iso())
    files:      dict[str, RepoFile] = field(default_factory=dict)

    # ── Convenience accessors ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[RepoFile]:
        return iter(self.files.values())

    def by_language(self, language: str) -> list[RepoFile]:
        """All files written in *language* (case-insensitive)."""
        lang = language.lower()
        return [f for f in self if f.language.lower() == lang]

    def source_files(self) -> list[RepoFile]:
        """Non-test, non-generated files that have a known language."""
        return [
            f for f in self
            if f.language and not f.is_test and not f.is_generated
        ]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise to a JSON file at *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "repo_id":    self.repo_id,
            "repo_root":  self.repo_root,
            "branch":     self.branch,
            "scanned_at": self.scanned_at,
            "files":      {k: v.to_dict() for k, v in self.files.items()},
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RepoManifest":
        """Deserialise a manifest previously saved by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        files = {k: RepoFile.from_dict(v) for k, v in data.get("files", {}).items()}
        return cls(
            repo_id=data["repo_id"],
            repo_root=data["repo_root"],
            branch=data.get("branch", ""),
            scanned_at=data.get("scanned_at", ""),
            files=files,
        )


# ---------------------------------------------------------------------------
# ManifestDiff
# ---------------------------------------------------------------------------

@dataclass
class ManifestDiff:
    """Change set between two consecutive RepoManifest snapshots.

    Attributes
    ----------
    added    : Files present in *new* but absent in *old*.
    modified : Files present in both whose content_hash changed.
    deleted  : Files present in *old* but absent in *new*.
    """

    added:    list[RepoFile] = field(default_factory=list)
    modified: list[RepoFile] = field(default_factory=list)
    deleted:  list[RepoFile] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.added or self.modified or self.deleted)

    def summary(self) -> str:
        return (
            f"+{len(self.added)} added  "
            f"~{len(self.modified)} modified  "
            f"-{len(self.deleted)} deleted"
        )


# ---------------------------------------------------------------------------
# Symbol  (GCR1.3 — Symbol Registry)
# ---------------------------------------------------------------------------

#: All valid symbol type strings.
SYMBOL_TYPES = frozenset({
    "class", "function", "method",
    "interface", "struct", "enum",
    "module", "namespace",
})


@dataclass
class Symbol:
    """A named code symbol extracted from a source file.

    Each Symbol corresponds to a meaningful declaration boundary and is
    stored independently of the raw code chunks so that the Symbol Registry
    can serve as a fast, queryable index over the codebase.

    Attributes
    ----------
    symbol_id     : Deterministic ID: ``"{repo_id}::{file_path}::{symbol_type}::{symbol_name}"``.
    symbol_name   : Fully-qualified symbol name (e.g. ``"MyClass.my_method"``).
    symbol_type   : One of the values in ``SYMBOL_TYPES``.
    repo_id       : Logical repository identifier.
    file_path     : Relative POSIX path from the repository root.
    start_line    : 1-based first line of the symbol declaration.
    end_line      : 1-based last line of the symbol declaration.
    parent_symbol : symbol_id of the enclosing symbol, or ``""`` if top-level.
    visibility    : ``"public"`` / ``"private"`` / ``"dunder"`` (Python convention).
                    ``"public"`` for non-Python languages when not determinable.
    """

    symbol_id:     str
    symbol_name:   str
    symbol_type:   str    # value must be in SYMBOL_TYPES
    repo_id:       str
    file_path:     str
    start_line:    int
    end_line:      int
    parent_symbol: str    # symbol_id of parent, or ""
    visibility:    str    # "public" | "private" | "dunder"

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Symbol":
        return cls(**d)


# ---------------------------------------------------------------------------
# CodeChunk  (GCR1.2 — AST-aware Parsing)
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A syntax-aware chunk extracted from a source file by AST parsing.

    Each chunk corresponds to a meaningful code boundary:
    module / class / function / method.

    Attributes
    ----------
    chunk_id     : Deterministic ID: ``"{repo_id}::{file_path}::{chunk_type}::{name}"``.
    repo_id      : Logical repository identifier.
    file_path    : Relative POSIX path from the repo root.
    language     : Programming language (e.g. "python").
    chunk_type   : One of ``"module"``, ``"class"``, ``"function"``, ``"method"``.
    name         : Fully-qualified symbol name (e.g. ``"MyClass.my_method"``).
    start_line   : 1-based first line of the chunk in the source file.
    end_line     : 1-based last line of the chunk in the source file.
    code         : Raw source text of the chunk (may be indented).
    docstring    : Extracted docstring, or ``None`` if absent.
    parent_name  : Name of the enclosing class or function, or ``None``.
    content_hash : SHA-256 hex digest of *code* for incremental indexing.
    """

    chunk_id:     str
    repo_id:      str
    file_path:    str
    language:     str
    chunk_type:   str    # "module" | "class" | "function" | "method"
    name:         str
    start_line:   int
    end_line:     int
    code:         str
    docstring:    str | None
    parent_name:  str | None
    content_hash: str

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_meta(self) -> dict:
        """Return a flat dict of Chroma-safe scalar values (no None)."""
        return {
            "chunk_id":     self.chunk_id,
            "repo_id":      self.repo_id,
            "file_path":    self.file_path,
            "language":     self.language,
            "chunk_type":   self.chunk_type,
            "name":         self.name,
            "start_line":   self.start_line,
            "end_line":     self.end_line,
            "parent_name":  self.parent_name or "",
            "content_hash": self.content_hash,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CodeChunk":
        return cls(**d)


# ---------------------------------------------------------------------------
# CommitInfo  (GCR1.5 — Git-aware Snapshot System)
# ---------------------------------------------------------------------------

@dataclass
class CommitInfo:
    """Lightweight record of one git commit.

    Attributes
    ----------
    commit_hash    : Full 40-char SHA-1 hex digest.
    author         : Author name (from git log ``%an``).
    date           : ISO-8601 UTC commit timestamp.
    message        : Full commit message.
    files_changed  : POSIX-relative paths of files touched by this commit.
    """

    commit_hash:   str
    author:        str
    date:          str    # ISO-8601 UTC
    message:       str
    files_changed: list[str] = field(default_factory=list)

    @property
    def short_hash(self) -> str:
        return self.commit_hash[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CommitInfo":
        return cls(**d)


# ---------------------------------------------------------------------------
# FileSnapshot  (GCR1.5 — Git-aware Snapshot System)
# ---------------------------------------------------------------------------

@dataclass
class FileSnapshot:
    """State of one file at a specific git commit — the unit of temporal knowledge.

    A FileSnapshot captures what a file *was* at a point in time, together
    with the symbols visible at that moment.  A series of FileSnapshots for
    the same ``file_path`` forms a timeline that lets the system reason about:

    - which symbols were added / removed between commits
    - which files change most frequently (instability signal)
    - how a module's public API evolved

    Attributes
    ----------
    snapshot_id  : Deterministic ID: ``"{repo_id}::{commit_hash[:12]}::{file_path}"``.
    repo_id      : Logical repository identifier.
    commit_hash  : Full 40-char SHA-1 git commit hash.
    file_path    : POSIX path relative to the repository root.
    content_hash : SHA-256 hex digest of the file content at this commit.
    symbols      : Fully-qualified symbol names extracted from this version
                   of the file (e.g. ``["MyClass", "MyClass.my_method"]``).
                   Stored as ``list[str]`` in memory; serialised to CSV in
                   ``to_meta()`` to remain Chroma-compatible.
    """

    snapshot_id:  str
    repo_id:      str
    commit_hash:  str
    file_path:    str
    content_hash: str
    symbols:      list[str] = field(default_factory=list)

    @property
    def short_hash(self) -> str:
        return self.commit_hash[:12]

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_meta(self) -> dict:
        """Return a flat dict of Chroma-safe scalar values.

        ``symbols`` is serialised as a comma-separated string so Chroma can
        store it as a scalar metadata field.
        """
        return {
            "snapshot_id":  self.snapshot_id,
            "repo_id":      self.repo_id,
            "commit_hash":  self.commit_hash,
            "file_path":    self.file_path,
            "content_hash": self.content_hash,
            "symbols":      ",".join(self.symbols),
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FileSnapshot":
        d = dict(d)
        if isinstance(d.get("symbols"), str):
            # Backwards-compat: CSV → list
            d["symbols"] = [s for s in d["symbols"].split(",") if s]
        return cls(**d)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
