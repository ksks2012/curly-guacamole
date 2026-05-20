"""
GCR1.5 — Git-aware Snapshot System: GitReader.

GitReader wraps subprocess git commands to produce CommitInfo and
FileSnapshot objects from a local git repository, without any external
git library dependencies.

Responsibilities
----------------
- List commits (full repo or scoped to a single file path).
- Retrieve file content at a specific commit via ``git show``.
- Detect Python symbol names in that content (via PythonASTParser).
- Assemble FileSnapshot objects that record what a file *was* at a point
  in time, together with the symbols visible at that moment.

Why no external library?
------------------------
GitPython and pygit2 add heavyweight dependencies.  The subset of git
operations we need is small and stable; thin subprocess wrappers are
sufficient, testable, and have no install-time overhead.

Typical usage
-------------
>>> from rag.code.git_reader import GitReader
>>> from rag.code.ast_parser import PythonASTParser
>>>
>>> reader = GitReader("/path/to/repo")
>>> commits = reader.commits(max_count=20)
>>> for ci in commits:
...     snaps = reader.snapshot_commit("my-repo", ci.commit_hash)
...     for snap in snaps:
...         print(snap.snapshot_id, snap.symbols[:3])
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Optional

from utils.logger import AppLogger
from rag.code.schema import CommitInfo, FileSnapshot

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: Path) -> tuple[str, int]:
    """Run a git command and return (stdout, returncode).

    Never raises; returns ("", non-zero) on failure so callers can handle
    gracefully without try/except boilerplate.
    """
    result = subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout, result.returncode


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _make_snapshot_id(repo_id: str, commit_hash: str, file_path: str) -> str:
    return f"{repo_id}::{commit_hash[:12]}::{file_path}"


# ---------------------------------------------------------------------------
# GitReader
# ---------------------------------------------------------------------------

class GitReader:
    """Read git history from a local repository.

    Parameters
    ----------
    repo_path : Absolute path to the repository root (must contain ``.git/``).
    """

    def __init__(self, repo_path: str | Path) -> None:
        self._root = Path(repo_path).resolve()
        if not (self._root / ".git").exists():
            log.warning("GitReader: .git not found at %s — git operations may fail", self._root)

    # ── Commit listing ────────────────────────────────────────────────────

    def commits(
        self,
        file_path: Optional[str] = None,
        max_count: Optional[int] = None,
        since: Optional[str] = None,
        branch: str = "HEAD",
    ) -> list[CommitInfo]:
        """Return a list of CommitInfo objects from git log.

        Parameters
        ----------
        file_path : Restrict history to commits that touched this path
                    (relative POSIX path from repo root).  None = all commits.
        max_count : Cap number of commits returned.  None = no limit.
        since     : ISO-8601 date string passed to ``--since`` (e.g. "2024-01-01").
        branch    : Branch / ref to start from (default ``"HEAD"``).

        Returns an empty list if not a git repo or on error.
        """
        # Format: NUL-separated fields per commit, commits separated by RS (\x1e)
        _FMT = "%x1e%H%x00%an%x00%aI%x00%s%x00%b"
        args = ["log", f"--format={_FMT}", branch]
        if max_count:
            args += [f"--max-count={max_count}"]
        if since:
            args += [f"--since={since}"]
        if file_path:
            args += ["--follow", "--", file_path]

        stdout, rc = _run_git(args, self._root)
        if rc != 0:
            log.warning("GitReader.commits: git log failed (rc=%d)", rc)
            return []

        commits: list[CommitInfo] = []
        for block in stdout.strip().split("\x1e"):
            block = block.strip()
            if not block:
                continue
            parts = block.split("\x00")
            if len(parts) < 4:
                continue
            commit_hash = parts[0].strip()
            author      = parts[1].strip()
            date        = parts[2].strip()
            subject     = parts[3].strip()
            body        = parts[4].strip() if len(parts) > 4 else ""
            message     = (subject + ("\n\n" + body if body else "")).strip()

            # Determine which files were touched
            files_changed = self.files_changed_at(commit_hash)
            commits.append(CommitInfo(
                commit_hash=commit_hash,
                author=author,
                date=date,
                message=message,
                files_changed=files_changed,
            ))

        log.debug("GitReader.commits: %d commits found", len(commits))
        return commits

    # ── File content at commit ────────────────────────────────────────────

    def files_changed_at(self, commit_hash: str) -> list[str]:
        """Return a list of POSIX-relative file paths touched by *commit_hash*.

        Returns an empty list on error or for the initial commit.
        """
        stdout, rc = _run_git(
            ["diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash],
            self._root,
        )
        if rc != 0:
            return []
        return [line.strip() for line in stdout.splitlines() if line.strip()]

    def file_content_at(self, commit_hash: str, file_path: str) -> Optional[str]:
        """Return the content of *file_path* at *commit_hash*, or None.

        Uses ``git show {commit_hash}:{file_path}``.
        Returns None when the file did not exist at that commit.
        """
        stdout, rc = _run_git(
            ["show", f"{commit_hash}:{file_path}"],
            self._root,
        )
        return stdout if rc == 0 else None

    # ── Snapshot builders ─────────────────────────────────────────────────

    def snapshot_file(
        self,
        repo_id: str,
        commit_hash: str,
        file_path: str,
        parser=None,
    ) -> Optional[FileSnapshot]:
        """Build a FileSnapshot for one file at a given commit.

        Parameters
        ----------
        repo_id     : Logical repository identifier.
        commit_hash : Full git commit hash.
        file_path   : Relative POSIX path from the repo root.
        parser      : Optional ``PythonASTParser`` instance.  When supplied
                      and the file is Python, symbols are extracted.
                      Pass None to skip symbol extraction (faster).

        Returns None if the file did not exist at that commit.
        """
        content = self.file_content_at(commit_hash, file_path)
        if content is None:
            return None

        content_hash = _sha256(content)
        symbols: list[str] = []

        if parser is not None and file_path.endswith(".py"):
            try:
                chunks = parser.parse(content, file_path=file_path, repo_id=repo_id)
                # Collect non-module symbol names (module chunk name is "<module>")
                symbols = [
                    c.name for c in chunks
                    if c.chunk_type != "module"
                ]
            except Exception as exc:
                log.debug(
                    "GitReader.snapshot_file: parser error at %s@%s: %s",
                    file_path, commit_hash[:8], exc,
                )

        return FileSnapshot(
            snapshot_id=_make_snapshot_id(repo_id, commit_hash, file_path),
            repo_id=repo_id,
            commit_hash=commit_hash,
            file_path=file_path,
            content_hash=content_hash,
            symbols=symbols,
        )

    def snapshot_commit(
        self,
        repo_id: str,
        commit_hash: str,
        parser=None,
        language_filter: Optional[str] = "python",
    ) -> list[FileSnapshot]:
        """Build FileSnapshots for all files changed in *commit_hash*.

        Parameters
        ----------
        repo_id         : Logical repository identifier.
        commit_hash     : Full git commit hash.
        parser          : Optional PythonASTParser for symbol extraction.
        language_filter : Only process files matching this extension filter.
                          ``"python"`` → ``.py`` files only.
                          ``None`` → process all changed files.

        Returns a list of FileSnapshot objects (may be empty for merge commits
        or the initial commit on some git versions).
        """
        files = self.files_changed_at(commit_hash)
        if language_filter == "python":
            files = [f for f in files if f.endswith(".py")]

        snapshots: list[FileSnapshot] = []
        for fp in files:
            snap = self.snapshot_file(repo_id, commit_hash, fp, parser=parser)
            if snap is not None:
                snapshots.append(snap)

        return snapshots

    # ── Convenience ───────────────────────────────────────────────────────

    def current_branch(self) -> str:
        """Return the current branch name, or ``""`` on error."""
        stdout, rc = _run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            self._root,
        )
        return stdout.strip() if rc == 0 else ""

    def head_commit(self) -> str:
        """Return the full hash of HEAD, or ``""`` on error."""
        stdout, rc = _run_git(["rev-parse", "HEAD"], self._root)
        return stdout.strip() if rc == 0 else ""
