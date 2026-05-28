"""
GCR1.1 — RepoScanner: Repository Intelligence Foundation.

RepoScanner walks a repository directory tree and produces a RepoManifest —
a structured, hash-stamped snapshot of every source file.  The manifest is the
single source of truth for all downstream stages:

    GCR1.1  RepoScanner         (this file)
    GCR2    Symbol / Git graph  → reads RepoManifest.source_files()
    GCR3    Knowledge extraction → driven by ManifestDiff.modified
    GCR4    Temporal memory      → consumes successive manifests

Usage
-----
    from rag.code.scanner import RepoScanner

    scanner  = RepoScanner()
    manifest = scanner.scan("/path/to/repo", repo_id="my-project")
    manifest.save("data/repo_manifest.json")

    # Incremental update
    old      = RepoManifest.load("data/repo_manifest.json")
    new      = scanner.scan("/path/to/repo", repo_id="my-project")
    diff     = scanner.diff(old, new)
    print(diff.summary())
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Collection

try:
    import pathspec as _pathspec
    _HAS_PATHSPEC = True
except ImportError:  # pragma: no cover
    _pathspec = None  # type: ignore[assignment]
    _HAS_PATHSPEC = False

from utils.logger import AppLogger
from rag.code.schema import ManifestDiff, RepoFile, RepoManifest

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# Language detection table
# ---------------------------------------------------------------------------

_EXT_LANGUAGE: dict[str, str] = {
    # Python ecosystem
    ".py":   "Python",
    ".pyi":  "Python",
    ".pyx":  "Python",
    ".ipynb": "Python",
    # JavaScript / TypeScript
    ".js":   "JavaScript",
    ".mjs":  "JavaScript",
    ".cjs":  "JavaScript",
    ".jsx":  "JavaScript",
    ".ts":   "TypeScript",
    ".tsx":  "TypeScript",
    # Web
    ".html": "HTML",
    ".htm":  "HTML",
    ".css":  "CSS",
    ".scss": "CSS",
    ".sass": "CSS",
    # Data / config
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml":  "YAML",
    ".toml": "TOML",
    ".ini":  "INI",
    ".env":  "ENV",
    # Documentation
    ".md":   "Markdown",
    ".mdx":  "Markdown",
    ".rst":  "reStructuredText",
    ".txt":  "Text",
    # Systems
    ".c":    "C",
    ".h":    "C",
    ".cpp":  "C++",
    ".cc":   "C++",
    ".cxx":  "C++",
    ".hpp":  "C++",
    ".rs":   "Rust",
    ".go":   "Go",
    ".java": "Java",
    ".kt":   "Kotlin",
    ".scala":"Scala",
    ".rb":   "Ruby",
    ".php":  "PHP",
    ".swift":"Swift",
    ".cs":   "C#",
    # Shell
    ".sh":   "Shell",
    ".bash": "Shell",
    ".zsh":  "Shell",
    ".fish": "Shell",
    ".ps1":  "PowerShell",
    # SQL
    ".sql":  "SQL",
    # Build / infra
    ".tf":   "Terraform",
    ".hcl":  "HCL",
    ".dockerfile": "Dockerfile",
}

# Filenames without extensions that map to a language
_NAME_LANGUAGE: dict[str, str] = {
    "dockerfile":   "Dockerfile",
    "makefile":     "Makefile",
    "rakefile":     "Ruby",
    "gemfile":      "Ruby",
    "podfile":      "Ruby",
    "jenkinsfile":  "Groovy",
}


# ---------------------------------------------------------------------------
# Heuristic matchers
# ---------------------------------------------------------------------------

# Directories that are always excluded from the scan
_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", ".npm", ".yarn",
    "venv", ".venv", "env", ".env",
    "build", "dist", "target",
    ".eggs", "*.egg-info",
    ".tox", ".nox",
    "htmlcov", ".coverage",
    ".idea", ".vscode",
})

# Regex patterns applied to the *relative POSIX path* to detect generated files
_GENERATED_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p) for p in [
        r"\.egg-info/",
        r"(^|/)build/",
        r"(^|/)dist/",
        r"(^|/)__pycache__/",
        r"\.generated\.",
        r"(^|/)migrations/",       # Django/Alembic auto-generated
        r"(^|/)generated/",
        r"_pb2\.py$",              # protobuf generated
        r"_pb2_grpc\.py$",
    ]
)

# Regex patterns applied to the *relative POSIX path* to detect test files
_TEST_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p) for p in [
        r"(^|/)tests?/",
        r"(^|/)testing/",
        r"(^|/)test_[^/]+$",
        r"(^|/)[^/]+_test\.py$",
        r"(^|/)testing_[^/]+\.py$",
        r"(^|/)spec/",
        r"(^|/)__tests__/",
        r"\.test\.(js|ts|jsx|tsx)$",
        r"\.spec\.(js|ts|jsx|tsx)$",
    ]
)


def _detect_language(path: Path) -> str:
    """Return the programming language label for *path* (empty string = unknown)."""
    name = path.name.lower()
    # Exact filename match first
    lang = _NAME_LANGUAGE.get(name)
    if lang:
        return lang
    return _EXT_LANGUAGE.get(path.suffix.lower(), "")


def _is_generated(rel_posix: str) -> bool:
    return any(p.search(rel_posix) for p in _GENERATED_PATTERNS)


def _is_test(rel_posix: str) -> bool:
    return any(p.search(rel_posix) for p in _TEST_PATTERNS)


def _sha256(path: Path, chunk_size: int = 65536) -> str:
    """Return the SHA-256 hex digest of *path*'s contents."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while True:
                block = fh.read(chunk_size)
                if not block:
                    break
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def _mtime_iso(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return ""


def _load_gitignore_specs(repo_root: Path) -> list:
    """Return a list of (base_path, PathSpec) pairs for every .gitignore found
    inside *repo_root* (including the root-level one).

    Each PathSpec is evaluated relative to the directory that contains the
    .gitignore file, which matches how git itself resolves patterns.

    Returns an empty list when pathspec is not installed or no .gitignore
    files exist.
    """
    if not _HAS_PATHSPEC:
        log.debug("pathspec not installed — .gitignore patterns will not be applied")
        return []

    specs: list[tuple[Path, object]] = []
    for gitignore in sorted(repo_root.rglob(".gitignore")):
        # Never descend into .git itself
        try:
            gitignore.relative_to(repo_root / ".git")
            continue
        except ValueError:
            pass
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
            spec = _pathspec.PathSpec.from_lines("gitwildmatch", lines)
            specs.append((gitignore.parent, spec))
        except OSError:
            pass

    log.debug("_load_gitignore_specs: loaded %d .gitignore file(s)", len(specs))
    return specs


def _is_gitignored(path: Path, is_dir: bool, specs: list) -> bool:
    """Return True if *path* is matched by any of the loaded .gitignore specs.

    Each spec is applied relative to the directory that owns it, so nested
    .gitignore files only affect paths beneath them.
    """
    for base, spec in specs:
        try:
            rel = path.relative_to(base).as_posix()
        except ValueError:
            continue  # path is not under this spec's base directory
        # For directories append '/' so patterns like 'build/' match correctly
        if is_dir:
            if spec.match_file(rel + "/") or spec.match_file(rel):
                return True
        else:
            if spec.match_file(rel):
                return True
    return False


def _git_branch(repo_root: Path) -> str:
    """Return the current git branch name, or '' if not a git repo / git absent."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


# ---------------------------------------------------------------------------
# RepoScanner
# ---------------------------------------------------------------------------

class RepoScanner:
    """Scan a repository directory and produce a RepoManifest.

    Args
    ----
    excluded_dirs    : Additional directory names to skip on top of the
                       built-in ``_EXCLUDED_DIRS`` set.
    extra_extensions : Additional extension → language mappings.
    max_file_size    : Files larger than this (bytes) are recorded in the
                       manifest but their content_hash is set to "".
                       Default: 10 MB.
    """

    def __init__(
        self,
        excluded_dirs:    Collection[str] | None = None,
        extra_extensions: dict[str, str] | None  = None,
        max_file_size:    int = 10 * 1024 * 1024,
        use_gitignore:    bool = True,
    ) -> None:
        self._excluded = _EXCLUDED_DIRS | set(excluded_dirs or [])
        self._ext_lang: dict[str, str] = {
            **_EXT_LANGUAGE,
            **(extra_extensions or {}),
        }
        self._max_file_size = max_file_size
        self._use_gitignore = use_gitignore

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        repo_path: str | Path,
        repo_id:   str,
        branch:    str | None = None,
    ) -> RepoManifest:
        """Walk *repo_path* and return a fully-populated RepoManifest.

        Args
        ----
        repo_path : Absolute path to the repository root.
        repo_id   : Logical identifier for this repository.
        branch    : Override the git branch name.  Auto-detected when None.
        """
        root    = Path(repo_path).resolve()
        branch  = branch if branch is not None else _git_branch(root)
        files: dict[str, RepoFile] = {}
        total   = 0
        skipped = 0

        gitignore_specs = _load_gitignore_specs(root) if self._use_gitignore else []

        for dirpath, dirnames, filenames in os.walk(root):
            cur_dir = Path(dirpath)
            # Prune excluded directories in-place so os.walk does not descend.
            # Both the hardcoded set and any .gitignore patterns are applied.
            dirnames[:] = [
                d for d in dirnames
                if (
                    d not in self._excluded
                    and not d.endswith(".egg-info")
                    and not _is_gitignored(cur_dir / d, is_dir=True, specs=gitignore_specs)
                )
            ]

            for filename in filenames:
                abs_path  = Path(dirpath) / filename
                rel_posix = abs_path.relative_to(root).as_posix()
                total    += 1

                # Skip files matched by .gitignore
                if gitignore_specs and _is_gitignored(abs_path, is_dir=False, specs=gitignore_specs):
                    skipped += 1
                    continue

                try:
                    stat = abs_path.stat()
                except OSError as exc:
                    log.warning("scan: cannot stat %s — %s", rel_posix, exc)
                    skipped += 1
                    continue

                language  = _detect_language(abs_path)
                generated = _is_generated(rel_posix)
                test      = _is_test(rel_posix)

                # Hash only if file is small enough
                if stat.st_size <= self._max_file_size:
                    content_hash = _sha256(abs_path)
                else:
                    content_hash = ""
                    log.debug("scan: skipping hash for large file %s (%d bytes)",
                              rel_posix, stat.st_size)

                mtime = datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds")

                files[rel_posix] = RepoFile(
                    repo_id=repo_id,
                    branch=branch,
                    file_path=rel_posix,
                    language=language,
                    size=stat.st_size,
                    is_test=test,
                    is_generated=generated,
                    content_hash=content_hash,
                    mtime=mtime,
                )

        log.info(
            "scan: repo=%r  root=%s  branch=%r  "
            "total=%d  indexed=%d  skipped=%d",
            repo_id, root, branch, total, len(files), skipped,
        )
        return RepoManifest(
            repo_id=repo_id,
            repo_root=str(root),
            branch=branch,
            files=files,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def diff(old: RepoManifest, new: RepoManifest) -> ManifestDiff:
        """Compute the change set between two consecutive manifests.

        A file is considered *modified* when its ``content_hash`` changed
        (not merely its mtime), so no-op touches are ignored.
        """
        old_keys = set(old.files)
        new_keys = set(new.files)

        added    = [new.files[k] for k in (new_keys - old_keys)]
        deleted  = [old.files[k] for k in (old_keys - new_keys)]
        modified = [
            new.files[k]
            for k in (old_keys & new_keys)
            if new.files[k].content_hash != old.files[k].content_hash
        ]

        diff = ManifestDiff(added=added, modified=modified, deleted=deleted)
        log.info("diff: %s", diff.summary())
        return diff
