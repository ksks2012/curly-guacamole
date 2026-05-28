"""Shared path-based heuristics for code indexing and retrieval filters."""

from __future__ import annotations

import re


_TEST_PATH_PATTERNS: tuple[str, ...] = (
    r"(^|/)(tests?|testing|spec|__tests__)(/|$)",
    r"(^|/)test_[^/]+$",
    r"(^|/)[^/]+_test\.py$",
    r"(^|/)testing_[^/]+\.py$",
    r"\.test\.(js|ts|jsx|tsx)$",
    r"\.spec\.(js|ts|jsx|tsx)$",
)

TEST_PATH_REGEX: re.Pattern = re.compile("|".join(_TEST_PATH_PATTERNS))


def normalize_rel_path(path: str) -> str:
    """Normalize a path for heuristic matching."""
    return str(path or "").strip().lower().replace("\\", "/")


def is_test_path(path: str) -> bool:
    """Return True if *path* matches known test path/file naming conventions."""
    normalized = normalize_rel_path(path)
    if not normalized:
        return False
    return bool(TEST_PATH_REGEX.search(normalized))
