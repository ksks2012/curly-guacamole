"""Smoke test for GCR1.1 — RepoScanner."""

from pathlib import Path

import pytest

from rag.code.scanner import RepoScanner
from rag.code.schema import RepoManifest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def manifest() -> RepoManifest:
    scanner = RepoScanner()
    return scanner.scan(str(REPO_ROOT), repo_id="langchain-test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_scan(manifest):
    assert len(manifest) > 0, "manifest is empty"
    src = manifest.source_files()
    assert len(src) > 0, "no source files found"


@pytest.mark.integration
def test_self_diff(manifest):
    diff = RepoScanner.diff(manifest, manifest)
    assert diff.is_empty(), f"self-diff should be empty: {diff.summary()}"


@pytest.mark.integration
def test_round_trip(manifest, tmp_path):
    p = tmp_path / "manifest.json"
    manifest.save(p)
    loaded = RepoManifest.load(p)
    assert len(loaded) == len(manifest), \
        f"round-trip count mismatch: {len(loaded)} vs {len(manifest)}"
