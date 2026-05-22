"""Shared fixtures for testing/code/.

The --integration flag and skip logic live in the parent testing/conftest.py.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

# Make workspace root importable without sys.path.insert in each test file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class _MockEmbeddings:
    """Deterministic fake embedding — sha256 digest → 16-dim float vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[:16]]


@pytest.fixture(scope="session")
def mock_embed() -> _MockEmbeddings:
    """Deterministic fake embedding function (sha256 → 16-dim, no server needed)."""
    return _MockEmbeddings()


@pytest.fixture
def graph_store(tmp_path):
    """Fresh GraphStore backed by a temporary SQLite database."""
    from rag.code.graph_store import GraphStore
    return GraphStore(str(tmp_path / "graph.db"))


@pytest.fixture
def commit_indexer(tmp_path, mock_embed):
    """Fresh CommitIndexer backed by a temporary Chroma directory."""
    from rag.code.commit_indexer import CommitIndexer
    return CommitIndexer(str(tmp_path), mock_embed)


# ---------------------------------------------------------------------------
# Fixtures for testing_symbol_store.py
# ---------------------------------------------------------------------------

_SYMBOL_SAMPLE = '''\
"""Module docstring."""


class Animal:
    """Base animal class."""

    def __init__(self, name: str) -> None:
        self._name = name

    def speak(self) -> str:
        """Return the animal sound."""
        return ""

    def __repr__(self) -> str:
        return f"Animal({self._name!r})"


class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"


def make_animal(kind: str) -> Animal:
    """Factory function."""

    def _validate(k: str) -> None:
        if not k:
            raise ValueError("kind must not be empty")

    _validate(kind)
    return Animal(kind)
'''


@pytest.fixture(scope="module")
def parser():
    """PythonASTParser instance shared across all tests in the module."""
    from rag.code.ast_parser import PythonASTParser
    return PythonASTParser()


@pytest.fixture(scope="module")
def store(parser):
    """SymbolStore built from the synthetic Animal sample source."""
    from rag.code.symbol_store import SymbolStore
    chunks = parser.parse(_SYMBOL_SAMPLE, file_path="animals.py", repo_id="test")
    return SymbolStore.from_chunks(chunks)
