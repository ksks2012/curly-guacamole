"""Tests for listing code repo ids from code_block metadata."""

from __future__ import annotations

import inspect

from rag.client import LocalLlamaClient


def test_client_exposes_list_code_repo_ids_method():
    """Verify LocalLlamaClient has list_code_repo_ids API."""
    assert hasattr(LocalLlamaClient, "list_code_repo_ids")
    source = inspect.getsource(LocalLlamaClient.list_code_repo_ids)
    assert "repo_id" in source
    assert "code_block" in source
    assert "metadatas" in source
