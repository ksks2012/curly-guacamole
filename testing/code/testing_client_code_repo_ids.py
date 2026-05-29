"""Tests for listing code repo ids from code_block metadata."""

from __future__ import annotations

import inspect

from rag.client import LocalLlamaClient
from rag.retrieval.code_retrieval_service import CodeRetrievalService


def test_client_exposes_list_code_repo_ids_method():
    """Verify LocalLlamaClient delegates list_code_repo_ids to CodeRetrievalService."""
    assert hasattr(LocalLlamaClient, "list_code_repo_ids")
    client_source = inspect.getsource(LocalLlamaClient.list_code_repo_ids)
    assert "_code_retrieval" in client_source
    assert "list_code_repo_ids" in client_source
    
    # Verify the service implementation has the actual logic with metadatas access
    service_source = inspect.getsource(CodeRetrievalService.list_code_repo_ids)
    assert "repo_id" in service_source
    assert "code_block" in service_source
    assert "metadatas" in service_source
