"""Tests for soft-scope code query conventions."""

from __future__ import annotations

import inspect

from langchain_core.documents import Document

from rag.client import LocalLlamaClient
from rag.retrieval.code_query_scope import parse_code_query_scope, rerank_code_rows_by_scope
from rag.retrieval.code_retrieval_service import CodeRetrievalService


def _doc(repo_id: str, file_path: str, name: str) -> Document:
    return Document(
        page_content=f"def {name}():\n    return 1",
        metadata={
            "repo_id": repo_id,
            "file_path": file_path,
            "name": name,
            "chunk_id": f"{repo_id}::{file_path}::function::{name}",
            "chunk_type": "function",
        },
    )


def test_parse_code_query_scope_extracts_soft_hints():
    scope = parse_code_query_scope(
        "repo:payments-service path:src/payments module:payments.service find caller"
    )

    assert scope.repo_terms == ("payments-service",)
    assert scope.path_terms == ("src/payments",)
    assert scope.module_terms == ("payments.service",)
    assert scope.semantic_query == "find caller"
    assert scope.has_hints is True


def test_rerank_code_rows_by_scope_promotes_matching_results():
    rows = [
        (_doc("inventory-service", "src/inventory/service.py", "sync_stock"), 0.82),
        (_doc("payments-service", "src/payments/service.py", "sync_payment"), 0.78),
        (_doc("payments-service", "src/payments/api.py", "get_payment"), 0.76),
    ]

    scope = parse_code_query_scope(
        "repo:payments-service path:src/payments module:payments.service find sync"
    )
    reranked = rerank_code_rows_by_scope(rows, scope)

    assert reranked[0][0].metadata["repo_id"] == "payments-service"
    assert reranked[0][0].metadata["file_path"] == "src/payments/service.py"
    assert reranked[0][1] > rows[0][1]


def test_search_code_blocks_uses_soft_scope_parser_and_reranking():
    source = inspect.getsource(CodeRetrievalService.search_code_blocks)
    assert "parse_code_query_scope" in source
    assert "rerank_code_rows_by_scope" in source
    assert "semantic_query" in source
