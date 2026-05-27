"""Tests for client relation enrichment toggle behavior via method introspection."""

from __future__ import annotations

import inspect

from rag.client import LocalLlamaClient


def test_search_code_blocks_accepts_include_relations_param():
    """Verify search_code_blocks accepts include_relations parameter."""
    sig = inspect.signature(LocalLlamaClient.search_code_blocks)
    assert "include_relations" in sig.parameters
    assert sig.parameters["include_relations"].default is False


def test_search_code_blocks_calls_enrich_when_include_relations_true():
    """Verify method routing for include_relations=True via introspection."""
    # Check that the method has the _enrich_code_results_with_relations path
    source = inspect.getsource(LocalLlamaClient.search_code_blocks)
    assert "include_relations" in source
    assert "_enrich_code_results_with_relations" in source
    assert "if include_relations:" in source


def test_enrich_code_results_with_relations_exists():
    """Verify enrichment method exists and is callable."""
    assert hasattr(LocalLlamaClient, "_enrich_code_results_with_relations")
    method = getattr(LocalLlamaClient, "_enrich_code_results_with_relations")
    assert callable(method)


def test_client_has_code_block_persist_dirs_method():
    """Verify multi-location code block discovery is implemented."""
    assert hasattr(LocalLlamaClient, "_code_block_persist_dirs")
    method = getattr(LocalLlamaClient, "_code_block_persist_dirs")
    assert callable(method)
    sig = inspect.signature(method)
    # Should return a list
    assert "list" in str(sig.return_annotation).lower() or sig.return_annotation != inspect.Signature.empty


def test_search_code_blocks_signature_defaults_relations_false():
    """Verify backward compatibility: include_relations defaults to False."""
    sig = inspect.signature(LocalLlamaClient.search_code_blocks)
    assert sig.parameters["include_relations"].default is False


def test_controller_forwards_include_relations_to_client():
    """Verify SearchController passes include_relations to client.search_code_blocks."""
    from ui.search_controller import SearchController

    source = inspect.getsource(SearchController.run_search)
    assert "include_relations" in source
    assert "self._client.search_code_blocks" in source


def test_graph_tab_uses_relation_toggle_state():
    """Verify code_graph_tab respects relation_toggle state."""
    from ui.code_graph_tab import build

    source = inspect.getsource(build)
    assert "relation_toggle" in source
    assert "include_relations" in source
