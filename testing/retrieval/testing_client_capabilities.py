"""Unit tests for LocalLlamaClient capability composition wrappers."""

from __future__ import annotations

from types import SimpleNamespace

from rag.client import LocalLlamaClient


def _build_dummy_client() -> LocalLlamaClient:
    client = LocalLlamaClient.__new__(LocalLlamaClient)
    return client


def test_retrieval_wrapper_delegates_to_retrieval_api() -> None:
    client = _build_dummy_client()
    calls = {"list_doc_ids": 0}

    def _list_doc_ids():
        calls["list_doc_ids"] += 1
        return ["doc-1"]

    client.retrieval_api = SimpleNamespace(list_doc_ids=_list_doc_ids)

    assert client.list_doc_ids() == ["doc-1"]
    assert calls["list_doc_ids"] == 1


def test_knowledge_wrapper_delegates_to_knowledge_api() -> None:
    client = _build_dummy_client()
    calls = {"enrich_doc": 0}

    def _enrich_doc(doc_id: str, overwrite: bool = False):
        calls["enrich_doc"] += 1
        return {"doc_id": doc_id, "overwrite": overwrite}

    client.knowledge_api = SimpleNamespace(enrich_doc=_enrich_doc)

    assert client.enrich_doc("d1", overwrite=True) == {"doc_id": "d1", "overwrite": True}
    assert calls["enrich_doc"] == 1


def test_generation_wrapper_delegates_to_generation_api() -> None:
    client = _build_dummy_client()
    calls = {"answer_query": 0}

    def _answer_query(query: str, **kwargs):
        calls["answer_query"] += 1
        return {"query": query, "kwargs": kwargs}

    client.generation_api = SimpleNamespace(answer_query=_answer_query)

    out = client.answer_query("hello", k=3, fetch_k=7)
    assert out["query"] == "hello"
    assert out["kwargs"]["k"] == 3
    assert out["kwargs"]["fetch_k"] == 7
    assert calls["answer_query"] == 1


def test_indexing_wrapper_delegates_to_indexing_api() -> None:
    client = _build_dummy_client()
    calls = {"add_texts": 0}

    def _add_texts(texts, metadatas=None, ids=None):
        calls["add_texts"] += 1
        return {"texts": texts, "metadatas": metadatas, "ids": ids}

    client.indexing_api = SimpleNamespace(add_texts=_add_texts)

    out = client.add_texts(["alpha"], metadatas=[{"doc_id": "d1"}], ids=["i1"])
    assert out["texts"] == ["alpha"]
    assert out["metadatas"] == [{"doc_id": "d1"}]
    assert out["ids"] == ["i1"]
    assert calls["add_texts"] == 1
