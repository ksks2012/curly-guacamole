"""Tests for SearchController code relation toggle forwarding."""

from __future__ import annotations

from ui.search_controller import SearchController


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_code_blocks(self, query: str, *, k: int, fetch_k: int, use_rerank: bool, include_relations: bool):
        self.calls.append(
            {
                "query": query,
                "k": k,
                "fetch_k": fetch_k,
                "use_rerank": use_rerank,
                "include_relations": include_relations,
            }
        )
        return {
            "vector": [],
            "bm25": None,
            "hybrid": None,
            "reranked": None,
            "trace": [],
        }

    def search_for_trace(self, *args, **kwargs):
        raise AssertionError("search_for_trace should not be called for code scope")


def test_run_search_passes_include_relations_true_for_code_scope():
    client = _FakeClient()
    ctrl = SearchController(client)

    error = ctrl.run_search(
        "find caller",
        k=5,
        fetch_k=20,
        use_rerank=False,
        result_scope="code",
        apply_filter=False,
        include_relations=True,
    )

    assert error is None
    assert len(client.calls) == 1
    assert client.calls[0]["include_relations"] is True


def test_run_search_defaults_include_relations_false():
    client = _FakeClient()
    ctrl = SearchController(client)

    error = ctrl.run_search(
        "find caller",
        k=5,
        fetch_k=20,
        use_rerank=False,
        result_scope="code",
        apply_filter=False,
    )

    assert error is None
    assert len(client.calls) == 1
    assert client.calls[0]["include_relations"] is False
