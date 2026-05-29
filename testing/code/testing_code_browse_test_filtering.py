"""Tests for browse_code_blocks test-data filtering behavior."""

from __future__ import annotations

from types import SimpleNamespace

from rag.client import LocalLlamaClient
from rag.retrieval.code_result_filter import CodeResultFilter
from rag.retrieval.code_retrieval_service import CodeRetrievalService


class _FakeChroma:
    def __init__(self, **_kwargs):
        pass

    def get(self, **_kwargs):
        return {
            "documents": [
                "def prod():\n    return 1",
                "def test_case():\n    assert True",
            ],
            "metadatas": [
                {"repo_id": "r1", "file_path": "src/app/service.py", "chunk_id": "a"},
                {"repo_id": "r1", "file_path": "testing/ui/testing_service.py", "chunk_id": "b"},
            ],
        }


class _DummyClient:
    browse_code_blocks = LocalLlamaClient.browse_code_blocks
    _is_test_code_metadata = staticmethod(LocalLlamaClient._is_test_code_metadata)
    _code_block_persist_dirs = lambda self: ["/tmp/code_rag"]

    def __init__(self):
        self.embed = object()
        self.config = SimpleNamespace(code_rag_root="/tmp/code_rag", graph_db_path="/tmp/graph.db")
        self.persist_directory = "/tmp/code_rag"
        self._code_retrieval = CodeRetrievalService(
            config=self.config,
            embed=self.embed,
            reranker=None,
            persist_directory=self.persist_directory,
            code_result_filter=CodeResultFilter(),
        )


def test_browse_code_blocks_excludes_test_rows_by_default(monkeypatch) -> None:
    import rag.retrieval.code_block_store as store_module

    monkeypatch.setattr(store_module, "Chroma", _FakeChroma)
    c = _DummyClient()
    rows = c.browse_code_blocks(limit=10)

    assert len(rows) == 1
    assert rows[0]["metadata"]["file_path"] == "src/app/service.py"


def test_browse_code_blocks_can_include_test_rows(monkeypatch) -> None:
    import rag.retrieval.code_block_store as store_module

    monkeypatch.setattr(store_module, "Chroma", _FakeChroma)
    c = _DummyClient()
    rows = c.browse_code_blocks(limit=10, exclude_tests=False)

    assert len(rows) == 2
    paths = {row["metadata"].get("file_path", "") for row in rows}
    assert "src/app/service.py" in paths
    assert "testing/ui/testing_service.py" in paths
