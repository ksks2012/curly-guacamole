"""Tests for GCR2.5 Phase 1 related block expansion."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from rag.retrieval.base import RetrievalResult
from rag.retrieval.related_code_retriever import RelatedCodeRetriever


class _FakeBlockDb:
    def __init__(self, exact_metas=None, file_metas=None):
        self.exact_metas = exact_metas or []
        self.file_metas = file_metas or []

    def get(self, where=None, include=None):
        where = where or {}
        clauses = where.get("$and", [])
        keys = {next(iter(c.keys())) for c in clauses if c}
        if "chunk_id" in keys:
            return {"metadatas": list(self.exact_metas)}
        if "file_path" in keys:
            return {"metadatas": list(self.file_metas)}
        return {"metadatas": []}


class _BaseWithIndexer:
    def __init__(self, result: RetrievalResult, db: _FakeBlockDb):
        self._result = result
        self._indexer = MagicMock()
        self._indexer._db.return_value = db

    def search(self, query, top_k=5, filters=None):
        return [self._result]


def _mk_result(
    *,
    chunk_id: str = "r1::a.py::function::f",
    repo_id: str = "r1",
    file_path: str = "a.py",
    start_line: int = 10,
) -> RetrievalResult:
    return RetrievalResult(
        content="def f(): pass",
        score=0.9,
        source="code",
        metadata={
            "chunk_id": chunk_id,
            "repo_id": repo_id,
            "file_path": file_path,
            "start_line": start_line,
            "name": "f",
            "chunk_type": "function",
        },
    )


def test_response_contract_related_blocks_keys():
    base = MagicMock()
    base.search.return_value = [_mk_result()]

    edge = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="r1::a.py::function::g",
        edge_type="CALLS",
        line_no=12,
    )

    graph = MagicMock()
    graph.get_edges.side_effect = [[edge], []]

    def fetch_block(repo_id: str, target_id: str):
        assert repo_id == "r1"
        if target_id == "r1::a.py::function::g":
            return {
                "chunk_id": target_id,
                "file_path": "a.py",
                "name": "g",
                "chunk_type": "function",
                "start_line": 20,
            }
        return None

    r = RelatedCodeRetriever(
        base,
        graph,
        max_related=5,
        max_nearby=0,
        block_fetcher=fetch_block,
    )

    out = r.search("query", top_k=3)
    assert len(out) == 1
    rel = out[0].metadata["related_blocks"][0]

    assert "target_id" in rel
    assert "edge_type" in rel
    assert "direction" in rel
    assert "reason" in rel
    assert "score" in rel
    assert "explain" in rel


def test_repo_ids_passthrough_to_base_retriever():
    base = MagicMock()
    base.search.return_value = [_mk_result()]

    graph = MagicMock()
    graph.get_edges.side_effect = [[], []]

    r = RelatedCodeRetriever(base, graph, max_related=0, max_nearby=0)
    r.search("query", top_k=2, repo_ids=["r1", "r2"])

    base.search.assert_called_once_with(
        "query",
        top_k=2,
        filters=None,
        repo_ids=["r1", "r2"],
    )


def test_fallback_when_base_does_not_accept_repo_ids():
    class _Base:
        def __init__(self):
            self.calls = []

        def search(self, query, top_k=5, filters=None):
            self.calls.append((query, top_k, filters))
            return [_mk_result()]

    base = _Base()
    graph = MagicMock()
    graph.get_edges.side_effect = [[], []]

    r = RelatedCodeRetriever(base, graph, max_related=0, max_nearby=0)
    out = r.search("q", top_k=1, repo_ids=["r1"])

    assert len(out) == 1
    assert base.calls == [("q", 1, None)]


def test_relation_expansion_outgoing_and_incoming():
    base = MagicMock()
    base.search.return_value = [_mk_result()]

    out_e = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="r1::a.py::function::g",
        edge_type="CALLS",
        line_no=11,
    )
    in_e = SimpleNamespace(
        src_id="r1::a.py::method::h",
        dst_id="r1::a.py::function::f",
        edge_type="IMPLEMENTS",
        line_no=7,
    )

    graph = MagicMock()
    graph.get_edges.side_effect = [[out_e], [in_e]]

    def fetch_block(_repo_id: str, target_id: str):
        return {
            "chunk_id": target_id,
            "file_path": "a.py",
            "name": "x",
            "chunk_type": "function",
            "start_line": 20,
        }

    r = RelatedCodeRetriever(base, graph, max_related=5, max_nearby=0, block_fetcher=fetch_block)
    out = r.search("q")

    rel = out[0].metadata["related_blocks"]
    assert len(rel) == 2
    assert {x["direction"] for x in rel} == {"outgoing", "incoming"}
    assert all("explain" in x for x in rel)


def test_skip_import_targets_and_limit_max_related():
    base = MagicMock()
    base.search.return_value = [_mk_result()]

    e1 = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="import::os.getcwd",
        edge_type="CALLS",
        line_no=1,
    )
    e2 = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="r1::a.py::function::g",
        edge_type="CALLS",
        line_no=2,
    )
    e3 = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="r1::a.py::function::k",
        edge_type="CALLS",
        line_no=3,
    )

    graph = MagicMock()
    graph.get_edges.side_effect = [[e1, e2, e3], []]

    def fetch_block(_repo_id: str, target_id: str):
        return {"chunk_id": target_id, "file_path": "a.py", "name": "x", "chunk_type": "function"}

    r = RelatedCodeRetriever(base, graph, max_related=1, max_nearby=0, block_fetcher=fetch_block)
    out = r.search("q")

    rel = out[0].metadata["related_blocks"]
    assert len(rel) == 1
    assert rel[0]["target_id"] != "import::os.getcwd"


def test_same_file_nearby_relations():
    base = MagicMock()
    base.search.return_value = [_mk_result(start_line=50)]

    graph = MagicMock()
    graph.get_edges.side_effect = [[], []]

    def file_blocks(_repo_id: str, _file_path: str):
        return [
            {"chunk_id": "r1::a.py::function::f", "start_line": 50, "name": "f", "chunk_type": "function", "file_path": "a.py"},
            {"chunk_id": "r1::a.py::function::g", "start_line": 55, "name": "g", "chunk_type": "function", "file_path": "a.py"},
            {"chunk_id": "r1::a.py::function::h", "start_line": 200, "name": "h", "chunk_type": "function", "file_path": "a.py"},
        ]

    r = RelatedCodeRetriever(
        base,
        graph,
        max_related=0,
        max_nearby=1,
        file_blocks_fetcher=file_blocks,
    )

    out = r.search("q")
    rel = out[0].metadata["related_blocks"]

    assert len(rel) == 1
    assert rel[0]["edge_type"] == "NEARBY"
    assert rel[0]["target_id"] == "r1::a.py::function::g"
    assert "line distance=" in rel[0]["explain"]


def test_relations_are_ranked_by_score_descending():
    base = MagicMock()
    base.search.return_value = [_mk_result()]

    # CALLS should rank higher than IMPORTS under fixed weights.
    e1 = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="r1::a.py::function::g",
        edge_type="IMPORTS",
        line_no=12,
    )
    e2 = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="r1::a.py::function::h",
        edge_type="CALLS",
        line_no=12,
    )

    graph = MagicMock()
    graph.get_edges.side_effect = [[e1, e2], []]

    def fetch_block(_repo_id: str, target_id: str):
        return {
            "chunk_id": target_id,
            "file_path": "a.py",
            "name": "x",
            "chunk_type": "function",
            "start_line": 20,
        }

    r = RelatedCodeRetriever(base, graph, max_related=5, max_nearby=0, block_fetcher=fetch_block)
    out = r.search("q")
    rel = out[0].metadata["related_blocks"]

    assert len(rel) == 2
    assert rel[0]["target_id"] == "r1::a.py::function::h"
    assert rel[0]["score"] >= rel[1]["score"]


def test_duplicate_targets_are_aggregated_with_evidence_count():
    base = MagicMock()
    base.search.return_value = [_mk_result()]

    # Same target from outgoing and incoming edges should aggregate into one row.
    out_e = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="r1::a.py::function::g",
        edge_type="CALLS",
        line_no=8,
    )
    in_e = SimpleNamespace(
        src_id="r1::a.py::function::g",
        dst_id="r1::a.py::function::f",
        edge_type="IMPLEMENTS",
        line_no=9,
    )

    graph = MagicMock()
    graph.get_edges.side_effect = [[out_e], [in_e]]

    def fetch_block(_repo_id: str, target_id: str):
        return {
            "chunk_id": target_id,
            "file_path": "a.py",
            "name": "g",
            "chunk_type": "function",
            "start_line": 20,
        }

    r = RelatedCodeRetriever(base, graph, max_related=5, max_nearby=0, block_fetcher=fetch_block)
    out = r.search("q")
    rel = out[0].metadata["related_blocks"]

    assert len(rel) == 1
    assert rel[0]["target_id"] == "r1::a.py::function::g"
    assert rel[0]["evidence_count"] == 2
    assert set(rel[0]["edge_types"]) == {"CALLS", "IMPLEMENTS"}


def test_main_search_order_is_preserved():
    # Related expansion must not reorder primary retrieval results.
    r1 = _mk_result(chunk_id="r1::a.py::function::f", start_line=10)
    r2 = _mk_result(chunk_id="r1::a.py::function::z", start_line=100)
    base = MagicMock()
    base.search.return_value = [r1, r2]

    graph = MagicMock()
    graph.get_edges.side_effect = [[], [], [], []]

    r = RelatedCodeRetriever(base, graph, max_related=5, max_nearby=0)
    out = r.search("q")

    assert [x.metadata["chunk_id"] for x in out] == [
        "r1::a.py::function::f",
        "r1::a.py::function::z",
    ]


def test_symbol_to_block_mapping_fallback_repo_file_name_key():
    # target_id is method symbol; no exact chunk_id row exists in block DB.
    edge = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="r1::a.py::method::Service.run",
        edge_type="CALLS",
        line_no=42,
    )

    file_metas = [
        {
            "chunk_id": "r1::a.py::method::Service.run",
            "repo_id": "r1",
            "file_path": "a.py",
            "name": "Service.run",
            "chunk_type": "method",
            "start_line": 40,
            "end_line": 60,
        }
    ]
    db = _FakeBlockDb(exact_metas=[], file_metas=file_metas)
    base = _BaseWithIndexer(_mk_result(), db)

    graph = MagicMock()
    graph.get_edges.side_effect = [[edge], []]

    r = RelatedCodeRetriever(base, graph, max_related=5, max_nearby=0)
    out = r.search("q")
    rel = out[0].metadata["related_blocks"]

    assert len(rel) == 1
    assert rel[0]["target_id"] == "r1::a.py::method::Service.run"
    assert rel[0]["mapping_strategy"] == "symbol_key_fallback"


def test_symbol_mapping_prefers_smallest_enclosing_block():
    edge = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="r1::a.py::function::core.logic",
        edge_type="CALLS",
        line_no=80,
    )

    # Both candidates match (repo_id, file_path, name).
    # Smaller span should win first.
    file_metas = [
        {
            "chunk_id": "r1::a.py::function::core.logic::wide",
            "repo_id": "r1",
            "file_path": "a.py",
            "name": "core.logic",
            "chunk_type": "function",
            "start_line": 1,
            "end_line": 200,
        },
        {
            "chunk_id": "r1::a.py::function::core.logic::narrow",
            "repo_id": "r1",
            "file_path": "a.py",
            "name": "core.logic",
            "chunk_type": "function",
            "start_line": 70,
            "end_line": 90,
        },
    ]
    db = _FakeBlockDb(exact_metas=[], file_metas=file_metas)
    base = _BaseWithIndexer(_mk_result(), db)

    graph = MagicMock()
    graph.get_edges.side_effect = [[edge], []]

    r = RelatedCodeRetriever(base, graph, max_related=5, max_nearby=0)
    out = r.search("q")
    rel = out[0].metadata["related_blocks"]

    assert len(rel) == 1
    assert rel[0]["target_id"] == "r1::a.py::function::core.logic"
    assert rel[0]["target_name"] == "core.logic"


def test_symbol_mapping_prefers_nearest_line_when_span_equal():
    edge = SimpleNamespace(
        src_id="r1::a.py::function::f",
        dst_id="r1::a.py::function::worker",
        edge_type="CALLS",
        line_no=90,
    )

    # Same span size, different start lines. Should choose nearest to edge line_no.
    file_metas = [
        {
            "chunk_id": "r1::a.py::function::worker::far",
            "repo_id": "r1",
            "file_path": "a.py",
            "name": "worker",
            "chunk_type": "function",
            "start_line": 20,
            "end_line": 30,
        },
        {
            "chunk_id": "r1::a.py::function::worker::near",
            "repo_id": "r1",
            "file_path": "a.py",
            "name": "worker",
            "chunk_type": "function",
            "start_line": 88,
            "end_line": 98,
        },
    ]
    db = _FakeBlockDb(exact_metas=[], file_metas=file_metas)
    base = _BaseWithIndexer(_mk_result(), db)

    graph = MagicMock()
    graph.get_edges.side_effect = [[edge], []]

    r = RelatedCodeRetriever(base, graph, max_related=5, max_nearby=0)
    out = r.search("q")
    rel = out[0].metadata["related_blocks"]

    assert len(rel) == 1
    assert rel[0]["target_id"] == "r1::a.py::function::worker"
    # Mapping fallback should still provide stable target metadata.
    assert rel[0]["target_file_path"] == "a.py"
