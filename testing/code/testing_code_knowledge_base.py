"""Tests for CodeKnowledgeBase unified multi-repo lifecycle."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_manifest(repo_id: str):
    m = MagicMock()
    m.repo_id = repo_id
    return m


def _stats(added=0, updated=0, skipped=0, deleted=0):
    from rag.indexer import IndexStats
    return IndexStats(added=added, updated=updated, skipped=skipped, deleted=deleted)


def _make_kb():
    from rag.code.knowledge_base import CodeKnowledgeBase
    kb = CodeKnowledgeBase.__new__(CodeKnowledgeBase)
    kb._repo_index = MagicMock()
    kb._code_indexer = MagicMock()
    return kb


def test_init_constructs_repo_and_code_indexers():
    from rag.code.knowledge_base import CodeKnowledgeBase

    with patch("rag.code.knowledge_base.RepoIndex") as MockRepo, patch(
        "rag.code.knowledge_base.CodeIndexer"
    ) as MockCode:
        CodeKnowledgeBase(
            "/persist",
            embedding_function="embed",
            code_collection_prefix="myprefix",
            code_collection_names={"file": "docs"},
            repo_collection_name="repo_docs",
        )

    MockRepo.assert_called_once_with(
        "/persist",
        "embed",
        collection_name="repo_docs",
    )
    MockCode.assert_called_once_with(
        "/persist",
        "embed",
        collection_prefix="myprefix",
        collection_names={"file": "docs"},
    )


def test_ingest_combines_repo_and_code_stats():
    kb = _make_kb()
    manifest = _make_manifest("r1")
    chunks = [MagicMock()]

    kb._repo_index.ingest.return_value = _stats(added=1)
    kb._code_indexer.ingest.return_value = _stats(added=10, updated=2)

    out = kb.ingest((manifest, chunks))

    kb._repo_index.ingest.assert_called_once_with(manifest, chunks=chunks)
    kb._code_indexer.ingest.assert_called_once_with((manifest, chunks), store=None)
    assert out.added == 11 and out.updated == 2


def test_ingest_can_skip_repo_level():
    kb = _make_kb()
    manifest = _make_manifest("r1")
    chunks = []
    kb._code_indexer.ingest.return_value = _stats(added=3)

    out = kb.ingest((manifest, chunks), include_repo=False)

    kb._repo_index.ingest.assert_not_called()
    assert out.added == 3


def test_add_repo_aliases_ingest():
    kb = _make_kb()
    manifest = _make_manifest("r1")
    kb.ingest = MagicMock(return_value=_stats(added=1))

    out = kb.add_repo((manifest, []), include_repo=False)

    kb.ingest.assert_called_once_with((manifest, []), store=None, include_repo=False)
    assert out.added == 1


def test_update_combines_repo_and_code_stats():
    kb = _make_kb()
    manifest = _make_manifest("r1")
    chunks = []

    kb._repo_index.update.return_value = _stats(updated=1)
    kb._code_indexer.update.return_value = _stats(updated=5, skipped=2)

    out = kb.update((manifest, chunks))

    kb._repo_index.update.assert_called_once_with(manifest, chunks=chunks)
    kb._code_indexer.update.assert_called_once_with((manifest, chunks), store=None)
    assert out.updated == 6 and out.skipped == 2


def test_reindex_combines_repo_and_code_stats():
    kb = _make_kb()
    manifest = _make_manifest("r1")
    chunks = []

    kb._repo_index.reindex.return_value = _stats(added=1, deleted=1)
    kb._code_indexer.reindex.return_value = _stats(added=9, deleted=7)

    out = kb.reindex((manifest, chunks))

    kb._repo_index.reindex.assert_called_once_with(manifest, chunks=chunks)
    kb._code_indexer.reindex.assert_called_once_with((manifest, chunks), store=None)
    assert out.added == 10 and out.deleted == 8


def test_delete_combines_repo_and_code_stats():
    kb = _make_kb()
    kb._repo_index.delete.return_value = _stats(deleted=1)
    kb._code_indexer.delete.return_value = _stats(deleted=7)

    out = kb.delete("r1")

    kb._repo_index.delete.assert_called_once_with("r1")
    kb._code_indexer.delete.assert_called_once_with("r1")
    assert out.deleted == 8


def test_remove_repo_aliases_delete():
    kb = _make_kb()
    kb.delete = MagicMock(return_value=_stats(deleted=2))

    out = kb.remove_repo("r1", include_repo=False)

    kb.delete.assert_called_once_with("r1", include_repo=False)
    assert out.deleted == 2


def test_collection_stats_merges_repo_and_code_levels():
    kb = _make_kb()
    kb._repo_index.collection_stats.return_value = 2
    kb._code_indexer.collection_stats.return_value = {
        "file": 100,
        "symbol": 500,
        "block": 1000,
    }

    out = kb.collection_stats()

    assert out["repo"] == 2
    assert out["file"] == 100
    assert out["symbol"] == 500
    assert out["block"] == 1000


def test_ingest_many_aggregates_all_repos():
    kb = _make_kb()
    m1 = _make_manifest("r1")
    m2 = _make_manifest("r2")

    kb.ingest = MagicMock(side_effect=[_stats(added=2), _stats(added=3, updated=1)])

    out = kb.ingest_many([(m1, []), (m2, [])])

    assert kb.ingest.call_count == 2
    assert out.added == 5 and out.updated == 1


def test_ingest_many_passes_store_by_repo():
    kb = _make_kb()
    m1 = _make_manifest("r1")
    s1 = MagicMock()

    kb.ingest = MagicMock(return_value=_stats(added=1))

    kb.ingest_many([(m1, [])], store_by_repo={"r1": s1})

    kb.ingest.assert_called_once_with((m1, []), store=s1, include_repo=True)


def test_delete_many_aggregates_all_repos():
    kb = _make_kb()
    kb.delete = MagicMock(side_effect=[_stats(deleted=4), _stats(deleted=6)])

    out = kb.delete_many(["r1", "r2"])

    assert kb.delete.call_count == 2
    assert out.deleted == 10


def test_update_many_aggregates_all_repos():
    kb = _make_kb()
    m1 = _make_manifest("r1")
    m2 = _make_manifest("r2")

    kb.update = MagicMock(side_effect=[_stats(updated=1), _stats(updated=2, skipped=1)])

    out = kb.update_many([(m1, []), (m2, [])])

    assert kb.update.call_count == 2
    assert out.updated == 3 and out.skipped == 1


def test_reindex_many_aggregates_all_repos():
    kb = _make_kb()
    m1 = _make_manifest("r1")
    m2 = _make_manifest("r2")

    kb.reindex = MagicMock(side_effect=[_stats(added=3, deleted=1), _stats(added=4, deleted=2)])

    out = kb.reindex_many([(m1, []), (m2, [])])

    assert kb.reindex.call_count == 2
    assert out.added == 7 and out.deleted == 3


def test_repo_index_property_exposes_internal_instance():
    kb = _make_kb()
    assert kb.repo_index is kb._repo_index


def test_code_indexer_property_exposes_internal_instance():
    kb = _make_kb()
    assert kb.code_indexer is kb._code_indexer
