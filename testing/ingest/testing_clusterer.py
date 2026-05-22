"""
Unit tests for B.3 Topic Clustering (testing/testing_clusterer.py).

Mock tests (no live LLM, no Chroma DB needed):
  1. _to_slug — normalisation edge cases
  2. TopicClusterer.fit — correct cluster/chunk assignments with fake embeddings
  3. TopicClusterer.assign — metadata merge written correctly
  4. TopicClusterer.fit_and_assign — end-to-end with mocks
  5. KnowledgeManager.cluster_topics — delegates to clusterer
  6. KnowledgeManager.cluster_topics without clusterer — raises RuntimeError

Live test (--no-live to skip):
  7. Full round-trip against real Chroma + real LLM
"""

from __future__ import annotations

import pytest
import types
from unittest.mock import MagicMock, patch

import numpy as np


def _make_fake_chroma(n_docs: int = 12, dim: int = 8) -> MagicMock:
    """Build a mock Chroma instance with deterministic embeddings."""
    rng = np.random.default_rng(0)
    # Two well-separated clusters: first half near [1,0,...], second half near [0,1,...]
    half = n_docs // 2
    embeddings = np.vstack([
        rng.normal([1] + [0] * (dim - 1), 0.05, (half, dim)),
        rng.normal([0, 1] + [0] * (dim - 2), 0.05, (n_docs - half, dim)),
    ]).tolist()

    ids       = [f"id_{i}" for i in range(n_docs)]
    documents = [f"Text about RAG retrieval #{i}" if i < half
                 else f"Text about evolutionary algorithms #{i}"
                 for i in range(n_docs)]

    db                = MagicMock()
    db.get.return_value = {
        "ids":        ids,
        "embeddings": embeddings,
        "documents":  documents,
        "metadatas":  [{"doc_id": "doc_a"} for _ in range(n_docs)],
    }
    db._collection    = MagicMock()
    return db, ids, documents


def _make_mock_llm(label: str = "rag") -> MagicMock:
    """LLM that always returns a fixed topic label."""
    llm = MagicMock()
    response = MagicMock()
    response.content = label
    llm.invoke.return_value = response
    return llm


# ---------------------------------------------------------------------------
# Test 1: _to_slug
# ---------------------------------------------------------------------------

def test_to_slug() -> None:
    from rag.knowledge.clusterer import _to_slug

    assert _to_slug("RAG Retrieval") == "topic_rag_retrieval"
    assert _to_slug("topic_rag")     == "topic_rag"
    assert _to_slug("topic: agents") == "topic_agents"
    assert _to_slug("label: LLM Fine-Tuning!") == "topic_llm_fine_tuning"
    # Long label is truncated to 40 chars after "topic_"
    long = "a" * 50
    result = _to_slug(long)
    assert result.startswith("topic_")
    assert len(result) <= len("topic_") + 40


# ---------------------------------------------------------------------------
# Test 2: TopicClusterer.fit
# ---------------------------------------------------------------------------

def test_fit() -> None:
    from rag.knowledge.clusterer import TopicClusterer

    db, ids, _ = _make_fake_chroma(n_docs=12, dim=8)
    llm        = _make_mock_llm("rag")

    clusterer = TopicClusterer(llm=llm, db=db, n_repr=3, random_state=0)
    topic_map = clusterer.fit(n_clusters=2)

    assert topic_map.n_clusters == 2
    assert topic_map.n_chunks   == 12
    assert len(topic_map.cluster_labels) == 2
    assert len(topic_map.chunk_topics)   == 12
    # All chunk_topics values must be in cluster_labels values
    valid_topics = set(topic_map.cluster_labels.values())
    for t in topic_map.chunk_topics.values():
        assert t in valid_topics, f"Unexpected topic: {t}"
    # LLM was called once per cluster
    assert llm.invoke.call_count == 2


# ---------------------------------------------------------------------------
# Test 3: TopicClusterer.fit — fewer chunks than n_clusters
# ---------------------------------------------------------------------------

def test_fit_fewer_chunks() -> None:
    from rag.knowledge.clusterer import TopicClusterer

    db, ids, _ = _make_fake_chroma(n_docs=3, dim=4)
    llm        = _make_mock_llm("small")

    clusterer = TopicClusterer(llm=llm, db=db, random_state=0)
    # Requesting 8 clusters with only 3 docs — should silently reduce
    topic_map = clusterer.fit(n_clusters=8)

    assert topic_map.n_clusters == 3
    assert topic_map.n_chunks   == 3


# ---------------------------------------------------------------------------
# Test 4: TopicClusterer.assign — correct metadata merge
# ---------------------------------------------------------------------------

def test_assign() -> None:
    from rag.knowledge.clusterer import TopicClusterer, TopicMap

    db     = MagicMock()
    db._collection = MagicMock()
    db.get.return_value = {
        "metadatas": [
            {"doc_id": "d1", "source": "file.pdf"},
            {"doc_id": "d2"},
        ]
    }

    topic_map = TopicMap(
        cluster_labels={0: "topic_rag", 1: "topic_agents"},
        chunk_topics={"chunk_a": "topic_rag", "chunk_b": "topic_agents"},
        n_clusters=2,
        n_chunks=2,
    )

    llm       = MagicMock()
    clusterer = TopicClusterer(llm=llm, db=db, random_state=0)
    count     = clusterer.assign(topic_map)

    assert count == 2
    call_args = db._collection.update.call_args
    written_metas = call_args.kwargs["metadatas"]
    assert written_metas[0]["topic_id"] == "topic_rag"
    assert written_metas[0]["source"]   == "file.pdf"   # original field preserved
    assert written_metas[1]["topic_id"] == "topic_agents"


# ---------------------------------------------------------------------------
# Test 5: TopicClusterer.fit_and_assign — end-to-end
# ---------------------------------------------------------------------------

def test_fit_and_assign() -> None:
    from rag.knowledge.clusterer import TopicClusterer

    db, ids, _ = _make_fake_chroma(n_docs=6, dim=4)
    # Patch assign to verify it's called
    db.get.side_effect = [
        # First call from fit()
        {"ids": ids, "embeddings": [[float(i)] * 4 for i in range(6)], "documents": ["t"] * 6},
        # Second call from assign() — get metadatas
        {"metadatas": [{"doc_id": "x"}] * 6},
    ]
    llm = _make_mock_llm("agents")

    clusterer = TopicClusterer(llm=llm, db=db, random_state=0)
    topic_map = clusterer.fit_and_assign(n_clusters=2)

    assert topic_map.n_clusters == 2
    assert db._collection.update.called


# ---------------------------------------------------------------------------
# Test 6: KnowledgeManager.cluster_topics delegates to clusterer
# ---------------------------------------------------------------------------

def test_manager_cluster_topics() -> None:
    from rag.knowledge.manager import KnowledgeManager

    mock_clusterer = MagicMock()
    mock_topic_map = MagicMock()
    mock_clusterer.fit_and_assign.return_value = mock_topic_map

    mgr = KnowledgeManager(
        db=MagicMock(), qa_db=MagicMock(), qa_indexer=MagicMock(),
        extractor=MagicMock(), qa_generator=MagicMock(),
        clusterer=mock_clusterer,
    )
    result = mgr.cluster_topics(n_clusters=5, doc_id="my_doc")

    mock_clusterer.fit_and_assign.assert_called_once_with(n_clusters=5, doc_id="my_doc")
    assert result is mock_topic_map


# ---------------------------------------------------------------------------
# Test 7: KnowledgeManager.cluster_topics raises without clusterer
# ---------------------------------------------------------------------------

def test_manager_no_clusterer_raises() -> None:
    from rag.knowledge.manager import KnowledgeManager

    mgr = KnowledgeManager(
        db=MagicMock(), qa_db=MagicMock(), qa_indexer=MagicMock(),
        extractor=MagicMock(), qa_generator=MagicMock(),
        clusterer=None,
    )
    try:
        mgr.cluster_topics()
        _fail("KnowledgeManager.cluster_topics (no clusterer)", Exception("expected RuntimeError"))
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Test 8 (live): real Chroma + real LLM round-trip
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_live_round_trip() -> None:
    import os
    from utils.config import AppConfig
    from rag.client import LocalLlamaClient

    config = AppConfig("etc/config.yaml")
    client = LocalLlamaClient(config)

    doc_ids = client.list_doc_ids()
    if not doc_ids:
        print("Live test skipped (no documents indexed)")
        return

    topic_map = client.cluster_topics(n_clusters=min(5, len(doc_ids)))
    assert topic_map.n_chunks > 0
    assert len(topic_map.cluster_labels) <= 5
    for label in topic_map.cluster_labels.values():
        assert label.startswith("topic_"), f"Bad label: {label}"

    print(f"Live test: {topic_map.n_chunks} chunks → {topic_map.n_clusters} topics")
    for c_int, t_id in sorted(topic_map.cluster_labels.items()):
        count = sum(1 for v in topic_map.chunk_topics.values() if v == t_id)
        print(f"  cluster {c_int}: {t_id}  ({count} chunks)")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------


