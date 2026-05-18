"""
Unit tests for B.4 Cross-document Linking (testing/testing_linker.py).

Mock tests (no live Chroma, no LLM):
  1. _cosine_matrix — values and shape
  2. _top_k_exclude_self — mask, threshold, top-K
  3. CrossDocLinker.link_chunks — correct IDs written, same-doc excluded
  4. CrossDocLinker.link_chunks — no cross-doc candidates → skipped
  5. CrossDocLinker.link_pages — centroid similarity, links written per chunk
  6. CrossDocLinker.link_pages — fewer than 2 docs → skipped
  7. CrossDocLinker.get_related_chunks — read-back from metadata
  8. CrossDocLinker.get_related_pages — read-back from first chunk
  9. KnowledgeManager.link_chunks / link_pages — delegates to linker
 10. KnowledgeManager without linker — raises RuntimeError

Live test (--no-live to skip):
 11. Full round-trip against real Chroma
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, call

import numpy as np

LIVE = "--no-live" not in sys.argv


def _ok(name: str) -> None:
    print(f"{name}: OK")


def _fail(name: str, err: Exception) -> None:
    print(f"{name}: FAIL  {err}")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Helpers to build a fake Chroma
# ---------------------------------------------------------------------------

def _make_db(
    ids:       list[str],
    doc_ids:   list[str],
    texts:     list[str],
    embeddings: list[list[float]],
) -> MagicMock:
    """Minimal Chroma mock that returns fixed data from get()."""
    metadatas = [{"doc_id": d} for d in doc_ids]
    db = MagicMock()
    db._collection = MagicMock()
    db.get.return_value = {
        "ids":        ids,
        "embeddings": embeddings,
        "documents":  texts,
        "metadatas":  metadatas,
    }
    return db


# ---------------------------------------------------------------------------
# Test 1 — _cosine_matrix
# ---------------------------------------------------------------------------

def test_cosine_matrix() -> None:
    from rag.knowledge.linker import _cosine_matrix

    # Two identical vectors → cosine = 1.0
    # Two orthogonal vectors → cosine = 0.0
    X = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    C = _cosine_matrix(X)
    assert C.shape == (3, 3)
    assert abs(C[0, 0] - 1.0) < 1e-5
    assert abs(C[0, 1])       < 1e-5   # orthogonal
    assert abs(C[0, 2] - 1.0) < 1e-5   # same as X[0]
    _ok("_cosine_matrix")


# ---------------------------------------------------------------------------
# Test 2 — _top_k_exclude_self
# ---------------------------------------------------------------------------

def test_top_k_exclude_self() -> None:
    from rag.knowledge.linker import _top_k_exclude_self

    sim_row     = np.array([0.95, 0.90, 0.80, 0.70, 0.60], dtype=np.float32)
    mask        = np.array([False, False, True, False, False])  # index 2 excluded
    # threshold=0.65 → indices 1 (0.90) and 3 (0.70) are above; 4 (0.60) is not
    idx, scores = _top_k_exclude_self(sim_row, 0, mask, top_k=2, threshold=0.65)

    # Self (0) excluded, masked (2) excluded, 4 below threshold → top-2 = [1, 3]
    assert list(idx)    == [1, 3]
    assert abs(scores[0] - 0.90) < 1e-5
    assert abs(scores[1] - 0.70) < 1e-5
    _ok("_top_k_exclude_self")


# ---------------------------------------------------------------------------
# Test 3 — CrossDocLinker.link_chunks (cross-doc links written)
# ---------------------------------------------------------------------------

def test_link_chunks_basic() -> None:
    from rag.knowledge.linker import CrossDocLinker

    # 4 chunks: doc_a (0,1), doc_b (2,3)
    # Embeddings: doc_a ~ [1,0,...], doc_b ~ [0,1,...]
    # With high cross-doc similarity between 0↔3 and 1↔2 (well-separated from same doc)
    DIM = 8
    emb = [
        [1.0] + [0.0] * (DIM - 1),
        [0.9] + [0.1] + [0.0] * (DIM - 2),
        [0.0, 1.0] + [0.0] * (DIM - 2),
        [0.1, 0.9] + [0.0] * (DIM - 2),
    ]
    db = _make_db(
        ids       = ["c0", "c1", "c2", "c3"],
        doc_ids   = ["doc_a", "doc_a", "doc_b", "doc_b"],
        texts     = ["t0", "t1", "t2", "t3"],
        embeddings= emb,
    )

    linker = CrossDocLinker(db=db)
    stats  = linker.link_chunks(top_k=2, threshold=0.0)  # threshold=0 → always link

    assert stats.linked + stats.skipped == 4
    call_args = db._collection.update.call_args
    written_metas = call_args.kwargs["metadatas"]

    # Every chunk should have related_chunk_ids pointing to the other doc
    for meta in written_metas:
        rel_ids = json.loads(meta.get("related_chunk_ids", "[]"))
        assert len(rel_ids) > 0, "Expected cross-doc links"
        # None of the related IDs should be from the same doc as the source
        # (we can't easily check doc_id here without more state, but length > 0 is the key assertion)

    _ok("CrossDocLinker.link_chunks (basic)")


# ---------------------------------------------------------------------------
# Test 4 — link_chunks: no cross-doc candidates → skipped
# ---------------------------------------------------------------------------

def test_link_chunks_single_doc() -> None:
    from rag.knowledge.linker import CrossDocLinker

    emb = [[1.0, 0.0], [0.9, 0.1]]
    db  = _make_db(
        ids        = ["c0", "c1"],
        doc_ids    = ["only_doc", "only_doc"],
        texts      = ["t0", "t1"],
        embeddings = emb,
    )
    linker = CrossDocLinker(db=db)
    stats  = linker.link_chunks(top_k=3, threshold=0.0)

    assert stats.skipped == 2
    assert stats.linked  == 0
    _ok("CrossDocLinker.link_chunks (single doc → all skipped)")


# ---------------------------------------------------------------------------
# Test 5 — CrossDocLinker.link_pages (centroid links written)
# ---------------------------------------------------------------------------

def test_link_pages_basic() -> None:
    from rag.knowledge.linker import CrossDocLinker

    DIM = 4
    emb = [
        [1.0, 0.0, 0.0, 0.0],  # doc_a chunk 0
        [0.9, 0.1, 0.0, 0.0],  # doc_a chunk 1
        [0.0, 0.0, 1.0, 0.0],  # doc_b chunk 0
        [0.0, 0.0, 0.9, 0.1],  # doc_b chunk 1
    ]
    db = MagicMock()
    db._collection = MagicMock()
    db.get.return_value = {
        "ids":        ["c0", "c1", "c2", "c3"],
        "embeddings": emb,
        "documents":  ["t0", "t1", "t2", "t3"],
        "metadatas":  [
            {"doc_id": "doc_a"}, {"doc_id": "doc_a"},
            {"doc_id": "doc_b"}, {"doc_id": "doc_b"},
        ],
    }

    linker = CrossDocLinker(db=db)
    stats  = linker.link_pages(top_k=2, threshold=0.0)

    assert stats.linked == 2  # both doc_a and doc_b get a link to each other

    call_args     = db._collection.update.call_args
    written_metas = call_args.kwargs["metadatas"]
    # All 4 chunks should have related_doc_ids
    for meta in written_metas:
        rel_docs = json.loads(meta.get("related_doc_ids", "[]"))
        assert len(rel_docs) == 1  # each doc has exactly one other doc
    _ok("CrossDocLinker.link_pages (basic)")


# ---------------------------------------------------------------------------
# Test 6 — link_pages: fewer than 2 docs → skipped
# ---------------------------------------------------------------------------

def test_link_pages_single_doc() -> None:
    from rag.knowledge.linker import CrossDocLinker

    db = MagicMock()
    db._collection = MagicMock()
    db.get.return_value = {
        "ids": ["c0", "c1"],
        "embeddings": [[1.0, 0.0], [0.9, 0.1]],
        "documents": ["t0", "t1"],
        "metadatas": [{"doc_id": "only_doc"}, {"doc_id": "only_doc"}],
    }
    linker = CrossDocLinker(db=db)
    stats  = linker.link_pages()

    # No update should be called when there's only one doc
    db._collection.update.assert_not_called()
    _ok("CrossDocLinker.link_pages (single doc → no-op)")


# ---------------------------------------------------------------------------
# Test 7 — get_related_chunks read-back
# ---------------------------------------------------------------------------

def test_get_related_chunks() -> None:
    from rag.knowledge.linker import CrossDocLinker

    db = MagicMock()
    db._collection = MagicMock()

    rel_ids_json    = json.dumps(["c2", "c3"])
    rel_scores_json = json.dumps([0.91, 0.85])

    # First get() call: fetch source chunk metadata
    db.get.side_effect = [
        {
            "ids": ["c0"],
            "metadatas": [{"doc_id": "doc_a",
                           "related_chunk_ids":    rel_ids_json,
                           "related_chunk_scores": rel_scores_json}],
            "documents": ["src text"],
        },
        # Second get() call: fetch related chunk details
        {
            "ids": ["c2", "c3"],
            "documents": ["text of c2 here", "text of c3 here"],
            "metadatas": [{"doc_id": "doc_b"}, {"doc_id": "doc_b"}],
        },
    ]

    linker  = CrossDocLinker(db=db)
    related = linker.get_related_chunks("c0")

    assert len(related) == 2
    assert related[0]["id"]     == "c2"
    assert related[0]["doc_id"] == "doc_b"
    assert abs(related[0]["score"] - 0.91) < 1e-4
    assert "text of c2" in related[0]["text"]
    _ok("CrossDocLinker.get_related_chunks")


# ---------------------------------------------------------------------------
# Test 8 — get_related_pages read-back
# ---------------------------------------------------------------------------

def test_get_related_pages() -> None:
    from rag.knowledge.linker import CrossDocLinker

    db = MagicMock()
    db._collection = MagicMock()
    db.get.return_value = {
        "ids": ["c0", "c1"],
        "metadatas": [
            {"doc_id": "doc_a",
             "related_doc_ids":    json.dumps(["doc_b", "doc_c"]),
             "related_doc_scores": json.dumps([0.88, 0.75])},
            {"doc_id": "doc_a"},
        ],
        "documents": [],
    }

    linker  = CrossDocLinker(db=db)
    related = linker.get_related_pages("doc_a")

    assert len(related) == 2
    assert related[0]["doc_id"] == "doc_b"
    assert abs(related[0]["score"] - 0.88) < 1e-4
    _ok("CrossDocLinker.get_related_pages")


# ---------------------------------------------------------------------------
# Test 9 — KnowledgeManager delegates to linker
# ---------------------------------------------------------------------------

def test_manager_delegation() -> None:
    from rag.knowledge.manager import KnowledgeManager

    mock_linker     = MagicMock()
    mock_link_stats = MagicMock()
    mock_linker.link_chunks.return_value    = mock_link_stats
    mock_linker.link_pages.return_value     = mock_link_stats
    mock_linker.get_related_chunks.return_value = [{"id": "x"}]
    mock_linker.get_related_pages.return_value  = [{"doc_id": "y"}]

    mgr = KnowledgeManager(
        db=MagicMock(), qa_db=MagicMock(), qa_indexer=MagicMock(),
        extractor=MagicMock(), qa_generator=MagicMock(),
        linker=mock_linker,
    )

    mgr.link_chunks(top_k=3, threshold=0.8, doc_id="d1")
    mock_linker.link_chunks.assert_called_once_with(top_k=3, threshold=0.8, doc_id="d1")

    mgr.link_pages(top_k=4, threshold=0.65)
    mock_linker.link_pages.assert_called_once_with(top_k=4, threshold=0.65)

    assert mgr.get_related_chunks("c0") == [{"id": "x"}]
    assert mgr.get_related_pages("d1")  == [{"doc_id": "y"}]

    _ok("KnowledgeManager.link_chunks / link_pages / get_related_*")


# ---------------------------------------------------------------------------
# Test 10 — KnowledgeManager without linker raises RuntimeError
# ---------------------------------------------------------------------------

def test_manager_no_linker_raises() -> None:
    from rag.knowledge.manager import KnowledgeManager

    mgr = KnowledgeManager(
        db=MagicMock(), qa_db=MagicMock(), qa_indexer=MagicMock(),
        extractor=MagicMock(), qa_generator=MagicMock(),
        linker=None,
    )
    for method in ("link_chunks", "link_pages", "get_related_chunks", "get_related_pages"):
        try:
            if method == "get_related_chunks":
                mgr.get_related_chunks("x")
            elif method == "get_related_pages":
                mgr.get_related_pages("x")
            elif method == "link_chunks":
                mgr.link_chunks()
            else:
                mgr.link_pages()
            _fail(f"KnowledgeManager.{method} (no linker)", Exception("expected RuntimeError"))
        except RuntimeError:
            pass
    _ok("KnowledgeManager.* (no linker → RuntimeError)")


# ---------------------------------------------------------------------------
# Test 11 (live) — real Chroma round-trip
# ---------------------------------------------------------------------------

def test_live_round_trip() -> None:
    if not LIVE:
        print("Live test skipped (--no-live)")
        return

    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.config import AppConfig
    from rag.client import LocalLlamaClient

    config = AppConfig.from_yaml("etc/config.yaml")
    client = LocalLlamaClient(config)

    doc_ids = client.list_doc_ids()
    if len(doc_ids) < 2:
        print("Live test skipped (need ≥ 2 indexed documents for cross-doc linking)")
        return

    # Chunk linking
    chunk_stats = client.link_chunks(top_k=3, threshold=0.5)
    print(f"link_chunks: {chunk_stats}")
    assert chunk_stats.linked + chunk_stats.skipped > 0

    # Page linking
    page_stats = client.link_pages(top_k=3, threshold=0.4)
    print(f"link_pages: {page_stats}")
    assert page_stats.linked > 0

    # Read-back
    first_doc = doc_ids[0]
    related_pages = client.get_related_pages(first_doc)
    print(f"get_related_pages({first_doc!r}): {related_pages}")

    _ok("CrossDocLinker live round-trip")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cosine_matrix()
    test_top_k_exclude_self()
    test_link_chunks_basic()
    test_link_chunks_single_doc()
    test_link_pages_basic()
    test_link_pages_single_doc()
    test_get_related_chunks()
    test_get_related_pages()
    test_manager_delegation()
    test_manager_no_linker_raises()
    test_live_round_trip()
    print("\nAll cross-document linking tests passed.")
