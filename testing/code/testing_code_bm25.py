"""
Tests for Phase U3 Step 3.1 — BM25 for Code.

Covers:
  code_tokenize
  - basic camelCase splitting
  - snake_case splitting
  - leading/trailing underscores stripped
  - dot and :: separators
  - brackets stripped
  - numbers preserved inside tokens (BM25Index)
  - mixed prose + identifiers
  - empty string returns empty list

  BM25Index._tokenize_text delegation
  - existing tokenizer still works via _tokenize_text

  CodeBM25Index
  - uses code_tokenize via _tokenize_text
  - matches "rag" query against "RAGEngine" document
  - does NOT match "ragengine" literally in prose tokenizer (demonstrates the difference)
  - build + search smoke test

  CodeRetriever hybrid flag
  - use_hybrid=False → no _bm25 attribute set (None)
  - use_hybrid=True  → _bm25 is CodeBM25Index instance
  - invalidate_bm25() sets _bm25_built = False
  - search() without hybrid calls search_with_scores (mock)
  - search() with hybrid calls rrf_fuse path (mock)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# code_tokenize
# ---------------------------------------------------------------------------

def test_camel_case():
    from rag.code.tokenizer import code_tokenize
    assert code_tokenize("RAGEngine") == ["rag", "engine"]


def test_camel_case_with_number():
    from rag.code.tokenizer import code_tokenize
    result = code_tokenize("BM25Index")
    assert "bm" in result and "25" in result and "index" in result


def test_snake_case():
    from rag.code.tokenizer import code_tokenize
    assert code_tokenize("content_hash") == ["content", "hash"]


def test_leading_underscores_stripped():
    from rag.code.tokenizer import code_tokenize
    result = code_tokenize("_DEFAULT_COLLECTION_NAMES")
    assert result == ["default", "collection", "names"]


def test_dunder_stripped():
    from rag.code.tokenizer import code_tokenize
    result = code_tokenize("__init__")
    assert result == ["init"]


def test_dot_separator():
    from rag.code.tokenizer import code_tokenize
    result = code_tokenize("RAGEngine.retrieve")
    assert result == ["rag", "engine", "retrieve"]


def test_double_colon_separator():
    from rag.code.tokenizer import code_tokenize
    result = code_tokenize("SearchFilter::to_chroma")
    assert "search" in result and "filter" in result
    assert "to" in result and "chroma" in result


def test_brackets_stripped():
    from rag.code.tokenizer import code_tokenize
    result = code_tokenize("index()")
    assert result == ["index"]


def test_all_caps_word():
    from rag.code.tokenizer import code_tokenize
    result = code_tokenize("RAG")
    assert result == ["rag"]


def test_mixed_prose_and_code():
    from rag.code.tokenizer import code_tokenize
    result = code_tokenize("parse Python AST chunks")
    assert "parse" in result and "python" in result and "ast" in result and "chunks" in result


def test_empty_string():
    from rag.code.tokenizer import code_tokenize
    assert code_tokenize("") == []


def test_already_lowercase():
    from rag.code.tokenizer import code_tokenize
    result = code_tokenize("retrieve")
    assert result == ["retrieve"]


# ---------------------------------------------------------------------------
# BM25Index default tokenizer still works
# ---------------------------------------------------------------------------

def test_bm25_index_default_tokenizer():
    from rag.retrieval.bm25 import BM25Index
    idx = BM25Index()  # no tokenizer arg → default prose tokenizer
    result = idx._tokenizer("hello world")
    assert "hello" in result and "world" in result


def test_bm25_index_custom_tokenizer_used():
    """BM25Index with code_tokenize tokenizer should split camelCase."""
    from rag.retrieval.bm25 import BM25Index
    from rag.code.tokenizer import code_tokenize
    idx = BM25Index(tokenizer=code_tokenize)
    result = idx._tokenizer("RAGEngine.retrieve")
    assert "rag" in result and "engine" in result and "retrieve" in result


def test_code_bm25_matches_camel_query():
    """Query 'rag engine' should match a document containing 'RAGEngine'."""
    from langchain_core.documents import Document
    from rag.retrieval.bm25 import BM25Index
    from rag.code.tokenizer import code_tokenize

    idx = BM25Index(tokenizer=code_tokenize)
    docs = [
        Document(page_content="class RAGEngine: pass", metadata={}),
        Document(page_content="def unrelated_function(): pass", metadata={}),
        Document(page_content="some other utility file here", metadata={}),
    ]
    idx.build(docs)

    results = idx.search("rag engine", k=5)
    assert len(results) >= 1
    assert "RAGEngine" in results[0][0].page_content


def test_code_bm25_matches_snake_query():
    from langchain_core.documents import Document
    from rag.retrieval.bm25 import BM25Index
    from rag.code.tokenizer import code_tokenize

    idx = BM25Index(tokenizer=code_tokenize)
    docs = [
        Document(page_content="content_hash comparison logic", metadata={}),
        Document(page_content="completely different text here", metadata={}),
        Document(page_content="another unrelated document", metadata={}),
    ]
    idx.build(docs)

    results = idx.search("content hash", k=5)
    assert len(results) >= 1
    assert "content_hash" in results[0][0].page_content


def test_code_bm25_empty_index_returns_empty():
    from rag.retrieval.bm25 import BM25Index
    from rag.code.tokenizer import code_tokenize
    idx = BM25Index(tokenizer=code_tokenize)
    assert idx.search("anything") == []


def test_code_bm25_exact_identifier_query():
    """Query 'RAGEngine' should match via camelCase split."""
    from langchain_core.documents import Document
    from rag.retrieval.bm25 import BM25Index
    from rag.code.tokenizer import code_tokenize

    idx = BM25Index(tokenizer=code_tokenize)
    docs = [
        Document(page_content="RAGEngine is the main answer engine", metadata={}),
        Document(page_content="some other text about parsers", metadata={}),
        Document(page_content="file processing utilities", metadata={}),
    ]
    idx.build(docs)

    results = idx.search("RAGEngine", k=5)
    assert len(results) >= 1
    assert "RAGEngine" in results[0][0].page_content


# ---------------------------------------------------------------------------
# CodeRetriever hybrid flag behaviour
# ---------------------------------------------------------------------------

def _make_code_retriever(use_hybrid=False):
    from rag.retrieval.code_retriever import CodeRetriever
    fake_indexer = MagicMock()
    return CodeRetriever(fake_indexer, level="symbol", use_hybrid=use_hybrid)


def test_code_retriever_no_hybrid_bm25_is_none():
    r = _make_code_retriever(use_hybrid=False)
    assert r._bm25 is None


def test_code_retriever_hybrid_bm25_is_bm25index():
    from rag.retrieval.bm25 import BM25Index
    r = _make_code_retriever(use_hybrid=True)
    assert isinstance(r._bm25, BM25Index)


def test_invalidate_bm25_sets_flag_false():
    r = _make_code_retriever(use_hybrid=True)
    r._bm25_built = True
    r.invalidate_bm25()
    assert r._bm25_built is False


def test_search_vector_only_calls_search_with_scores():
    from langchain_core.documents import Document
    from rag.retrieval.code_retriever import CodeRetriever

    fake_indexer = MagicMock()
    fake_indexer.search_with_scores.return_value = [
        (Document(page_content="x", metadata={}), 0.9),
    ]
    r = CodeRetriever(fake_indexer, level="symbol", use_hybrid=False)
    results = r.search("query", top_k=5)

    fake_indexer.search_with_scores.assert_called_once()
    assert len(results) == 1
    assert results[0].source == "code"


def test_search_hybrid_calls_both_and_fuses():
    """Hybrid search should call search_with_scores AND bm25.search."""
    from langchain_core.documents import Document
    from rag.retrieval.code_retriever import CodeRetriever
    from rag.retrieval.bm25 import BM25Index

    fake_indexer = MagicMock()
    fake_indexer.search_with_scores.return_value = [
        (Document(page_content="vector result", metadata={}), 0.9),
    ]

    r = CodeRetriever(fake_indexer, level="symbol", use_hybrid=True)
    r._bm25_built = True  # skip actual build
    r._bm25 = MagicMock(spec=BM25Index)
    r._bm25.search.return_value = [
        (Document(page_content="bm25 result", metadata={}), 2.5),
    ]

    results = r.search("RAGEngine", top_k=5)

    fake_indexer.search_with_scores.assert_called_once()
    r._bm25.search.assert_called_once()
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------


