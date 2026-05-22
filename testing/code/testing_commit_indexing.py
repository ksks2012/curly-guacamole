"""
Tests for GCR2.4 — Commit Semantic Indexing.

Covers:
  CommitRecord schema
  - defaults: summary="", content_hash="", affected_symbols=[], files_changed=[]
  - to_dict / from_dict roundtrip (with list fields)
  - from_dict backward compat (CSV strings for list fields)
  - to_document: page_content includes summary and affected symbols
  - to_document: metadata keys present (commit_hash, repo_id, source_type, content_hash)
  - short_hash property

  CommitRecord._commit_record_id
  - deterministic (same input → same output)
  - different commits → different IDs

  derive_affected_symbols
  - symbol introduced in commit → included
  - symbol modified in commit → included
  - symbol deleted in commit → included
  - symbol not touched → excluded
  - empty evolutions → []
  - deduplication when introduced and modified in same commit

  CommitAnalyzer.build
  - affected_symbols derived correctly (no LLM)
  - summary from mock LLM stored in record
  - LLM failure returns "" summary, content_hash still set
  - no LLM (None) → summary = ""
  - content_hash = sha256(summary) when summary present
  - content_hash = sha256(message) when summary empty

  CommitIndexer
  - upsert adds new records
  - upsert is idempotent (same content_hash → skipped)
  - upsert DO UPDATE when content_hash changes
  - delete_repo removes only target repo
  - count() returns correct number
  - search() returns records (mock embedding)
  - prune_missing=True removes absent records for same repo
"""

from __future__ import annotations

import hashlib


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REPO_ID = "test-repo"
CA = "a" * 40
CB = "b" * 40
CC = "c" * 40


def _commit_info(commit_hash=CA, message="init commit", files=None):
    from rag.code.schema import CommitInfo
    return CommitInfo(
        commit_hash=commit_hash,
        author="Alice",
        date="2024-01-01T00:00:00+00:00",
        message=message,
        files_changed=files or ["rag/engine.py"],
    )


def _evolution(symbol_name, introduced=CA, modified=None, deleted=""):
    from rag.code.schema import SymbolEvolution, _evolution_id
    return SymbolEvolution(
        evolution_id=_evolution_id(REPO_ID, "rag/engine.py", symbol_name),
        symbol_name=symbol_name,
        repo_id=REPO_ID,
        file_path="rag/engine.py",
        introduced_in=introduced,
        modified_in=modified or [],
        deleted_in=deleted,
    )


def _commit_record(**kwargs):
    from rag.code.schema import CommitRecord, _commit_record_id
    defaults = dict(
        commit_id=_commit_record_id(REPO_ID, CA),
        repo_id=REPO_ID,
        commit_hash=CA,
        author="Alice",
        date="2024-01-01T00:00:00+00:00",
        message="init commit",
        summary="introduced retrieval engine",
        content_hash=hashlib.sha256(b"introduced retrieval engine").hexdigest(),
    )
    defaults.update(kwargs)
    return CommitRecord(**defaults)


# ---------------------------------------------------------------------------
# CommitRecord schema
# ---------------------------------------------------------------------------

def test_commit_record_defaults():
    from rag.code.schema import CommitRecord, _commit_record_id
    r = CommitRecord(
        commit_id=_commit_record_id(REPO_ID, CA),
        repo_id=REPO_ID, commit_hash=CA,
        author="Alice", date="2024-01-01", message="init",
    )
    assert r.summary          == ""
    assert r.content_hash     == ""
    assert r.affected_symbols == []
    assert r.files_changed    == []


def test_commit_record_roundtrip():
    r = _commit_record(affected_symbols=["foo", "bar"], files_changed=["a.py"])
    from rag.code.schema import CommitRecord
    r2 = CommitRecord.from_dict(r.to_dict())
    assert r2 == r


def test_commit_record_from_dict_csv_strings():
    from rag.code.schema import CommitRecord, _commit_record_id
    d = {
        "commit_id":        _commit_record_id(REPO_ID, CA),
        "repo_id":          REPO_ID,
        "commit_hash":      CA,
        "author":           "Alice",
        "date":             "2024-01-01",
        "message":          "init",
        "files_changed":    "a.py,b.py",       # CSV string (from Chroma metadata)
        "affected_symbols": "foo,bar.baz",
    }
    r = CommitRecord.from_dict(d)
    assert r.files_changed    == ["a.py", "b.py"]
    assert r.affected_symbols == ["foo", "bar.baz"]


def test_commit_record_to_document_content():
    r = _commit_record(affected_symbols=["foo", "bar"])
    doc = r.to_document()
    assert "introduced retrieval engine" in doc.page_content
    assert "foo" in doc.page_content
    assert "bar" in doc.page_content


def test_commit_record_to_document_metadata_keys():
    r = _commit_record()
    meta = r.to_document().metadata
    for key in ("commit_hash", "repo_id", "author", "date", "source_type", "content_hash"):
        assert key in meta, f"missing metadata key: {key}"
    assert meta["source_type"] == "commit"


def test_commit_record_short_hash():
    r = _commit_record()
    assert r.short_hash == CA[:12]


def test_commit_record_no_summary_falls_back_to_message_in_doc():
    from rag.code.schema import CommitRecord, _commit_record_id
    r = CommitRecord(
        commit_id=_commit_record_id(REPO_ID, CA),
        repo_id=REPO_ID, commit_hash=CA,
        author="Alice", date="2024-01-01",
        message="add reranking support",
        summary="",
    )
    doc = r.to_document()
    assert "add reranking support" in doc.page_content


# ---------------------------------------------------------------------------
# _commit_record_id
# ---------------------------------------------------------------------------

def test_commit_record_id_deterministic():
    from rag.code.schema import _commit_record_id
    assert _commit_record_id(REPO_ID, CA) == _commit_record_id(REPO_ID, CA)


def test_commit_record_id_unique_per_commit():
    from rag.code.schema import _commit_record_id
    assert _commit_record_id(REPO_ID, CA) != _commit_record_id(REPO_ID, CB)


# ---------------------------------------------------------------------------
# derive_affected_symbols
# ---------------------------------------------------------------------------

def test_derive_introduced_in():
    from rag.code.commit_analyzer import derive_affected_symbols
    evos = [_evolution("foo", introduced=CA)]
    assert "foo" in derive_affected_symbols(CA, evos)


def test_derive_modified_in():
    from rag.code.commit_analyzer import derive_affected_symbols
    evos = [_evolution("foo", introduced=CA, modified=[CB])]
    assert "foo" in derive_affected_symbols(CB, evos)


def test_derive_deleted_in():
    from rag.code.commit_analyzer import derive_affected_symbols
    evos = [_evolution("foo", introduced=CA, deleted=CB)]
    assert "foo" in derive_affected_symbols(CB, evos)


def test_derive_not_touched():
    from rag.code.commit_analyzer import derive_affected_symbols
    evos = [_evolution("foo", introduced=CA)]
    assert derive_affected_symbols(CB, evos) == []


def test_derive_empty_evolutions():
    from rag.code.commit_analyzer import derive_affected_symbols
    assert derive_affected_symbols(CA, []) == []


def test_derive_dedup_when_introduced_and_modified():
    from rag.code.commit_analyzer import derive_affected_symbols
    # same symbol both introduced AND listed in modified_in (edge case)
    evos = [_evolution("foo", introduced=CA, modified=[CA])]
    result = derive_affected_symbols(CA, evos)
    assert result.count("foo") == 1


def test_derive_sorted():
    from rag.code.commit_analyzer import derive_affected_symbols
    evos = [
        _evolution("zoo", introduced=CA),
        _evolution("alpha", introduced=CA),
        _evolution("mid", introduced=CA),
    ]
    result = derive_affected_symbols(CA, evos)
    assert result == sorted(result)


# ---------------------------------------------------------------------------
# CommitAnalyzer.build
# ---------------------------------------------------------------------------

class _MockLLM:
    def __init__(self, text: str) -> None:
        self._text = text
    def invoke(self, _prompt: str):
        return type("Msg", (), {"content": self._text})()


class _FailingLLM:
    def invoke(self, _prompt: str):
        raise RuntimeError("LLM unavailable")


def test_analyzer_build_affected_symbols():
    from rag.code.commit_analyzer import CommitAnalyzer
    evos = [
        _evolution("foo", introduced=CA),
        _evolution("bar", introduced=CB),  # different commit
    ]
    analyzer = CommitAnalyzer(llm=None)
    record = analyzer.build(_commit_info(CA), evos, REPO_ID)
    assert record.affected_symbols == ["foo"]


def test_analyzer_build_summary_from_mock_llm():
    from rag.code.commit_analyzer import CommitAnalyzer
    evos = [_evolution("foo", introduced=CA)]
    analyzer = CommitAnalyzer(_MockLLM("introduced the foo retrieval class"))
    record = analyzer.build(_commit_info(CA), evos, REPO_ID)
    assert record.summary == "introduced the foo retrieval class"


def test_analyzer_build_no_llm_summary_empty():
    from rag.code.commit_analyzer import CommitAnalyzer
    analyzer = CommitAnalyzer(llm=None)
    record = analyzer.build(_commit_info(CA), [], REPO_ID)
    assert record.summary == ""


def test_analyzer_build_llm_failure_summary_empty():
    from rag.code.commit_analyzer import CommitAnalyzer
    analyzer = CommitAnalyzer(_FailingLLM())
    record = analyzer.build(_commit_info(CA), [], REPO_ID)
    assert record.summary == ""


def test_analyzer_build_content_hash_uses_summary():
    from rag.code.commit_analyzer import CommitAnalyzer
    summary = "introduced retrieval pipeline"
    analyzer = CommitAnalyzer(_MockLLM(summary))
    record = analyzer.build(_commit_info(CA), [], REPO_ID)
    expected = hashlib.sha256(summary.encode()).hexdigest()
    assert record.content_hash == expected


def test_analyzer_build_content_hash_falls_back_to_message():
    from rag.code.commit_analyzer import CommitAnalyzer
    ci = _commit_info(CA, message="add reranker")
    analyzer = CommitAnalyzer(llm=None)
    record = analyzer.build(ci, [], REPO_ID)
    expected = hashlib.sha256(b"add reranker").hexdigest()
    assert record.content_hash == expected


def test_analyzer_build_multiline_llm_first_line_only():
    from rag.code.commit_analyzer import CommitAnalyzer
    analyzer = CommitAnalyzer(_MockLLM("first line\nsecond line"))
    record = analyzer.build(_commit_info(CA), [], REPO_ID)
    assert record.summary == "first line"


def test_analyzer_build_repo_id_and_commit_id():
    from rag.code.commit_analyzer import CommitAnalyzer
    from rag.code.schema import _commit_record_id
    analyzer = CommitAnalyzer(llm=None)
    record = analyzer.build(_commit_info(CA), [], REPO_ID)
    assert record.repo_id   == REPO_ID
    assert record.commit_id == _commit_record_id(REPO_ID, CA)


# ---------------------------------------------------------------------------
# CommitIndexer — requires Chroma + fake embeddings
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    """Deterministic fake embedding — no model needed."""
    def embed_documents(self, texts):
        return [self._hash_vec(t) for t in texts]
    def embed_query(self, text):
        return self._hash_vec(text)
    @staticmethod
    def _hash_vec(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:16]]  # 16-dim vector


def _make_indexer(tmp_path, embed=None):
    from rag.code.commit_indexer import CommitIndexer
    return CommitIndexer(str(tmp_path), embed or _FakeEmbeddings())


def test_commit_indexer_upsert_adds(tmp_path):
    idx = _make_indexer(tmp_path)
    r = _commit_record()
    stats = idx.upsert([r], repo_id=REPO_ID)
    assert stats.added == 1
    assert idx.count(REPO_ID) == 1


def test_commit_indexer_upsert_idempotent(tmp_path):
    idx = _make_indexer(tmp_path)
    r = _commit_record()
    idx.upsert([r], repo_id=REPO_ID)
    stats = idx.upsert([r], repo_id=REPO_ID)
    assert stats.added   == 0
    assert stats.skipped == 1


def test_commit_indexer_upsert_updates_on_hash_change(tmp_path):
    idx = _make_indexer(tmp_path)
    r1 = _commit_record(summary="old summary",
                         content_hash=hashlib.sha256(b"old summary").hexdigest())
    idx.upsert([r1], repo_id=REPO_ID)

    r2 = _commit_record(summary="new summary",
                         content_hash=hashlib.sha256(b"new summary").hexdigest())
    stats = idx.upsert([r2], repo_id=REPO_ID)
    assert stats.updated == 1


def test_commit_indexer_prune_missing(tmp_path):
    idx = _make_indexer(tmp_path)
    r1 = _commit_record(commit_hash=CA,
                         commit_id=hashlib.sha256(f"{REPO_ID}|{CA}".encode()).hexdigest(),
                         content_hash=hashlib.sha256(b"a").hexdigest())
    r2 = _commit_record(commit_hash=CB,
                         commit_id=hashlib.sha256(f"{REPO_ID}|{CB}".encode()).hexdigest(),
                         content_hash=hashlib.sha256(b"b").hexdigest())
    idx.upsert([r1, r2], repo_id=REPO_ID)
    assert idx.count(REPO_ID) == 2

    # Second upsert with only r1 → r2 pruned
    stats = idx.upsert([r1], repo_id=REPO_ID, prune_missing=True)
    assert stats.deleted == 1
    assert idx.count(REPO_ID) == 1


def test_commit_indexer_delete_repo(tmp_path):
    idx = _make_indexer(tmp_path)
    idx.upsert([_commit_record()], repo_id=REPO_ID)

    # Add a second repo
    from rag.code.schema import _commit_record_id
    other = _commit_record(
        commit_id=_commit_record_id("other", CB),
        repo_id="other",
        commit_hash=CB,
        content_hash=hashlib.sha256(b"other").hexdigest(),
    )
    idx.upsert([other], repo_id="other")

    deleted = idx.delete_repo(REPO_ID)
    assert deleted == 1
    assert idx.count(REPO_ID) == 0
    assert idx.count("other") == 1


def test_commit_indexer_search_returns_records(tmp_path):
    idx = _make_indexer(tmp_path)
    idx.upsert([_commit_record(summary="introduced reranking support")], repo_id=REPO_ID)
    results = idx.search("reranking", repo_id=REPO_ID, k=5)
    assert len(results) >= 1
    assert results[0].repo_id == REPO_ID


def test_commit_indexer_count_across_repos(tmp_path):
    idx = _make_indexer(tmp_path)
    from rag.code.schema import _commit_record_id
    r1 = _commit_record()
    r2 = _commit_record(
        commit_id=_commit_record_id("other", CB),
        repo_id="other", commit_hash=CB,
        content_hash=hashlib.sha256(b"x").hexdigest(),
    )
    idx.upsert([r1], repo_id=REPO_ID)
    idx.upsert([r2], repo_id="other")
    assert idx.count() == 2
    assert idx.count(REPO_ID) == 1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


