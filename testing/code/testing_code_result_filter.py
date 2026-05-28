"""Unit tests for retrieval-layer code result filtering."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.retrieval.code_result_filter import CodeResultFilter


def _doc(path: str, *, is_test: bool | None = None) -> Document:
    meta = {"file_path": path, "chunk_id": path}
    if is_test is not None:
        meta["is_test"] = is_test
    return Document(page_content="x", metadata=meta)


def test_filter_detects_test_metadata_flag() -> None:
    f = CodeResultFilter()
    assert f.is_test_metadata({"file_path": "src/app.py", "is_test": True}) is True


def test_filter_detects_test_path_patterns() -> None:
    f = CodeResultFilter()
    assert f.is_test_metadata({"file_path": "testing/ui/test_demo.py"}) is True
    assert f.is_test_metadata({"file_path": "src/app/service.py"}) is False


def test_filter_scored_documents_excludes_test_rows() -> None:
    f = CodeResultFilter()
    rows = [
        (_doc("src/app/service.py"), 0.8),
        (_doc("testing/ui/testing_demo.py"), 0.7),
    ]
    out = f.filter_scored_documents(rows)
    assert len(out) == 1
    assert out[0][0].metadata["file_path"] == "src/app/service.py"


def test_filter_content_rows_excludes_test_rows() -> None:
    f = CodeResultFilter()
    rows = [
        {"content": "a", "metadata": {"file_path": "src/app/service.py"}},
        {"content": "b", "metadata": {"file_path": "testing/ui/testing_demo.py"}},
    ]
    out = f.filter_content_rows(rows)
    assert len(out) == 1
    assert out[0]["metadata"]["file_path"] == "src/app/service.py"
