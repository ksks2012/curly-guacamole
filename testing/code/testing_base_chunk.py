"""
Smoke tests for Phase 2 Step 2.2 — Unified Chunk Model (BaseChunk).

Tests:
  - BaseChunk can be instantiated directly with required fields
  - BaseChunk metadata and embedding default to empty
  - SourceType literals are accepted by BaseChunk
  - CodeChunk is a subclass of BaseChunk
  - CodeChunk source_type defaults to "code"
  - CodeChunk.code property returns content (backward compat)
  - CodeChunk.to_meta() returns code-specific fields (no content/embedding)
  - CodeChunk.to_dict() includes content key (not code)
  - CodeChunk.from_dict() accepts old "code" key (backward compat)
  - CodeChunk.from_dict() accepts new "content" key
  - isinstance(code_chunk, BaseChunk) is True
  - PythonASTParser still produces valid CodeChunk objects
  - Parsed chunks have content equal to source text slices
  - chunk.code == chunk.content for all parsed chunks
"""

from __future__ import annotations

import sys

PASS = []
FAIL = []

def ok(name: str) -> None:
    print(f"{name}: OK")
    PASS.append(name)

def fail(name: str, exc: Exception) -> None:
    print(f"{name}: FAIL — {exc}")
    FAIL.append(name)

# ---------------------------------------------------------------------------
# BaseChunk tests
# ---------------------------------------------------------------------------

def test_base_chunk_basic():
    from rag.chunk import BaseChunk
    c = BaseChunk(chunk_id="test-1", source_type="document", content="hello world")
    assert c.chunk_id == "test-1"
    assert c.source_type == "document"
    assert c.content == "hello world"
    assert c.metadata == {}
    assert c.embedding == []
    ok("test_base_chunk_basic")

def test_base_chunk_with_metadata():
    from rag.chunk import BaseChunk
    c = BaseChunk(chunk_id="x", source_type="code", content="def f(): pass",
                  metadata={"doc_id": "repo1"}, embedding=[0.1, 0.2])
    assert c.metadata["doc_id"] == "repo1"
    assert c.embedding == [0.1, 0.2]
    ok("test_base_chunk_with_metadata")

def test_base_chunk_all_source_types():
    from rag.chunk import BaseChunk
    for st in ("document", "code", "commit", "note", "qa", "summary"):
        c = BaseChunk(chunk_id="x", source_type=st, content="")
        assert c.source_type == st
    ok("test_base_chunk_all_source_types")

# ---------------------------------------------------------------------------
# CodeChunk inheritance tests
# ---------------------------------------------------------------------------

def test_code_chunk_is_base_chunk():
    from rag.chunk import BaseChunk
    from rag.code.schema import CodeChunk
    assert issubclass(CodeChunk, BaseChunk)
    ok("test_code_chunk_is_base_chunk")

def test_code_chunk_source_type_default():
    from rag.code.schema import CodeChunk
    c = CodeChunk(chunk_id="r::f::fn::foo", content="def foo(): pass",
                  repo_id="r", file_path="f.py", language="python",
                  chunk_type="function", name="foo",
                  start_line=1, end_line=1, content_hash="abc")
    assert c.source_type == "code"
    ok("test_code_chunk_source_type_default")

def test_code_chunk_isinstance_base():
    from rag.chunk import BaseChunk
    from rag.code.schema import CodeChunk
    c = CodeChunk(chunk_id="r::f::fn::foo", content="def foo(): pass",
                  repo_id="r", file_path="f.py", language="python",
                  chunk_type="function", name="foo",
                  start_line=1, end_line=1, content_hash="abc")
    assert isinstance(c, BaseChunk)
    ok("test_code_chunk_isinstance_base")

def test_code_chunk_code_property_backward_compat():
    from rag.code.schema import CodeChunk
    src = "def bar(): return 42"
    c = CodeChunk(chunk_id="x", content=src, repo_id="r", file_path="f.py",
                  language="python", chunk_type="function", name="bar",
                  start_line=1, end_line=1, content_hash="h")
    assert c.code == src
    assert c.code == c.content
    ok("test_code_chunk_code_property_backward_compat")

def test_code_chunk_to_meta_has_no_content():
    from rag.code.schema import CodeChunk
    c = CodeChunk(chunk_id="r::f::fn::foo", content="pass",
                  repo_id="r", file_path="f.py", language="python",
                  chunk_type="function", name="foo",
                  start_line=1, end_line=1, content_hash="h")
    meta = c.to_meta()
    assert "content" not in meta
    assert "embedding" not in meta
    assert meta["chunk_id"] == "r::f::fn::foo"
    assert meta["repo_id"] == "r"
    ok("test_code_chunk_to_meta_has_no_content")

def test_code_chunk_to_dict_has_content():
    from rag.code.schema import CodeChunk
    c = CodeChunk(chunk_id="x", content="pass", repo_id="r", file_path="f.py",
                  language="python", chunk_type="function", name="f",
                  start_line=1, end_line=1, content_hash="h")
    d = c.to_dict()
    assert "content" in d
    assert "code" not in d   # old key gone
    assert d["content"] == "pass"
    ok("test_code_chunk_to_dict_has_content")

def test_code_chunk_from_dict_new_format():
    from rag.code.schema import CodeChunk
    d = {"chunk_id": "x", "content": "pass", "repo_id": "r",
         "file_path": "f.py", "language": "python", "chunk_type": "function",
         "name": "f", "start_line": 1, "end_line": 1, "content_hash": "h"}
    c = CodeChunk.from_dict(d)
    assert c.content == "pass"
    ok("test_code_chunk_from_dict_new_format")

def test_code_chunk_from_dict_old_code_key():
    """Backward compat: dicts serialised before Step 2.2 had 'code' not 'content'."""
    from rag.code.schema import CodeChunk
    d = {"chunk_id": "x", "code": "pass", "repo_id": "r",
         "file_path": "f.py", "language": "python", "chunk_type": "function",
         "name": "f", "start_line": 1, "end_line": 1, "content_hash": "h"}
    c = CodeChunk.from_dict(d)
    assert c.content == "pass"
    assert c.code == "pass"
    ok("test_code_chunk_from_dict_old_code_key")

# ---------------------------------------------------------------------------
# PythonASTParser integration
# ---------------------------------------------------------------------------

SOURCE = '''\
def greet(name: str) -> str:
    """Return a greeting string."""
    return f"Hello, {name}"

class Calculator:
    """Simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b
'''

def test_ast_parser_produces_code_chunks():
    from rag.code.ast_parser import PythonASTParser
    from rag.code.schema import CodeChunk
    parser = PythonASTParser()
    chunks = parser.parse(SOURCE, file_path="calc.py", repo_id="test")
    assert len(chunks) > 0
    for c in chunks:
        assert isinstance(c, CodeChunk)
    ok("test_ast_parser_produces_code_chunks")

def test_ast_parser_chunks_isinstance_base_chunk():
    from rag.chunk import BaseChunk
    from rag.code.ast_parser import PythonASTParser
    parser = PythonASTParser()
    chunks = parser.parse(SOURCE, file_path="calc.py", repo_id="test")
    for c in chunks:
        assert isinstance(c, BaseChunk), f"chunk {c.chunk_id!r} is not a BaseChunk"
    ok("test_ast_parser_chunks_isinstance_base_chunk")

def test_ast_parser_code_equals_content():
    from rag.code.ast_parser import PythonASTParser
    parser = PythonASTParser()
    chunks = parser.parse(SOURCE, file_path="calc.py", repo_id="test")
    for c in chunks:
        assert c.code == c.content, f"code != content for {c.chunk_id!r}"
    ok("test_ast_parser_code_equals_content")

def test_ast_parser_source_type_is_code():
    from rag.code.ast_parser import PythonASTParser
    parser = PythonASTParser()
    chunks = parser.parse(SOURCE, file_path="calc.py", repo_id="test")
    for c in chunks:
        assert c.source_type == "code", f"Expected 'code', got {c.source_type!r}"
    ok("test_ast_parser_source_type_is_code")

def test_ast_parser_chunk_types():
    from rag.code.ast_parser import PythonASTParser
    parser = PythonASTParser()
    chunks = parser.parse(SOURCE, file_path="calc.py", repo_id="test")
    types = {c.chunk_type for c in chunks}
    assert "module" in types
    assert "function" in types
    assert "class" in types
    assert "method" in types
    ok("test_ast_parser_chunk_types")

# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_base_chunk_basic,
        test_base_chunk_with_metadata,
        test_base_chunk_all_source_types,
        test_code_chunk_is_base_chunk,
        test_code_chunk_source_type_default,
        test_code_chunk_isinstance_base,
        test_code_chunk_code_property_backward_compat,
        test_code_chunk_to_meta_has_no_content,
        test_code_chunk_to_dict_has_content,
        test_code_chunk_from_dict_new_format,
        test_code_chunk_from_dict_old_code_key,
        test_ast_parser_produces_code_chunks,
        test_ast_parser_chunks_isinstance_base_chunk,
        test_ast_parser_code_equals_content,
        test_ast_parser_source_type_is_code,
        test_ast_parser_chunk_types,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            fail(t.__name__, e)

    print()
    if FAIL:
        print(f"FAIL  ({len(FAIL)} failed, {len(PASS)} passed)")
        sys.exit(1)
    else:
        print("PASS")
