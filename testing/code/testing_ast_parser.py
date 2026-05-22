"""Smoke test for GCR1.2 — AST-aware Parsing (PythonASTParser)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.code.ast_parser import PythonASTParser
from rag.code.schema import CodeChunk

# ---------------------------------------------------------------------------
# Module-level parse (executed once on import)
# ---------------------------------------------------------------------------

SAMPLE = '''\
"""Module docstring."""

import os
from pathlib import Path

CONST = 42


class Greeter:
    """Greeter class."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        """Return greeting string."""
        return f"Hello, {self.name}!"


class Inner:
    class Nested:
        def nested_method(self) -> None:
            pass


def top_level(x: int) -> int:
    """Top-level function."""

    def inner(y: int) -> int:
        return y * 2

    return inner(x)
'''

_parser = PythonASTParser()
chunks = _parser.parse(SAMPLE, file_path="sample.py", repo_id="test")
by_key: dict[tuple[str, str], CodeChunk] = {(c.chunk_type, c.name): c for c in chunks}

# ---------------------------------------------------------------------------
# Module chunk
# ---------------------------------------------------------------------------

def test_module_chunk_present():
    assert ("module", "<module>") in by_key, "missing module chunk"


def test_module_chunk_docstring():
    mod = by_key[("module", "<module>")]
    assert mod.docstring == "Module docstring.", f"module docstring: {mod.docstring!r}"


def test_module_chunk_start_line():
    mod = by_key[("module", "<module>")]
    assert mod.start_line == 1


# ---------------------------------------------------------------------------
# Class chunks
# ---------------------------------------------------------------------------

def test_class_greeter_present():
    assert ("class", "Greeter") in by_key, "missing Greeter"


def test_class_inner_present():
    assert ("class", "Inner") in by_key, "missing Inner"


def test_class_inner_nested_present():
    assert ("class", "Inner.Nested") in by_key, "missing Inner.Nested"


def test_inner_nested_parent_name():
    inner_nested = by_key[("class", "Inner.Nested")]
    assert inner_nested.parent_name == "Inner", f"Inner.Nested parent: {inner_nested.parent_name!r}"


# ---------------------------------------------------------------------------
# Method chunks
# ---------------------------------------------------------------------------

def test_greeter_init_present():
    assert ("method", "Greeter.__init__") in by_key, "missing Greeter.__init__"


def test_greeter_greet_present():
    assert ("method", "Greeter.greet") in by_key, "missing Greeter.greet"


def test_nested_method_present():
    assert ("method", "Inner.Nested.nested_method") in by_key, \
        f"missing Inner.Nested.nested_method; keys={list(by_key.keys())}"


def test_greet_parent_name():
    greet = by_key[("method", "Greeter.greet")]
    assert greet.parent_name == "Greeter", f"greet parent: {greet.parent_name!r}"


def test_greet_docstring():
    greet = by_key[("method", "Greeter.greet")]
    assert greet.docstring == "Return greeting string.", f"greet docstring: {greet.docstring!r}"


# ---------------------------------------------------------------------------
# Function chunks
# ---------------------------------------------------------------------------

def test_top_level_function_present():
    assert ("function", "top_level") in by_key, "missing top_level"


def test_top_level_docstring():
    top = by_key[("function", "top_level")]
    assert top.docstring == "Top-level function.", f"top_level docstring: {top.docstring!r}"


def test_top_level_no_parent():
    top = by_key[("function", "top_level")]
    assert top.parent_name is None, f"top_level.parent_name: {top.parent_name!r}"


def test_nested_function_present():
    assert ("function", "top_level.inner") in by_key, \
        f"missing nested function top_level.inner; keys={list(by_key.keys())}"


def test_nested_function_parent_name():
    inner_fn = by_key[("function", "top_level.inner")]
    assert inner_fn.parent_name == "top_level", f"inner.parent_name: {inner_fn.parent_name!r}"


# ---------------------------------------------------------------------------
# Line ranges
# ---------------------------------------------------------------------------

def test_all_chunks_have_valid_line_ranges():
    for c in chunks:
        assert c.start_line >= 1, f"invalid start_line for {c.name}: {c.start_line}"
        assert c.end_line >= c.start_line, f"invalid end_line for {c.name}"


def test_greeter_start_line():
    greeter = by_key[("class", "Greeter")]
    assert greeter.start_line == 9, f"Greeter.start_line={greeter.start_line}"


# ---------------------------------------------------------------------------
# chunk_id / content_hash / to_meta
# ---------------------------------------------------------------------------

def test_chunk_ids_have_correct_prefix():
    for c in chunks:
        assert c.chunk_id.startswith("test::sample.py::"), f"bad chunk_id: {c.chunk_id}"


def test_content_hash_is_sha256_hex():
    for c in chunks:
        assert len(c.content_hash) == 64, f"bad hash for {c.name}: {c.content_hash}"


def test_to_meta_has_required_keys():
    _REQUIRED = (
        "chunk_id", "repo_id", "file_path", "language",
        "chunk_type", "name", "start_line", "end_line",
        "parent_name", "content_hash",
    )
    for c in chunks:
        m = c.to_meta()
        for k in _REQUIRED:
            assert k in m, f"missing key {k!r} in to_meta for {c.name}"


def test_to_meta_no_none_values():
    for c in chunks:
        for k, v in c.to_meta().items():
            assert v is not None, f"None value for key {k!r} in to_meta for {c.name}"


# ---------------------------------------------------------------------------
# Round-trip serialisation
# ---------------------------------------------------------------------------

def test_round_trip_serialisation():
    for c in chunks:
        c2 = CodeChunk.from_dict(c.to_dict())
        assert c2 == c, f"round-trip failed for {c.name}"


# ---------------------------------------------------------------------------
# Real-file parse
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_file_parse():
    real_path = Path(__file__).resolve().parent.parent.parent / "rag" / "engine.py"
    if not real_path.exists():
        pytest.skip("rag/engine.py not found")
    repo_root = real_path.parent.parent
    real_chunks = _parser.parse_file(real_path, repo_root=repo_root, repo_id="langchain-test")
    types = [c.chunk_type for c in real_chunks]
    assert types.count("module") == 1, "expected exactly one module chunk"
    assert types.count("class") >= 1, "expected at least one class"
    assert types.count("method") >= 1, "expected at least one method"
