"""Smoke test for GCR1.2 — AST-aware Parsing (PythonASTParser)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.code.ast_parser import PythonASTParser
from rag.code.schema import CodeChunk

# ---------------------------------------------------------------------------
# Fixture: synthetic Python source covering all chunk types
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

# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

parser = PythonASTParser()
chunks = parser.parse(SAMPLE, file_path="sample.py", repo_id="test")

# Build lookup by (chunk_type, name)
by_key: dict[tuple[str, str], CodeChunk] = {(c.chunk_type, c.name): c for c in chunks}

# ---------------------------------------------------------------------------
# Assertions: module chunk
# ---------------------------------------------------------------------------

assert ("module", "<module>") in by_key, "missing module chunk"
mod = by_key[("module", "<module>")]
assert mod.docstring == "Module docstring.", f"module docstring: {mod.docstring!r}"
assert mod.start_line == 1

# ---------------------------------------------------------------------------
# Assertions: class chunks
# ---------------------------------------------------------------------------

assert ("class", "Greeter") in by_key, "missing Greeter"
assert ("class", "Inner") in by_key, "missing Inner"
assert ("class", "Inner.Nested") in by_key, "missing Inner.Nested"

inner_nested = by_key[("class", "Inner.Nested")]
assert inner_nested.parent_name == "Inner", f"Inner.Nested parent: {inner_nested.parent_name!r}"

# ---------------------------------------------------------------------------
# Assertions: method chunks
# ---------------------------------------------------------------------------

assert ("method", "Greeter.__init__") in by_key, "missing Greeter.__init__"
assert ("method", "Greeter.greet") in by_key, "missing Greeter.greet"
assert ("method", "Inner.Nested.nested_method") in by_key, \
    f"missing Inner.Nested.nested_method; keys={list(by_key.keys())}"

greet = by_key[("method", "Greeter.greet")]
assert greet.parent_name == "Greeter", f"greet parent: {greet.parent_name!r}"
assert greet.docstring == "Return greeting string.", f"greet docstring: {greet.docstring!r}"

# ---------------------------------------------------------------------------
# Assertions: function chunks
# ---------------------------------------------------------------------------

assert ("function", "top_level") in by_key, "missing top_level"
top = by_key[("function", "top_level")]
assert top.docstring == "Top-level function.", f"top_level docstring: {top.docstring!r}"
assert top.parent_name is None, f"top_level.parent_name: {top.parent_name!r}"

assert ("function", "top_level.inner") in by_key, \
    f"missing nested function top_level.inner; keys={list(by_key.keys())}"
inner_fn = by_key[("function", "top_level.inner")]
assert inner_fn.parent_name == "top_level", f"inner.parent_name: {inner_fn.parent_name!r}"

# ---------------------------------------------------------------------------
# Assertions: line ranges
# ---------------------------------------------------------------------------

for c in chunks:
    assert c.start_line >= 1, f"invalid start_line for {c.name}: {c.start_line}"
    assert c.end_line >= c.start_line, f"invalid end_line for {c.name}"

# Greeter class starts at line 9
greeter = by_key[("class", "Greeter")]
assert greeter.start_line == 9, f"Greeter.start_line={greeter.start_line}"

# ---------------------------------------------------------------------------
# Assertions: chunk_id format
# ---------------------------------------------------------------------------

for c in chunks:
    assert c.chunk_id.startswith("test::sample.py::"), f"bad chunk_id: {c.chunk_id}"

# ---------------------------------------------------------------------------
# Assertions: content_hash is a 64-char SHA-256 hex string
# ---------------------------------------------------------------------------

for c in chunks:
    assert len(c.content_hash) == 64, f"bad hash for {c.name}: {c.content_hash}"

# ---------------------------------------------------------------------------
# Assertions: to_meta — Chroma-safe (no None values)
# ---------------------------------------------------------------------------

_REQUIRED_META_KEYS = (
    "chunk_id", "repo_id", "file_path", "language",
    "chunk_type", "name", "start_line", "end_line",
    "parent_name", "content_hash",
)

for c in chunks:
    m = c.to_meta()
    for k in _REQUIRED_META_KEYS:
        assert k in m, f"missing key {k!r} in to_meta for {c.name}"
    for k, v in m.items():
        assert v is not None, f"None value for key {k!r} in to_meta for {c.name}"

# ---------------------------------------------------------------------------
# Assertions: round-trip serialisation
# ---------------------------------------------------------------------------

for c in chunks:
    c2 = CodeChunk.from_dict(c.to_dict())
    assert c2 == c, f"round-trip failed for {c.name}"

# ---------------------------------------------------------------------------
# Real-file parse: rag/engine.py
# ---------------------------------------------------------------------------

real_path = Path(__file__).resolve().parent.parent / "rag" / "engine.py"
if real_path.exists():
    repo_root = real_path.parent.parent
    real_chunks = parser.parse_file(real_path, repo_root=repo_root, repo_id="langchain-test")
    types = [c.chunk_type for c in real_chunks]
    print(
        f"engine.py → {len(real_chunks)} chunks: "
        f"module={types.count('module')}, "
        f"class={types.count('class')}, "
        f"function={types.count('function')}, "
        f"method={types.count('method')}"
    )
    assert types.count("module") == 1, "expected exactly one module chunk"
    assert types.count("class") >= 1, "expected at least one class"
    assert types.count("method") >= 1, "expected at least one method"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\nsample.py → {len(chunks)} chunks total")
summary: dict[str, list[str]] = {}
for c in chunks:
    summary.setdefault(c.chunk_type, []).append(c.name)
for t, names in sorted(summary.items()):
    print(f"  {t:12s}: {names}")

print("\nPASS")
