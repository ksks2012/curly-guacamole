"""
GCR1.2 — AST-aware Parsing for Python source files.

Uses libcst (Concrete Syntax Tree) to produce syntax-aware chunks with
exact source boundaries, preserving comments and indentation.

Chunk types
-----------
module   : Entire file — gives file-level context (imports, constants, etc.).
class    : Class definition including its full body.
function : Top-level function or nested function (inside another function).
method   : Function defined directly inside a class body.

Naming convention for qualified names
--------------------------------------
- Top-level function:             ``top_level``
- Class method:                   ``MyClass.my_method``
- Nested class:                   ``Outer.Inner``
- Method on nested class:         ``Outer.Inner.nested_method``
- Nested function inside method:  ``MyClass.method_name.helper``
- Nested function inside function: ``outer.inner``

Design notes
------------
- libcst is used instead of token-based splitting to preserve exact syntax
  boundaries (class / function / method) rather than arbitrary token windows.
- Each chunk carries start_line / end_line from PositionProvider so chunks
  can be linked back to their exact source location.
- content_hash (SHA-256) enables incremental re-indexing: only chunks whose
  hash changed need to be re-embedded.
- Files with syntax errors are skipped gracefully (returns empty list).
"""

from __future__ import annotations

import ast as stdlib_ast
import hashlib
from pathlib import Path
from typing import Optional

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from rag.code.schema import CodeChunk, DependencyEdge


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _make_chunk_id(repo_id: str, file_path: str, chunk_type: str, name: str) -> str:
    return f"{repo_id}::{file_path}::{chunk_type}::{name}"


def _dotted_attr_to_str(node: cst.BaseExpression) -> str:
    """Convert a dotted Attribute or Name CST node to a dotted string."""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return f"{_dotted_attr_to_str(node.value)}.{node.attr.value}"
    return ""


def _call_root(func: cst.BaseExpression) -> tuple[str | None, str]:
    """Return (root_name, dot_suffix) for a call target expression.

    Examples
    --------
    Name("foo")                    → ("foo", "")
    Attribute(Name("os"), "path")  → ("os", ".path")
    os.path.join                   → ("os", ".path.join")
    """
    if isinstance(func, cst.Name):
        return func.value, ""
    if isinstance(func, cst.Attribute):
        root, suffix = _call_root(func.value)
        return root, f"{suffix}.{func.attr.value}"
    return None, ""


def _make_edge_id(src_id: str, edge_type: str, dst_id: str, file_path: str) -> str:
    raw = f"{src_id}|{edge_type}|{dst_id}|{file_path}"
    return _sha256(raw)


# Bases that signal a IMPLEMENTS edge rather than EXTENDS.
_PROTOCOL_BASES: frozenset[str] = frozenset({"Protocol", "ABC", "ABCMeta"})


def _extract_docstring(body: cst.BaseSuite) -> Optional[str]:
    """Return the first-statement docstring from an IndentedBlock, or None."""
    if not isinstance(body, cst.IndentedBlock) or not body.body:
        return None
    first = body.body[0]
    if not isinstance(first, cst.SimpleStatementLine) or len(first.body) != 1:
        return None
    stmt = first.body[0]
    if not isinstance(stmt, cst.Expr):
        return None
    val = stmt.value
    if isinstance(val, cst.SimpleString):
        try:
            return stdlib_ast.literal_eval(val.value)
        except Exception:
            return None
    # ConcatenatedString / FormattedString: skip
    return None


def _module_docstring(module: cst.Module) -> Optional[str]:
    """Return the module-level docstring, or None."""
    if not module.body:
        return None
    first = module.body[0]
    if not isinstance(first, cst.SimpleStatementLine) or len(first.body) != 1:
        return None
    stmt = first.body[0]
    if not isinstance(stmt, cst.Expr):
        return None
    val = stmt.value
    if isinstance(val, cst.SimpleString):
        try:
            return stdlib_ast.literal_eval(val.value)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Edge collector — GCR2.1 Dependency Graph
# ---------------------------------------------------------------------------

class _EdgeCollector(cst.CSTVisitor):
    """Walk a CST and collect IMPORTS / EXTENDS / IMPLEMENTS / CALLS edges.

    All edges use the file-level module symbol as src (``module::<module>``).
    except EXTENDS / IMPLEMENTS which use the class symbol as src.

    Only calls to directly imported names are tracked for CALLS to ensure
    near-100 % accuracy without type inference.
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, module: cst.Module, file_path: str, repo_id: str) -> None:
        self._module    = module
        self._file_path = file_path
        self._repo_id   = repo_id
        self.edges: list[DependencyEdge] = []
        # local_alias → fully-qualified import target
        self._import_map: dict[str, str] = {}
        # dedup within one parse pass
        self._seen: set[str] = set()
        # class name stack for qualified name building
        self._class_stack: list[str] = []

    def _module_sym_id(self) -> str:
        return f"{self._repo_id}::{self._file_path}::module::<module>"

    def _add_edge(
        self, src_id: str, dst_id: str, edge_type: str, line_no: int
    ) -> None:
        eid = _make_edge_id(src_id, edge_type, dst_id, self._file_path)
        if eid in self._seen:
            return
        self._seen.add(eid)
        self.edges.append(DependencyEdge(
            edge_id=eid,
            src_id=src_id,
            dst_id=dst_id,
            edge_type=edge_type,
            repo_id=self._repo_id,
            file_path=self._file_path,
            line_no=line_no,
        ))

    # ── import foo.bar [as alias] ─────────────────────────────────────────

    def visit_Import(self, node: cst.Import) -> None:
        if isinstance(node.names, cst.ImportStar):
            return
        pos = self.get_metadata(PositionProvider, node)
        for alias in node.names:
            mod_path = _dotted_attr_to_str(alias.name)
            if not mod_path:
                continue
            if alias.asname and isinstance(alias.asname, cst.AsName):
                asname_node = alias.asname.name
                local = (
                    asname_node.value
                    if isinstance(asname_node, cst.Name)
                    else mod_path.split(".")[0]
                )
            else:
                local = mod_path.split(".")[0]
            self._import_map[local] = mod_path
            self._add_edge(
                self._module_sym_id(),
                f"import::{mod_path}",
                "IMPORTS",
                pos.start.line,
            )

    # ── from foo.bar import Baz [as B] ────────────────────────────────────

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        # Skip relative imports — cannot resolve without package context.
        if node.relative:
            return
        if isinstance(node.names, cst.ImportStar):
            return
        pos = self.get_metadata(PositionProvider, node)
        mod_base = _dotted_attr_to_str(node.module) if node.module else ""
        for alias in node.names:
            name = (
                alias.name.value
                if isinstance(alias.name, cst.Name)
                else _dotted_attr_to_str(alias.name)
            )
            qualified = f"{mod_base}.{name}" if mod_base else name
            if alias.asname and isinstance(alias.asname, cst.AsName):
                asname_node = alias.asname.name
                local = (
                    asname_node.value
                    if isinstance(asname_node, cst.Name)
                    else name
                )
            else:
                local = name
            self._import_map[local] = qualified
            self._add_edge(
                self._module_sym_id(),
                f"import::{qualified}",
                "IMPORTS",
                pos.start.line,
            )

    # ── class Foo(Bar, Protocol): ─────────────────────────────────────────

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self._class_stack.append(node.name.value)
        return True  # visit children

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        pos = self.get_metadata(PositionProvider, original_node)
        qualified = ".".join(self._class_stack)
        src_id = f"{self._repo_id}::{self._file_path}::class::{qualified}"
        for arg in original_node.bases:
            base_str = _dotted_attr_to_str(arg.value)
            if not base_str:
                continue
            root = base_str.split(".")[0]
            resolved_root = self._import_map.get(root, root)
            resolved = resolved_root + base_str[len(root):]
            leaf = base_str.split(".")[-1]
            edge_type = "IMPLEMENTS" if leaf in _PROTOCOL_BASES else "EXTENDS"
            self._add_edge(src_id, f"import::{resolved}", edge_type, pos.start.line)
        self._class_stack.pop()

    # ── any_imported_name(...) ────────────────────────────────────────────

    def visit_Call(self, node: cst.Call) -> None:
        pos = self.get_metadata(PositionProvider, node)
        root_name, suffix = _call_root(node.func)
        if root_name is None or root_name not in self._import_map:
            return
        resolved = self._import_map[root_name]
        if suffix:
            resolved = f"{resolved}{suffix}"
        self._add_edge(
            self._module_sym_id(),
            f"import::{resolved}",
            "CALLS",
            pos.start.line,
        )


# ---------------------------------------------------------------------------
# CST visitor
# ---------------------------------------------------------------------------

class _ChunkCollector(cst.CSTVisitor):
    """Walk a libcst tree and collect CodeChunk objects for each code boundary."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, module: cst.Module, file_path: str, repo_id: str) -> None:
        self._module = module
        self._file_path = file_path
        self._repo_id = repo_id
        self.chunks: list[CodeChunk] = []

        # Stacks track nesting context.
        self._class_stack: list[str] = []   # class names currently open
        self._func_stack: list[str] = []    # function names currently open

    # ── class ────────────────────────────────────────────────────────────

    def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
        self._class_stack.append(node.name.value)
        return True  # visit children

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        pos = self.get_metadata(PositionProvider, original_node)
        qualified = ".".join(self._class_stack)
        parent = ".".join(self._class_stack[:-1]) or None
        code = self._module.code_for_node(original_node)
        self.chunks.append(CodeChunk(
            chunk_id=_make_chunk_id(self._repo_id, self._file_path, "class", qualified),
            repo_id=self._repo_id,
            file_path=self._file_path,
            language="python",
            chunk_type="class",
            name=qualified,
            start_line=pos.start.line,
            end_line=pos.end.line,
            content=code,
            docstring=_extract_docstring(original_node.body),
            parent_name=parent,
            content_hash=_sha256(code),
        ))
        self._class_stack.pop()

    # ── function / method ────────────────────────────────────────────────

    def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
        self._func_stack.append(node.name.value)
        return True  # visit children

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        func_name = original_node.name.value
        pos = self.get_metadata(PositionProvider, original_node)
        code = self._module.code_for_node(original_node)
        docstring = _extract_docstring(original_node.body)

        if self._class_stack and len(self._func_stack) == 1:
            # Direct child of a class → method
            chunk_type = "method"
            parent_name: Optional[str] = ".".join(self._class_stack)
            qualified = f"{parent_name}.{func_name}"
        elif self._class_stack:
            # Nested function inside a class method → function with class-qualified name
            chunk_type = "function"
            parent_name = ".".join(self._class_stack) + "." + ".".join(self._func_stack[:-1])
            qualified = ".".join(self._class_stack) + "." + ".".join(self._func_stack)
        else:
            # Top-level or nested free function
            chunk_type = "function"
            parent_name = ".".join(self._func_stack[:-1]) or None
            qualified = ".".join(self._func_stack)

        self.chunks.append(CodeChunk(
            chunk_id=_make_chunk_id(self._repo_id, self._file_path, chunk_type, qualified),
            repo_id=self._repo_id,
            file_path=self._file_path,
            language="python",
            chunk_type=chunk_type,
            name=qualified,
            start_line=pos.start.line,
            end_line=pos.end.line,
            content=code,
            docstring=docstring,
            parent_name=parent_name,
            content_hash=_sha256(code),
        ))
        self._func_stack.pop()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PythonASTParser:
    """Parse a Python source file into syntax-aware CodeChunk objects.

    Each significant syntax boundary (module, class, function, method)
    becomes an independent chunk with exact line ranges and source text.

    Usage
    -----
    >>> parser = PythonASTParser()
    >>> chunks = parser.parse(source_code, file_path="rag/engine.py", repo_id="my-repo")
    """

    def parse(self, source: str, file_path: str, repo_id: str) -> list[CodeChunk]:
        """Parse *source* and return a list of CodeChunk objects.

        Parameters
        ----------
        source    : Raw Python source text.
        file_path : Relative POSIX path from the repo root (used for IDs).
        repo_id   : Logical repository identifier (used for IDs).

        Returns an empty list if the source has a syntax error.
        """
        try:
            module = cst.parse_module(source)
        except cst.ParserSyntaxError:
            return []

        end_line = source.count("\n") + 1

        # Module-level chunk (entire file)
        module_chunk = CodeChunk(
            chunk_id=_make_chunk_id(repo_id, file_path, "module", "<module>"),
            repo_id=repo_id,
            file_path=file_path,
            language="python",
            chunk_type="module",
            name="<module>",
            start_line=1,
            end_line=end_line,
            content=source,
            docstring=_module_docstring(module),
            parent_name=None,
            content_hash=_sha256(source),
        )

        # Class / function / method chunks via CST visitor
        wrapper = MetadataWrapper(module)
        collector = _ChunkCollector(module, file_path, repo_id)
        wrapper.visit(collector)

        return [module_chunk] + collector.chunks

    def parse_file(self, path: Path, repo_root: Path, repo_id: str) -> list[CodeChunk]:
        """Read *path* from disk and parse it.

        Parameters
        ----------
        path      : Absolute path to the Python source file.
        repo_root : Absolute path to the repository root.
        repo_id   : Logical repository identifier.
        """
        source = path.read_text(encoding="utf-8", errors="replace")
        file_path = path.relative_to(repo_root).as_posix()
        return self.parse(source, file_path, repo_id)

    def parse_edges(
        self, source: str, file_path: str, repo_id: str
    ) -> list[DependencyEdge]:
        """Parse *source* and return dependency edges.

        Returns an empty list if the source has a syntax error.

        Parameters
        ----------
        source    : Raw Python source text.
        file_path : Relative POSIX path from the repo root.
        repo_id   : Logical repository identifier.
        """
        try:
            module = cst.parse_module(source)
        except cst.ParserSyntaxError:
            return []
        wrapper   = MetadataWrapper(module)
        collector = _EdgeCollector(module, file_path, repo_id)
        wrapper.visit(collector)
        return collector.edges

    def parse_edges_file(
        self, path: Path, repo_root: Path, repo_id: str
    ) -> list[DependencyEdge]:
        """Read *path* from disk and return dependency edges."""
        source    = path.read_text(encoding="utf-8", errors="replace")
        file_path = path.relative_to(repo_root).as_posix()
        return self.parse_edges(source, file_path, repo_id)
