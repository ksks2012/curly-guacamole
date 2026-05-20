"""
GCR1.3 — Symbol Registry.

SymbolStore is an in-memory registry of all named code symbols extracted
from a repository.  It can be:

- Built directly from a list of CodeChunk objects (via PythonASTParser).
- Queried by file, type, name, or parent.
- Persisted to / loaded from a JSON file for incremental indexing.

Typical usage
-------------
>>> from rag.code.ast_parser import PythonASTParser
>>> from rag.code.symbol_store import SymbolStore
>>>
>>> parser  = PythonASTParser()
>>> chunks  = parser.parse_file(path, repo_root, "my-repo")
>>> store   = SymbolStore.from_chunks(chunks)
>>> store.save("my_db/symbols.json")

Symbol visibility (Python convention)
--------------------------------------
- ``"dunder"``  : name starts and ends with ``__`` (e.g. ``__init__``)
- ``"private"`` : name starts with a single ``_`` (e.g. ``_helper``)
- ``"public"``  : everything else
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rag.code.schema import CodeChunk, Symbol, SYMBOL_TYPES


# ---------------------------------------------------------------------------
# Visibility helper
# ---------------------------------------------------------------------------

# Mapping from CodeChunk.chunk_type to Symbol.symbol_type.
# "module" → "module"  (1-to-1); chunk_type values not in this map are kept as-is.
_CHUNK_TO_SYMBOL_TYPE: dict[str, str] = {
    "module":   "module",
    "class":    "class",
    "function": "function",
    "method":   "method",
}


def _infer_visibility(name: str) -> str:
    """Infer Python visibility from the leaf part of a qualified name."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf.startswith("__") and leaf.endswith("__") and len(leaf) > 4:
        return "dunder"
    if leaf.startswith("_"):
        return "private"
    return "public"


def _make_symbol_id(repo_id: str, file_path: str, symbol_type: str, symbol_name: str) -> str:
    return f"{repo_id}::{file_path}::{symbol_type}::{symbol_name}"


# ---------------------------------------------------------------------------
# SymbolStore
# ---------------------------------------------------------------------------

class SymbolStore:
    """In-memory registry of code symbols for a repository.

    Symbols are indexed by ``symbol_id`` for O(1) lookup.  Auxiliary indexes
    are maintained for fast queries by file path and symbol type.

    Parameters
    ----------
    repo_id : Optional default repo_id used when ``from_chunks`` is called
              without explicit ``repo_id``.  Can be ``None`` if symbols are
              always added via :meth:`add` / :meth:`add_many`.
    """

    def __init__(self, repo_id: str = "") -> None:
        self._repo_id = repo_id
        self._by_id:   dict[str, Symbol] = {}
        # secondary indexes
        self._by_file: dict[str, list[str]] = {}  # file_path → [symbol_id, ...]
        self._by_type: dict[str, list[str]] = {}  # symbol_type → [symbol_id, ...]

    # ── Core mutators ─────────────────────────────────────────────────────

    def add(self, symbol: Symbol) -> None:
        """Register a single symbol.  Duplicate symbol_ids are overwritten."""
        sid = symbol.symbol_id
        self._by_id[sid] = symbol
        self._by_file.setdefault(symbol.file_path, [])
        if sid not in self._by_file[symbol.file_path]:
            self._by_file[symbol.file_path].append(sid)
        self._by_type.setdefault(symbol.symbol_type, [])
        if sid not in self._by_type[symbol.symbol_type]:
            self._by_type[symbol.symbol_type].append(sid)

    def add_many(self, symbols: Iterable[Symbol]) -> None:
        for s in symbols:
            self.add(s)

    # ── Factory: build from CodeChunk list ────────────────────────────────

    @classmethod
    def from_chunks(cls, chunks: Iterable[CodeChunk], repo_id: str = "") -> "SymbolStore":
        """Build a SymbolStore from a list of CodeChunk objects.

        Each chunk (module / class / function / method) becomes one Symbol.
        The ``repo_id`` is taken from the first chunk if not supplied.
        """
        chunks = list(chunks)
        if not chunks:
            return cls(repo_id=repo_id)
        rid = repo_id or (chunks[0].repo_id if chunks else "")
        store = cls(repo_id=rid)
        # Sort by start_line so that enclosing symbols (parent classes/functions)
        # are registered before their children, enabling parent resolution.
        sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
        for chunk in sorted_chunks:
            stype = _CHUNK_TO_SYMBOL_TYPE.get(chunk.chunk_type, chunk.chunk_type)
            if stype not in SYMBOL_TYPES:
                continue  # skip unknown types
            parent_sid = ""
            if chunk.parent_name:
                # Determine parent's symbol_type heuristically:
                # If parent contains no dot or is in class stack, it could be class or function.
                # We resolve against what's already registered by scanning by file.
                parent_sid = _resolve_parent_sid(store, chunk)
            sym = Symbol(
                symbol_id=_make_symbol_id(chunk.repo_id, chunk.file_path, stype, chunk.name),
                symbol_name=chunk.name,
                symbol_type=stype,
                repo_id=chunk.repo_id,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                parent_symbol=parent_sid,
                visibility=_infer_visibility(chunk.name),
            )
            store.add(sym)
        return store

    # ── Queries ───────────────────────────────────────────────────────────

    def get(self, symbol_id: str) -> Symbol | None:
        """Return the Symbol with the given id, or None."""
        return self._by_id.get(symbol_id)

    def by_file(self, file_path: str) -> list[Symbol]:
        """All symbols declared in *file_path*, in registration order."""
        ids = self._by_file.get(file_path, [])
        return [self._by_id[i] for i in ids]

    def by_type(self, symbol_type: str) -> list[Symbol]:
        """All symbols of *symbol_type* across the whole store."""
        ids = self._by_type.get(symbol_type, [])
        return [self._by_id[i] for i in ids]

    def find(self, name: str, *, exact: bool = False) -> list[Symbol]:
        """Search symbols by name.

        Parameters
        ----------
        name  : Substring (or exact when *exact=True*) to match against
                ``symbol_name``.
        exact : When True, only symbols whose ``symbol_name == name`` are returned.
        """
        if exact:
            return [s for s in self._by_id.values() if s.symbol_name == name]
        needle = name.lower()
        return [s for s in self._by_id.values() if needle in s.symbol_name.lower()]

    def children_of(self, symbol_id: str) -> list[Symbol]:
        """Return all symbols whose ``parent_symbol`` equals *symbol_id*."""
        return [s for s in self._by_id.values() if s.parent_symbol == symbol_id]

    # ── Statistics ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def summary(self) -> str:
        parts = [f"total={len(self)}"]
        for t in sorted(self._by_type):
            parts.append(f"{t}={len(self._by_type[t])}")
        return "  ".join(parts)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "repo_id": self._repo_id,
            "symbols": [s.to_dict() for s in self._by_id.values()],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SymbolStore":
        """Deserialise a store previously saved by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(repo_id=data.get("repo_id", ""))
        for d in data.get("symbols", []):
            store.add(Symbol.from_dict(d))
        return store

    def merge(self, other: "SymbolStore") -> None:
        """Merge all symbols from *other* into this store (in-place)."""
        for sym in other:
            self.add(sym)


# ---------------------------------------------------------------------------
# Internal: parent resolution
# ---------------------------------------------------------------------------

def _resolve_parent_sid(store: "SymbolStore", chunk: CodeChunk) -> str:
    """Attempt to find the symbol_id of chunk's parent in the store.

    Strategy: iterate over already-registered symbols in the same file
    and match by symbol_name == chunk.parent_name.  Returns "" on failure.
    """
    if not chunk.parent_name:
        return ""
    for sym in store.by_file(chunk.file_path):
        if sym.symbol_name == chunk.parent_name:
            return sym.symbol_id
    return ""
