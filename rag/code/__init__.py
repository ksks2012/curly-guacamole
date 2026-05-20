"""
rag.code — Engineering Knowledge Operating System (Engineering KOS)

Stage GCR1 — Repository Intelligence Foundation

Submodules
----------
schema       : RepoFile, RepoManifest, CodeChunk, Symbol dataclasses
scanner      : RepoScanner — filesystem walk + manifest builder
ast_parser   : PythonASTParser — AST-aware chunking (GCR1.2)
symbol_store : SymbolStore — symbol registry (GCR1.3)
"""

from rag.code.schema import RepoFile, RepoManifest, ManifestDiff, CodeChunk, Symbol, SYMBOL_TYPES
from rag.code.scanner import RepoScanner
from rag.code.ast_parser import PythonASTParser
from rag.code.symbol_store import SymbolStore

__all__ = [
    "RepoFile",
    "RepoManifest",
    "ManifestDiff",
    "RepoScanner",
    "CodeChunk",
    "PythonASTParser",
    "Symbol",
    "SYMBOL_TYPES",
    "SymbolStore",
]
