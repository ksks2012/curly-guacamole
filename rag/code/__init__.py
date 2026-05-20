"""
rag.code — Engineering Knowledge Operating System (Engineering KOS)

Stage GCR1 — Repository Intelligence Foundation

Submodules
----------
schema     : RepoFile, RepoManifest, CodeChunk dataclasses
scanner    : RepoScanner — filesystem walk + manifest builder
ast_parser : PythonASTParser — AST-aware chunking (GCR1.2)
"""

from rag.code.schema import RepoFile, RepoManifest, ManifestDiff, CodeChunk
from rag.code.scanner import RepoScanner
from rag.code.ast_parser import PythonASTParser

__all__ = [
    "RepoFile",
    "RepoManifest",
    "ManifestDiff",
    "RepoScanner",
    "CodeChunk",
    "PythonASTParser",
]
