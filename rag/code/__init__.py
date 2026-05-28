"""
rag.code — Engineering Knowledge Operating System (Engineering KOS)

Stage GCR1 — Repository Intelligence Foundation

Submodules
----------
schema         : RepoFile, RepoManifest, CodeChunk, Symbol, CommitInfo, FileSnapshot
scanner        : RepoScanner — filesystem walk + manifest builder
ast_parser     : PythonASTParser — AST-aware chunking (GCR1.2)
symbol_store   : SymbolStore — symbol registry (GCR1.3)
indexer        : CodeIndexer — multi-resolution Chroma indexer (GCR1.4)
repo_index     : RepoIndex — repo-level Chroma indexer (GCR1.4)
knowledge_base : CodeKnowledgeBase — unified multi-repo lifecycle orchestrator
git_reader     : GitReader — git log + per-commit file content (GCR1.5)
snapshot_store : SnapshotStore — temporal file snapshot registry (GCR1.5)
"""

from rag.code.schema import (
    RepoFile, RepoManifest, ManifestDiff,
    CodeChunk,
    Symbol, SYMBOL_TYPES,
    CommitInfo, FileSnapshot,
)
from rag.code.scanner import RepoScanner
from rag.code.path_rules import is_test_path, normalize_rel_path
from rag.code.ast_parser import PythonASTParser
from rag.code.symbol_store import SymbolStore
from rag.code.indexer import CodeIndexer
from rag.code.repo_index import RepoIndex
from rag.code.knowledge_base import CodeKnowledgeBase
from rag.code.git_reader import GitReader
from rag.code.snapshot_store import SnapshotStore, SymbolDiff
from rag.code.orchestration_service import (
    CodeOrchestrationService,
    OrchestrationResult,
    ParseStats,
    EdgeStats,
)
from rag.code.state_repository import CodeOrchestrationStateRepository

__all__ = [
    # schema
    "RepoFile",
    "RepoManifest",
    "ManifestDiff",
    "CodeChunk",
    "Symbol",
    "SYMBOL_TYPES",
    "CommitInfo",
    "FileSnapshot",
    # modules
    "RepoScanner",
    "is_test_path",
    "normalize_rel_path",
    "PythonASTParser",
    "SymbolStore",
    "CodeIndexer",
    "RepoIndex",
    "CodeKnowledgeBase",
    "GitReader",
    "SnapshotStore",
    "SymbolDiff",
    "CodeOrchestrationService",
    "OrchestrationResult",
    "ParseStats",
    "EdgeStats",
    "CodeOrchestrationStateRepository",
]
