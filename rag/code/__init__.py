"""
rag.code — Engineering Knowledge Operating System (Engineering KOS)

Stage GCR1 — Repository Intelligence Foundation

Submodules
----------
schema  : RepoFile, RepoManifest dataclasses
scanner : RepoScanner — filesystem walk + manifest builder
"""

from rag.code.schema import RepoFile, RepoManifest, ManifestDiff
from rag.code.scanner import RepoScanner

__all__ = [
    "RepoFile",
    "RepoManifest",
    "ManifestDiff",
    "RepoScanner",
]
