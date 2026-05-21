"""
Phase 2 Step 2.2 — Unified Chunk Model.

Defines ``BaseChunk``, the single base type that every chunk in the system
extends regardless of its origin (document, code, commit, note, qa, summary).

Motivation
----------
Before this step the system had two independent chunk representations:
  - LangChain ``Document`` + flat metadata dict  (document ingestion)
  - ``CodeChunk`` dataclass                       (code ingestion)

Downstream components (retrieval, reranking, evaluation, memory) had to
branch on source type to handle differences.  ``BaseChunk`` closes that gap.

Hierarchy
---------
    BaseChunk                  ← you are here
        ├── CodeChunk          (rag/code/schema.py)
        ├── DocumentChunk      (future — rag/ingest/)
        ├── CommitChunk        (future — rag/code/)
        └── NoteChunk          (future — rag/memory/)

Source types
------------
    "document"  — PDF, Markdown, plain-text files
    "code"      — AST-parsed source code chunks
    "commit"    — git commit metadata and diffs
    "note"      — user or system generated notes
    "qa"        — generated question-answer pairs
    "summary"   — LLM-generated summaries of other chunks

Design notes
------------
- ``@dataclass(kw_only=True)`` avoids the default-ordering constraint in
  Python dataclass inheritance so every subclass can freely add required
  fields without hitting "non-default follows default" errors.
- ``metadata`` is the escape-hatch bag for backend-specific scalar values.
  Subclasses that need typed access to specific metadata keys should declare
  them as first-class dataclass fields instead.
- ``embedding`` is optional and defaults to an empty list.  Vector stores
  populate it when reading back from storage; during ingestion it stays empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# SourceType
# ---------------------------------------------------------------------------

SourceType = Literal["document", "code", "commit", "note", "qa", "summary"]


# ---------------------------------------------------------------------------
# BaseChunk
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class BaseChunk:
    """Core representation of a knowledge chunk regardless of its origin.

    Attributes
    ----------
    chunk_id    : Globally unique, deterministic identifier.
                  Subclasses define the ID format; e.g. CodeChunk uses
                  ``"{repo_id}::{file_path}::{chunk_type}::{name}"``.
    source_type : Origin label — one of the ``SourceType`` literals.
                  Subclasses should redeclare this with their own default.
    content     : The text that gets embedded and shown to the LLM.
                  For code chunks this is the raw source text; for document
                  chunks it is the prose content of the chunk window.
    metadata    : Flat dict of Chroma-safe scalar values (str / int / float /
                  bool).  Used for metadata filtering at retrieval time.
    embedding   : Pre-computed embedding vector, if available.  Populated
                  by vector-store readers; empty list during ingestion.
    """

    chunk_id:    str
    source_type: SourceType
    content:     str
    metadata:    dict        = field(default_factory=dict)
    embedding:   list[float] = field(default_factory=list)
