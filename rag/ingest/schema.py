"""
schema.py — canonical metadata schema for the knowledge system.

Every Document that flows through the ingestion pipeline is stamped with this
schema before being written to the vector store.

Document-level fields (identical for every chunk from the same file):
    source_id      : str   — dedup / cleanup key for LangChain index()
    doc_id         : str   — retrieval-time filter key (= source_id)
    document_type  : str   — "pdf" | "markdown" | "text"
    title          : str   — human-readable document title
    tags           : str   — comma-separated tag list (Chroma requires str)
    created_time   : str   — ISO-8601 UTC timestamp derived from file mtime
    updated_time   : str   — ISO-8601 UTC timestamp derived from file mtime
    source         : str   — absolute file path
    filename       : str   — basename of the source file
    workspace      : str   — logical workspace / project label (default "")
    importance     : float — caller-assigned priority (0.0 = neutral)

Chunk-level fields (vary per chunk):
    chunk_id : int  — 0-based sequential index within the document
    section  : str  — heading text (Markdown), "page N" 1-based (PDF), "" (text)

Parser-specific fields (preserved as-is, not part of core schema):
    page        : int — PDF page index (0-based)
    total_pages : int — PDF total page count
    heading     : str — nearest Markdown heading text
    hierarchy   : str — comma-joined ancestor headings (serialised for Chroma)
    level       : int — Markdown heading depth (0 = before first heading)
"""

import os
from datetime import datetime, timezone

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Extension → document_type mapping
# ---------------------------------------------------------------------------

_EXT_TO_TYPE: dict[str, str] = {
    ".pdf":      "pdf",
    ".md":       "markdown",
    ".markdown": "markdown",
    ".txt":      "text",
    ".text":     "text",
}


def doc_type_from_path(path: str) -> str:
    """Return the ``document_type`` string for *path* based on its extension."""
    ext = os.path.splitext(path)[1].lower()
    return _EXT_TO_TYPE.get(ext, "text")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mtime_iso(path: str) -> str:
    """Return the file mtime as an ISO-8601 UTC string, or now() on error."""
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_document_meta(
    path: str,
    *,
    doc_id: str = "",
    title: str = "",
    tags: list[str] | None = None,
    workspace: str = "",
    importance: float = 0.0,
    embedding_version: str = "",
    chunk_version: str = "",
) -> dict:
    """Build the document-level metadata dict for *path*.

    This dict is merged into every chunk produced from *path* by
    :func:`finalise_chunk`.  All values are Chroma-safe scalars (str / int /
    float / bool) — *tags* is stored as a comma-separated string.

    Args:
        path              : Absolute or relative path to the source file.
        doc_id            : Grouping / filter key.  Defaults to the filename stem.
        title             : Human-readable title.  Defaults to *doc_id*.
        tags              : Tag strings; stored joined by commas.
        workspace         : Logical workspace or project label.
        importance        : Float priority (0.0 = neutral, higher = more important).
        embedding_version : Embedding model name/version, e.g. "text-embedding-ada-002".
        chunk_version     : Chunking strategy + version tag, e.g. "heading-v1".
    """
    abs_path = os.path.abspath(path)
    filename = os.path.basename(abs_path)
    stem = os.path.splitext(filename)[0]
    resolved_doc_id = doc_id.strip() or stem
    mtime = _mtime_iso(abs_path)
    tag_str = ",".join(t.strip() for t in (tags or []) if t.strip())

    return {
        "source":            abs_path,
        "source_path":       abs_path,
        "filename":          filename,
        "source_id":         resolved_doc_id,
        "doc_id":            resolved_doc_id,
        "document_type":     doc_type_from_path(abs_path),
        "source_type":       "local",
        "title":             title.strip() or resolved_doc_id,
        "tags":              tag_str,
        "created_time":      mtime,
        "updated_time":      mtime,
        "created_at":        mtime,
        "updated_at":        mtime,
        "workspace":         workspace,
        "importance":        float(importance),
        "content_type":      "",
        "embedding_version": embedding_version,
        "chunk_version":     chunk_version,
    }


def finalise_chunk(chunk: Document, idx: int, doc_meta: dict) -> Document:
    """Merge *doc_meta* into *chunk* and add chunk-level fields.

    Parser-specific fields (page, total_pages, heading, hierarchy, level) are
    preserved under their original keys.  *section* is derived automatically:
      - Markdown : heading text
      - PDF      : "page N" (1-based)
      - Text     : ""

    Stale schema keys from previous pipeline runs are overwritten.
    """
    _SCHEMA_KEYS = frozenset(doc_meta.keys()) | {"chunk_id", "section"}
    parser_meta = {k: v for k, v in chunk.metadata.items() if k not in _SCHEMA_KEYS}

    heading = parser_meta.get("heading", "")
    page = parser_meta.get("page")

    if heading:
        section = heading
    elif page is not None:
        section = f"page {int(page) + 1}"
    else:
        section = ""

    chunk.metadata = {
        **doc_meta,
        **parser_meta,
        "chunk_id": idx,
        "section":  section,
    }
    return chunk
