"""
Step 0.2 — Canonical metadata schema for the knowledge system.

Every Chunk that flows into the vector store is stamped with this schema.
Fields are deliberately a superset of the legacy rag/ingest/schema.py keys so
the existing retrieval layer remains fully backward-compatible.

Metadata hierarchy
------------------
    Workspace-level  : workspace_id, workspace_name
    Page-level       : page_id, page_title, source_url, document_type,
                       language, project, tags, topics,
                       created_time, last_edited_time, importance
    Block-level      : block_id, block_type
    Chunk-level      : chunk_id, section, heading_path

Legacy aliases (kept for Chroma filter compatibility)
    doc_id    = page_id
    source_id = page_id
    source    = source_url
    title     = page_title
    workspace = workspace_name

All values stored in Chroma must be Chroma-safe scalars:
    str / int / float / bool
Lists (tags, topics, heading_path) are stored as comma-joined strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Chroma serialisation helpers
# ---------------------------------------------------------------------------

def _join(values: list[str]) -> str:
    """Comma-join a list of strings for Chroma scalar storage."""
    return ",".join(v.strip() for v in values if v.strip())


def _split(raw: str) -> list[str]:
    """Inverse of _join — returns [] for empty string."""
    return [v.strip() for v in raw.split(",") if v.strip()] if raw else []


# ---------------------------------------------------------------------------
# ChunkMetadata
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    """Full metadata attached to every indexed Chunk.

    Attributes
    ----------
    Workspace
        workspace_id   : UUID of the owning Workspace.
        workspace_name : Human-readable workspace label.
    Page
        page_id        : UUID of the parent Page (= doc_id / source_id).
        page_title     : Human-readable page/document title.
        source_url     : Notion page URL or absolute local file path.
        document_type  : "notion" | "pdf" | "markdown" | "text".
        language       : ISO-639-1 code, e.g. "en", "zh-tw". Empty = unknown.
        project        : Project/repository label this page belongs to.
        tags           : Free-form tag list.
        topics         : LLM-inferred topic list (populated in Phase 3+).
        created_time   : ISO-8601 UTC — page first created.
        last_edited_time: ISO-8601 UTC — page last edited.
        importance     : Float priority weight (0.0 = neutral).
        notion_page_id : Notion UUID (only set for Notion-origin pages).
    Block
        block_id       : UUID of the originating Block (empty for local files).
        block_type     : BlockType enum value as string.
    Chunk
        chunk_id       : 0-based sequential index within the page.
        section        : Nearest preceding heading text.
        heading_path   : Ordered ancestor heading list, e.g. ["H1", "H2"].
    """

    # ── Workspace ─────────────────────────────────────────────────────────
    workspace_id:       str = ""
    workspace_name:     str = ""

    # ── Page ──────────────────────────────────────────────────────────────
    page_id:            str = ""
    page_title:         str = ""
    source_url:         str = ""
    document_type:      str = "text"
    language:           str = ""
    project:            str = ""
    tags:               list[str] = field(default_factory=list)
    topics:             list[str] = field(default_factory=list)
    created_time:       str = ""
    last_edited_time:   str = ""
    importance:         float = 0.0
    notion_page_id:     str = ""

    # ── Block ─────────────────────────────────────────────────────────────
    block_id:           str = ""
    block_type:         str = ""

    # ── Chunk ─────────────────────────────────────────────────────────────
    chunk_id:           int = 0
    section:            str = ""
    heading_path:       list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Chroma serialisation
    # ------------------------------------------------------------------

    def to_chroma(self) -> dict[str, Any]:
        """Return a flat Chroma-safe dict (str / int / float / bool only).

        Includes legacy aliases so existing Chroma filters keep working
        without any changes to the retrieval layer.
        """
        return {
            # Workspace
            "workspace_id":      self.workspace_id,
            "workspace_name":    self.workspace_name,
            # Page
            "page_id":           self.page_id,
            "page_title":        self.page_title,
            "source_url":        self.source_url,
            "document_type":     self.document_type,
            "language":          self.language,
            "project":           self.project,
            "tags":              _join(self.tags),
            "topics":            _join(self.topics),
            "created_time":      self.created_time,
            "last_edited_time":  self.last_edited_time,
            "importance":        self.importance,
            "notion_page_id":    self.notion_page_id,
            # Block
            "block_id":          self.block_id,
            "block_type":        self.block_type,
            # Chunk
            "chunk_id":          self.chunk_id,
            "section":           self.section,
            "heading_path":      _join(self.heading_path),
            # ── Legacy aliases (backward-compat) ─────────────────────────
            "doc_id":            self.page_id,
            "source_id":         self.page_id,
            "source":            self.source_url,
            "title":             self.page_title,
            "workspace":         self.workspace_name,
        }

    @classmethod
    def from_chroma(cls, meta: dict) -> "ChunkMetadata":
        """Reconstruct a ChunkMetadata from a Chroma metadata dict."""
        return cls(
            workspace_id=meta.get("workspace_id", ""),
            workspace_name=meta.get("workspace_name", meta.get("workspace", "")),
            page_id=meta.get("page_id", meta.get("doc_id", "")),
            page_title=meta.get("page_title", meta.get("title", "")),
            source_url=meta.get("source_url", meta.get("source", "")),
            document_type=meta.get("document_type", "text"),
            language=meta.get("language", ""),
            project=meta.get("project", ""),
            tags=_split(meta.get("tags", "")),
            topics=_split(meta.get("topics", "")),
            created_time=meta.get("created_time", ""),
            last_edited_time=meta.get("last_edited_time",
                                      meta.get("updated_time", "")),
            importance=float(meta.get("importance", 0.0)),
            notion_page_id=meta.get("notion_page_id", ""),
            block_id=meta.get("block_id", ""),
            block_type=meta.get("block_type", ""),
            chunk_id=int(meta.get("chunk_id", 0)),
            section=meta.get("section", ""),
            heading_path=_split(meta.get("heading_path", "")),
        )

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def from_page(
        cls,
        page,          # rag.knowledge.models.Page
        workspace,     # rag.knowledge.models.Workspace
        *,
        chunk_id: int = 0,
        section: str = "",
        heading_path: list[str] | None = None,
        block_id: str = "",
        block_type: str = "",
    ) -> "ChunkMetadata":
        """Build a ChunkMetadata from models.Page + models.Workspace objects."""
        return cls(
            workspace_id=workspace.id,
            workspace_name=workspace.name,
            page_id=page.id,
            page_title=page.title,
            source_url=page.source,
            document_type=page.document_type,
            project=workspace.name,
            tags=list(page.tags),
            created_time=page.created_time.isoformat(),
            last_edited_time=page.updated_time.isoformat(),
            importance=page.importance,
            notion_page_id=getattr(page, "notion_page_id", "") or "",
            block_id=block_id,
            block_type=block_type,
            chunk_id=chunk_id,
            section=section,
            heading_path=heading_path or [],
        )
