"""
Raw Storage Layer (SQLite) — backed by SQLAlchemy Core.

Stores the raw Notion / local-file data before it enters the vector store.
This layer enables:
    - Rebuilding chunks without re-fetching from Notion
    - Swapping embedding models (re-embed from stored raw text)
    - Incremental sync (compare stored hash vs. live hash to skip unchanged pages)
    - Future graph extraction and knowledge evolution analysis

Schema
------
    workspaces        — one row per Workspace
    pages             — one row per Page; stores content_hash for change detection
    blocks            — one row per Block; raw content + metadata JSON
    document_versions — point-in-time snapshots for change tracking
    sync_cursors      — stores Notion next_cursor values for incremental API polling

All datetimes are stored as ISO-8601 UTC strings.
JSON fields store arbitrary dicts serialised with json.dumps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    func,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection, Engine

from utils.logger import AppLogger
from rag.knowledge.models import (
    Block,
    BlockType,
    ChangeType,
    DocumentVersion,
    Page,
    Workspace,
)

log = AppLogger.get(__name__)

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

_meta = MetaData()

_workspaces = Table(
    "workspaces", _meta,
    Column("id",                  String, primary_key=True),
    Column("name",                String, nullable=False),
    Column("description",         Text,   nullable=False, default=""),
    Column("notion_workspace_id", String),
    Column("created_at",          String, nullable=False),
    Column("metadata_json",       Text,   nullable=False, default="{}"),
)

_pages = Table(
    "pages", _meta,
    Column("id",             String,  primary_key=True),
    Column("workspace_id",   String,  nullable=False),
    Column("title",          String,  nullable=False),
    Column("source_url",     Text,    nullable=False, default=""),
    Column("document_type",  String,  nullable=False, default="text"),
    Column("language",       String,  nullable=False, default=""),
    Column("project",        String,  nullable=False, default=""),
    Column("tags",           Text,    nullable=False, default=""),
    Column("importance",     Float,   nullable=False, default=0.0),
    Column("notion_page_id", String),
    Column("parent_page_id", String),
    Column("content_hash",   String,  nullable=False, default=""),
    Column("created_time",   String,  nullable=False),
    Column("updated_time",   String,  nullable=False),
    Column("last_synced_at", String),
    Column("metadata_json",  Text,    nullable=False, default="{}"),
    Index("idx_pages_workspace", "workspace_id"),
    Index("idx_pages_notion",    "notion_page_id"),
    Index("idx_pages_hash",      "content_hash"),
)

_blocks = Table(
    "blocks", _meta,
    Column("id",              String,  primary_key=True),
    Column("page_id",         String,  nullable=False),
    Column("block_type",      String,  nullable=False),
    Column("content",         Text,    nullable=False, default=""),
    Column("block_order",     Integer, nullable=False),
    Column("parent_block_id", String),
    Column("notion_block_id", String),
    Column("metadata_json",   Text,    nullable=False, default="{}"),
    Index("idx_blocks_page",   "page_id"),
    Index("idx_blocks_order",  "page_id", "block_order"),
    Index("idx_blocks_notion", "notion_block_id"),
)

_document_versions = Table(
    "document_versions", _meta,
    Column("id",           String,  primary_key=True),
    Column("page_id",      String,  nullable=False),
    Column("version",      Integer, nullable=False),
    Column("content_hash", String,  nullable=False),
    Column("created_at",   String,  nullable=False),
    Column("change_type",  String,  nullable=False),
    Column("chunk_count",  Integer, nullable=False, default=0),
    Column("diff_summary", Text,    nullable=False, default=""),
    UniqueConstraint("page_id", "version", name="uq_versions_page_version"),
    Index("idx_versions_page", "page_id"),
)

_sync_cursors = Table(
    "sync_cursors", _meta,
    Column("key",        String, primary_key=True),
    Column("cursor",     String),
    Column("updated_at", String, nullable=False),
)


def _make_engine(db_path: str) -> Engine:
    """Create a SQLAlchemy engine with WAL mode and foreign-key enforcement."""
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine

# Columns updated on conflict for upsert_page (excludes id, workspace_id, created_time)
_PAGE_UPDATE_COLS = (
    "title", "source_url", "document_type", "language", "project",
    "tags", "importance", "notion_page_id", "parent_page_id",
    "content_hash", "updated_time", "last_synced_at", "metadata_json",
)


class RawStore:
    """SQLite-backed raw storage layer (SQLAlchemy Core).

    Usage
    -----
        store = RawStore("./my_db/raw.db")

        ws  = Workspace.new("My Workspace")
        store.upsert_workspace(ws)

        pg  = Page.new(ws.id, "My Page", "https://notion.so/...")
        store.upsert_page(pg)

        blk = Block.new(pg.id, BlockType.PARAGRAPH, "Hello world", 0)
        store.upsert_blocks([blk])
    """

    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = _make_engine(str(path))
        _meta.create_all(self._engine)
        log.info("RawStore ready at %s", path)

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def upsert_workspace(self, ws: Workspace) -> None:
        ins = sqlite_insert(_workspaces).values(
            id=ws.id,
            name=ws.name,
            description=ws.description,
            notion_workspace_id=ws.notion_workspace_id,
            created_at=ws.created_at.isoformat(),
            metadata_json=json.dumps(ws.metadata),
        )
        stmt = ins.on_conflict_do_update(
            index_elements=["id"],
            set_={c: ins.excluded[c] for c in ("name", "description", "notion_workspace_id", "metadata_json")},
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)
        log.debug("upsert_workspace: %s (%s)", ws.name, ws.id[:8])

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        stmt = select(_workspaces).where(_workspaces.c.id == workspace_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return self._row_to_workspace(row) if row else None

    def list_workspaces(self) -> list[Workspace]:
        stmt = select(_workspaces).order_by(_workspaces.c.name)
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [self._row_to_workspace(r) for r in rows]

    @staticmethod
    def _row_to_workspace(row) -> Workspace:
        return Workspace(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            notion_workspace_id=row["notion_workspace_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    # ------------------------------------------------------------------
    # Page
    # ------------------------------------------------------------------

    def upsert_page(self, page: Page, content_hash: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        ins = sqlite_insert(_pages).values(
            id=page.id,
            workspace_id=page.workspace_id,
            title=page.title,
            source_url=page.source,
            document_type=page.document_type,
            language=page.metadata.get("language", ""),
            project=page.metadata.get("project", ""),
            tags=",".join(page.tags),
            importance=page.importance,
            notion_page_id=page.notion_page_id,
            parent_page_id=page.parent_page_id,
            content_hash=content_hash,
            created_time=page.created_time.isoformat(),
            updated_time=page.updated_time.isoformat(),
            last_synced_at=now,
            metadata_json=json.dumps(page.metadata),
        )
        stmt = ins.on_conflict_do_update(
            index_elements=["id"],
            set_={c: ins.excluded[c] for c in _PAGE_UPDATE_COLS},
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)
        log.debug("upsert_page: %r (%s)", page.title, page.id[:8])

    def get_page(self, page_id: str) -> Page | None:
        stmt = select(_pages).where(_pages.c.id == page_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return self._row_to_page(row) if row else None

    def get_page_by_notion_id(self, notion_page_id: str) -> Page | None:
        stmt = select(_pages).where(_pages.c.notion_page_id == notion_page_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return self._row_to_page(row) if row else None

    def get_content_hash(self, page_id: str) -> str | None:
        """Return the stored content_hash for *page_id*, or None if not stored."""
        stmt = select(_pages.c.content_hash).where(_pages.c.id == page_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).first()
        return row[0] if row else None

    def list_pages(self, workspace_id: str | None = None) -> list[Page]:
        stmt = select(_pages).order_by(_pages.c.title)
        if workspace_id:
            stmt = stmt.where(_pages.c.workspace_id == workspace_id)
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [self._row_to_page(r) for r in rows]

    @staticmethod
    def _row_to_page(row) -> Page:
        meta = json.loads(row["metadata_json"] or "{}")
        return Page(
            id=row["id"],
            workspace_id=row["workspace_id"],
            title=row["title"],
            source=row["source_url"],
            document_type=row["document_type"],
            tags=[t for t in (row["tags"] or "").split(",") if t],
            importance=float(row["importance"]),
            notion_page_id=row["notion_page_id"],
            parent_page_id=row["parent_page_id"],
            created_time=datetime.fromisoformat(row["created_time"]),
            updated_time=datetime.fromisoformat(row["updated_time"]),
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Block
    # ------------------------------------------------------------------

    def upsert_blocks(self, blocks: list[Block]) -> None:
        """Insert-or-replace a batch of Blocks for a Page."""
        if not blocks:
            return
        rows = [
            dict(
                id=b.id,
                page_id=b.page_id,
                block_type=b.block_type.value if hasattr(b.block_type, "value") else str(b.block_type),
                content=b.content,
                block_order=b.order,
                parent_block_id=b.parent_block_id,
                notion_block_id=b.notion_block_id,
                metadata_json=json.dumps(b.metadata),
            )
            for b in blocks
        ]
        stmt = sqlite_insert(_blocks).on_conflict_do_update(
            index_elements=["id"],
            set_=dict(
                block_type=sqlite_insert(_blocks).excluded.block_type,
                content=sqlite_insert(_blocks).excluded.content,
                block_order=sqlite_insert(_blocks).excluded.block_order,
                parent_block_id=sqlite_insert(_blocks).excluded.parent_block_id,
                notion_block_id=sqlite_insert(_blocks).excluded.notion_block_id,
                metadata_json=sqlite_insert(_blocks).excluded.metadata_json,
            ),
        )
        with self._engine.begin() as conn:
            conn.execute(stmt, rows)
        log.debug("upsert_blocks: %d blocks for page %s", len(blocks), blocks[0].page_id[:8])

    def get_blocks(self, page_id: str) -> list[Block]:
        """Return all Blocks for *page_id* sorted by block_order."""
        stmt = (
            select(_blocks)
            .where(_blocks.c.page_id == page_id)
            .order_by(_blocks.c.block_order)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [self._row_to_block(r) for r in rows]

    def delete_blocks(self, page_id: str) -> int:
        """Delete all Blocks for *page_id*. Returns the number deleted."""
        stmt = delete(_blocks).where(_blocks.c.page_id == page_id)
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
        count = result.rowcount
        log.debug("delete_blocks: %d removed for page %s", count, page_id[:8])
        return count

    @staticmethod
    def _row_to_block(row) -> Block:
        try:
            btype = BlockType(row["block_type"])
        except ValueError:
            btype = BlockType.UNSUPPORTED
        return Block(
            id=row["id"],
            page_id=row["page_id"],
            block_type=btype,
            content=row["content"],
            order=row["block_order"],
            parent_block_id=row["parent_block_id"],
            notion_block_id=row["notion_block_id"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    # ------------------------------------------------------------------
    # DocumentVersion
    # ------------------------------------------------------------------

    def record_version(self, version: DocumentVersion) -> None:
        stmt = (
            sqlite_insert(_document_versions)
            .values(
                id=version.id,
                page_id=version.page_id,
                version=version.version,
                content_hash=version.content_hash,
                created_at=version.created_at.isoformat(),
                change_type=(
                    version.change_type.value
                    if hasattr(version.change_type, "value")
                    else str(version.change_type)
                ),
                chunk_count=version.chunk_count,
                diff_summary=version.diff_summary,
            )
            .on_conflict_do_nothing()
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)

    def latest_version(self, page_id: str) -> DocumentVersion | None:
        """Return the most recent DocumentVersion for *page_id*, or None."""
        stmt = (
            select(_document_versions)
            .where(_document_versions.c.page_id == page_id)
            .order_by(_document_versions.c.version.desc())
            .limit(1)
        )
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        if not row:
            return None
        return DocumentVersion(
            id=row["id"],
            page_id=row["page_id"],
            version=row["version"],
            content_hash=row["content_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            change_type=ChangeType(row["change_type"]),
            chunk_count=row["chunk_count"],
            diff_summary=row["diff_summary"] or "",
        )

    def next_version_number(self, page_id: str) -> int:
        """Return the next version number for *page_id* (starts at 1)."""
        stmt = select(func.max(_document_versions.c.version)).where(
            _document_versions.c.page_id == page_id
        )
        with self._engine.connect() as conn:
            result = conn.execute(stmt).scalar()
        return (result or 0) + 1

    # ------------------------------------------------------------------
    # Sync cursor (for incremental Notion polling)
    # ------------------------------------------------------------------

    def get_cursor(self, key: str) -> str | None:
        stmt = select(_sync_cursors.c.cursor).where(_sync_cursors.c.key == key)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).first()
        return row[0] if row else None

    def set_cursor(self, key: str, cursor: str | None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        stmt = (
            sqlite_insert(_sync_cursors)
            .values(key=key, cursor=cursor, updated_at=now)
            .on_conflict_do_update(
                index_elements=["key"],
                set_=dict(cursor=cursor, updated_at=now),
            )
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._engine.connect() as conn:
            ws_count  = conn.execute(select(func.count()).select_from(_workspaces)).scalar()
            pg_count  = conn.execute(select(func.count()).select_from(_pages)).scalar()
            blk_count = conn.execute(select(func.count()).select_from(_blocks)).scalar()
            ver_count = conn.execute(select(func.count()).select_from(_document_versions)).scalar()
        return {
            "workspaces":        ws_count,
            "pages":             pg_count,
            "blocks":            blk_count,
            "document_versions": ver_count,
        }
