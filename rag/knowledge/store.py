"""
Step 0.3 — Raw Storage Layer (SQLite).

Stores the raw Notion / local-file data before it enters the vector store.
This layer enables:
    - Rebuilding chunks without re-fetching from Notion
    - Swapping embedding models (re-embed from stored raw text)
    - Incremental sync (compare stored hash vs. live hash to skip unchanged pages)
    - Future graph extraction and knowledge evolution analysis

Schema
------
    workspaces      — one row per Workspace
    pages           — one row per Page; stores content_hash for change detection
    blocks          — one row per Block; raw content + metadata JSON
    sync_cursors    — stores Notion next_cursor values for incremental API polling

All datetimes are stored as ISO-8601 UTC strings.
JSON fields store arbitrary dicts serialised with json.dumps.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

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

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS workspaces (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    description         TEXT NOT NULL DEFAULT '',
    notion_workspace_id TEXT,
    created_at          TEXT NOT NULL,
    metadata_json       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS pages (
    id              TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL REFERENCES workspaces(id),
    title           TEXT NOT NULL,
    source_url      TEXT NOT NULL DEFAULT '',
    document_type   TEXT NOT NULL DEFAULT 'text',
    language        TEXT NOT NULL DEFAULT '',
    project         TEXT NOT NULL DEFAULT '',
    tags            TEXT NOT NULL DEFAULT '',     -- comma-joined
    importance      REAL NOT NULL DEFAULT 0.0,
    notion_page_id  TEXT,
    parent_page_id  TEXT REFERENCES pages(id),
    content_hash    TEXT NOT NULL DEFAULT '',
    created_time    TEXT NOT NULL,
    updated_time    TEXT NOT NULL,
    last_synced_at  TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_pages_workspace   ON pages(workspace_id);
CREATE INDEX IF NOT EXISTS idx_pages_notion      ON pages(notion_page_id);
CREATE INDEX IF NOT EXISTS idx_pages_hash        ON pages(content_hash);

CREATE TABLE IF NOT EXISTS blocks (
    id               TEXT PRIMARY KEY,
    page_id          TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    block_type       TEXT NOT NULL,
    content          TEXT NOT NULL DEFAULT '',
    block_order      INTEGER NOT NULL,
    parent_block_id  TEXT,
    notion_block_id  TEXT,
    metadata_json    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_blocks_page      ON blocks(page_id);
CREATE INDEX IF NOT EXISTS idx_blocks_order     ON blocks(page_id, block_order);
CREATE INDEX IF NOT EXISTS idx_blocks_notion    ON blocks(notion_block_id);

CREATE TABLE IF NOT EXISTS document_versions (
    id           TEXT PRIMARY KEY,
    page_id      TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    version      INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    change_type  TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    diff_summary TEXT NOT NULL DEFAULT '',
    UNIQUE(page_id, version)
);

CREATE INDEX IF NOT EXISTS idx_versions_page ON document_versions(page_id);

CREATE TABLE IF NOT EXISTS sync_cursors (
    key         TEXT PRIMARY KEY,   -- e.g. "notion_search" or workspace_id
    cursor      TEXT,               -- Notion next_cursor (NULL = start from beginning)
    updated_at  TEXT NOT NULL
);
"""


class RawStore:
    """SQLite-backed raw storage layer.

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
        self._db_path = str(path)
        self._init_schema()
        log.info("RawStore ready at %s", self._db_path)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def upsert_workspace(self, ws: Workspace) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO workspaces
                    (id, name, description, notion_workspace_id,
                     created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name                = excluded.name,
                    description         = excluded.description,
                    notion_workspace_id = excluded.notion_workspace_id,
                    metadata_json       = excluded.metadata_json
                """,
                (
                    ws.id, ws.name, ws.description,
                    ws.notion_workspace_id,
                    ws.created_at.isoformat(),
                    json.dumps(ws.metadata),
                ),
            )
        log.debug("upsert_workspace: %s (%s)", ws.name, ws.id[:8])

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
            ).fetchone()
        if not row:
            return None
        return Workspace(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            notion_workspace_id=row["notion_workspace_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def list_workspaces(self) -> list[Workspace]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM workspaces ORDER BY name").fetchall()
        return [self.get_workspace(r["id"]) for r in rows]  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Page
    # ------------------------------------------------------------------

    def upsert_page(self, page: Page, content_hash: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO pages
                    (id, workspace_id, title, source_url, document_type,
                     language, project, tags, importance, notion_page_id,
                     parent_page_id, content_hash,
                     created_time, updated_time, last_synced_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title          = excluded.title,
                    source_url     = excluded.source_url,
                    document_type  = excluded.document_type,
                    language       = excluded.language,
                    project        = excluded.project,
                    tags           = excluded.tags,
                    importance     = excluded.importance,
                    notion_page_id = excluded.notion_page_id,
                    parent_page_id = excluded.parent_page_id,
                    content_hash   = excluded.content_hash,
                    updated_time   = excluded.updated_time,
                    last_synced_at = excluded.last_synced_at,
                    metadata_json  = excluded.metadata_json
                """,
                (
                    page.id,
                    page.workspace_id,
                    page.title,
                    page.source,
                    page.document_type,
                    page.metadata.get("language", ""),
                    page.metadata.get("project", ""),
                    ",".join(page.tags),
                    page.importance,
                    page.notion_page_id,
                    page.parent_page_id,
                    content_hash,
                    page.created_time.isoformat(),
                    page.updated_time.isoformat(),
                    now,
                    json.dumps(page.metadata),
                ),
            )
        log.debug("upsert_page: %r (%s)", page.title, page.id[:8])

    def get_page(self, page_id: str) -> Page | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM pages WHERE id = ?", (page_id,)
            ).fetchone()
        return self._row_to_page(row) if row else None

    def get_page_by_notion_id(self, notion_page_id: str) -> Page | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM pages WHERE notion_page_id = ?",
                (notion_page_id,),
            ).fetchone()
        return self._row_to_page(row) if row else None

    def get_content_hash(self, page_id: str) -> str | None:
        """Return the stored content_hash for *page_id*, or None if not stored."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content_hash FROM pages WHERE id = ?", (page_id,)
            ).fetchone()
        return row["content_hash"] if row else None

    def list_pages(self, workspace_id: str | None = None) -> list[Page]:
        with self._conn() as conn:
            if workspace_id:
                rows = conn.execute(
                    "SELECT * FROM pages WHERE workspace_id = ? ORDER BY title",
                    (workspace_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM pages ORDER BY title"
                ).fetchall()
        return [self._row_to_page(r) for r in rows if r]  # type: ignore[misc]

    @staticmethod
    def _row_to_page(row: sqlite3.Row) -> Page:
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
            (
                b.id, b.page_id, b.block_type.value if hasattr(b.block_type, "value") else str(b.block_type),
                b.content, b.order,
                b.parent_block_id, b.notion_block_id,
                json.dumps(b.metadata),
            )
            for b in blocks
        ]
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT INTO blocks
                    (id, page_id, block_type, content, block_order,
                     parent_block_id, notion_block_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    block_type      = excluded.block_type,
                    content         = excluded.content,
                    block_order     = excluded.block_order,
                    parent_block_id = excluded.parent_block_id,
                    notion_block_id = excluded.notion_block_id,
                    metadata_json   = excluded.metadata_json
                """,
                rows,
            )
        log.debug("upsert_blocks: %d blocks for page %s", len(blocks), blocks[0].page_id[:8])

    def get_blocks(self, page_id: str) -> list[Block]:
        """Return all Blocks for *page_id* sorted by block_order."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM blocks WHERE page_id = ? ORDER BY block_order",
                (page_id,),
            ).fetchall()
        return [self._row_to_block(r) for r in rows]

    def delete_blocks(self, page_id: str) -> int:
        """Delete all Blocks for *page_id*. Returns the number deleted."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM blocks WHERE page_id = ?", (page_id,))
            count = cur.rowcount
        log.debug("delete_blocks: %d removed for page %s", count, page_id[:8])
        return count

    @staticmethod
    def _row_to_block(row: sqlite3.Row) -> Block:
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
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO document_versions
                    (id, page_id, version, content_hash, created_at,
                     change_type, chunk_count, diff_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.id, version.page_id, version.version,
                    version.content_hash, version.created_at.isoformat(),
                    version.change_type.value
                    if hasattr(version.change_type, "value")
                    else str(version.change_type),
                    version.chunk_count, version.diff_summary,
                ),
            )

    def latest_version(self, page_id: str) -> DocumentVersion | None:
        """Return the most recent DocumentVersion for *page_id*, or None."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM document_versions
                WHERE page_id = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (page_id,),
            ).fetchone()
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
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(version) AS v FROM document_versions WHERE page_id = ?",
                (page_id,),
            ).fetchone()
        return (row["v"] or 0) + 1

    # ------------------------------------------------------------------
    # Sync cursor (for incremental Notion polling)
    # ------------------------------------------------------------------

    def get_cursor(self, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT cursor FROM sync_cursors WHERE key = ?", (key,)
            ).fetchone()
        return row["cursor"] if row else None

    def set_cursor(self, key: str, cursor: str | None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sync_cursors (key, cursor, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    cursor     = excluded.cursor,
                    updated_at = excluded.updated_at
                """,
                (key, cursor, now),
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._conn() as conn:
            ws_count  = conn.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0]
            pg_count  = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            blk_count = conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
            ver_count = conn.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0]
        return {
            "workspaces":        ws_count,
            "pages":             pg_count,
            "blocks":            blk_count,
            "document_versions": ver_count,
        }
