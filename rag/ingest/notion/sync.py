"""
Step 0.3 — Notion Sync Pipeline.

Orchestrates:
    1. List all pages via data source query (or /search as fallback)
    2. For each page: fetch full content as Markdown via /v1/pages/{id}/markdown
    3. Compute content_hash; skip pages whose hash has not changed
    4. Store raw Page + Block data in RawStore (SQLite)
    5. Record a DocumentVersion for change tracking
    6. Persist the next_cursor so the next run starts from where this one stopped

This layer writes to the Raw Storage Layer ONLY.
Chunking → embedding → Chroma is handled by a separate step (Phase 1).

Usage
-----
    from rag.ingest.notion.sync import NotionSyncPipeline

    pipeline = NotionSyncPipeline(
        token="secret_...",
        workspace=workspace,       # rag.knowledge.models.Workspace
        store=raw_store,           # rag.knowledge.store.RawStore
        data_source_id="...",      # obtain from NotionClient.get_database()
    )
    result = pipeline.sync()
    print(result)
    # {"pages_seen": 42, "pages_updated": 3, "pages_skipped": 39, "errors": 0}
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

from utils.logger import AppLogger

from rag.knowledge.models import (
    Block,
    BlockType,
    ChangeType,
    DocumentVersion,
    Page,
    Workspace,
)
from rag.knowledge.store import RawStore
from rag.ingest.notion.client import NotionClient

log = AppLogger.get(__name__)

# Cursor key prefixes stored in sync_cursors table
_CURSOR_KEY_SEARCH      = "notion_sync:{workspace_id}"
_CURSOR_KEY_DATA_SOURCE = "notion_datasource:{data_source_id}"


@dataclass
class SyncResult:
    pages_seen:    int = 0
    pages_updated: int = 0
    pages_skipped: int = 0
    errors:        int = 0
    error_ids:     list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "pages_seen":    self.pages_seen,
            "pages_updated": self.pages_updated,
            "pages_skipped": self.pages_skipped,
            "errors":        self.errors,
        }


class NotionSyncPipeline:
    """Sync Notion pages into the Raw Storage Layer.

    Args:
        token          : Notion integration secret.
        workspace      : Target Workspace domain model.
        store          : RawStore instance to persist into.
        data_source_id : Data source UUID (from ``NotionClient.get_database()``).
                         When provided, pages are listed via
                         POST /v1/data_sources/{id}/query instead of /v1/search.
        progress_cb    : Optional callback ``(pages_seen: int, total: int | None)``
                         called after each page is processed.
        full_sync      : When True, ignore stored cursor and start from the
                         beginning (re-processes all pages).
    """

    def __init__(
        self,
        token: str,
        workspace: Workspace,
        store: RawStore,
        data_source_id: str | None = None,
        progress_cb: Callable[[int, int | None], None] | None = None,
        full_sync: bool = False,
    ) -> None:
        self._client         = NotionClient(token)
        self._workspace      = workspace
        self._store          = store
        self._data_source_id = data_source_id
        self._progress_cb    = progress_cb
        self._full_sync      = full_sync

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(self) -> SyncResult:
        """Run the sync pipeline and return a SyncResult."""
        log.info(
            "NotionSyncPipeline.sync: workspace=%r  data_source=%s  full_sync=%s",
            self._workspace.name,
            self._data_source_id or "<search>",
            self._full_sync,
        )
        self._store.upsert_workspace(self._workspace)

        if self._data_source_id:
            cursor_key = _CURSOR_KEY_DATA_SOURCE.format(
                data_source_id=self._data_source_id
            )
            page_iter = self._client.iter_data_source_pages(
                self._data_source_id,
                start_cursor=None if self._full_sync else self._store.get_cursor(cursor_key),
            )
        else:
            cursor_key = _CURSOR_KEY_SEARCH.format(workspace_id=self._workspace.id)
            page_iter = self._client.iter_all_pages(
                start_cursor=None if self._full_sync else self._store.get_cursor(cursor_key),
            )

        log.info("  cursor_key: %s", cursor_key)

        result = SyncResult()

        for raw_pages_batch, next_cursor in page_iter:
            for raw_page in raw_pages_batch:
                result.pages_seen += 1
                try:
                    updated = self._sync_one_page(raw_page)
                    if updated:
                        result.pages_updated += 1
                    else:
                        result.pages_skipped += 1
                except Exception as exc:
                    result.errors += 1
                    result.error_ids.append(raw_page.get("id", "?"))
                    log.error(
                        "  Error syncing page %s: %s",
                        raw_page.get("id", "?"), exc, exc_info=True,
                    )

                if self._progress_cb:
                    self._progress_cb(result.pages_seen, None)

            # Persist cursor after every API response batch so partial runs
            # can resume from the latest successfully processed position.
            if next_cursor:
                self._store.set_cursor(cursor_key, next_cursor)

        # Cursor = None means "next run starts from the beginning (full refresh)"
        # which is the safe default after a completed sync.
        self._store.set_cursor(cursor_key, None)

        log.info(
            "Sync complete: seen=%d  updated=%d  skipped=%d  errors=%d",
            result.pages_seen, result.pages_updated,
            result.pages_skipped, result.errors,
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_one_page(self, raw_page: dict) -> bool:
        """Sync a single Notion page.  Returns True if the page was updated."""
        notion_page_id = raw_page["id"]

        # Convert raw API response → Page model
        page = NotionClient.raw_page_to_model(
            raw_page,
            workspace_id=self._workspace.id,
            workspace_name=self._workspace.name,
        )

        # Fetch page content as Markdown
        markdown = self._client.get_page_markdown(notion_page_id)

        # Compute content hash from markdown text
        new_hash = hashlib.sha256(markdown.encode()).hexdigest() if markdown else ""

        # Change detection: compare against stored hash
        stored_hash = self._store.get_content_hash(page.id)
        if stored_hash and stored_hash == new_hash:
            log.debug("  skip (unchanged): %r", page.title)
            return False

        # Determine change type
        change_type = ChangeType.CREATED if stored_hash is None else ChangeType.UPDATED

        # Represent the full markdown content as a single block
        blocks: list[Block] = []
        if markdown:
            blocks = [
                Block.new(
                    page_id=page.id,
                    block_type=BlockType.LOCAL_PAGE_TEXT,
                    content=markdown,
                    order=0,
                    metadata={"source": "notion_markdown"},
                )
            ]

        # Persist page + blocks
        self._store.upsert_page(page, content_hash=new_hash)
        if blocks:
            self._store.delete_blocks(page.id)
            self._store.upsert_blocks(blocks)

        # Record version
        version_num = self._store.next_version_number(page.id)
        version = DocumentVersion.new(
            page_id=page.id,
            version=version_num,
            content_hash=new_hash,
            change_type=change_type,
            chunk_count=0,  # updated by the chunking step later
        )
        self._store.record_version(version)

        log.info(
            "  %s: %r  v%d  blocks=%d",
            change_type.value, page.title, version_num, len(blocks),
        )
        return True
