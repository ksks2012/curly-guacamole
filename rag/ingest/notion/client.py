"""
Notion API client — thin wrapper around the Notion REST API.

Handles:
    - Authentication (Bearer token)
    - Pagination (next_cursor loop)
    - Rate-limit retry (HTTP 429, Retry-After header)
    - Response → domain model conversion (Page, Block)

Only the endpoints needed for the sync pipeline are implemented:
    GET  /v1/search                   — list all pages
    GET  /v1/pages/{id}               — fetch one page + properties
    GET  /v1/blocks/{id}/children     — fetch block children (paginated)

Reference: https://developers.notion.com/reference/block
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Iterator

import requests

from utils.logger import AppLogger
from rag.knowledge.models import Block, BlockType, Page, Workspace

log = AppLogger.get(__name__)

_BASE_URL = "https://api.notion.com/v1"
_NOTION_VERSION = "2026-03-11"
_MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# BlockType mapping  (Notion API → BlockType enum)
# ---------------------------------------------------------------------------

_NOTION_TO_BLOCK_TYPE: dict[str, BlockType] = {
    "paragraph":           BlockType.PARAGRAPH,
    "heading_1":           BlockType.HEADING_1,
    "heading_2":           BlockType.HEADING_2,
    "heading_3":           BlockType.HEADING_3,
    "heading_4":           BlockType.HEADING_4,
    "bulleted_list_item":  BlockType.BULLETED_LIST_ITEM,
    "numbered_list_item":  BlockType.NUMBERED_LIST_ITEM,
    "to_do":               BlockType.TO_DO,
    "toggle":              BlockType.TOGGLE,
    "code":                BlockType.CODE,
    "quote":               BlockType.QUOTE,
    "callout":             BlockType.CALLOUT,
    "equation":            BlockType.EQUATION,
    "table":               BlockType.TABLE,
    "table_row":           BlockType.TABLE_ROW,
    "divider":             BlockType.DIVIDER,
    "table_of_contents":   BlockType.TABLE_OF_CONTENTS,
    "child_page":          BlockType.CHILD_PAGE,
    "child_database":      BlockType.CHILD_DATABASE,
    "image":               BlockType.IMAGE,
    "video":               BlockType.VIDEO,
    "audio":               BlockType.AUDIO,
    "file":                BlockType.FILE,
    "pdf":                 BlockType.PDF,
    "bookmark":            BlockType.BOOKMARK,
    "embed":               BlockType.EMBED,
}


def _block_type(notion_type: str) -> BlockType:
    return _NOTION_TO_BLOCK_TYPE.get(notion_type, BlockType.UNSUPPORTED)


# ---------------------------------------------------------------------------
# Plain-text extraction from Notion rich_text arrays
# ---------------------------------------------------------------------------

def _plain_text(rich_text_list: list) -> str:
    """Concatenate plain_text fields from a Notion rich_text array."""
    return "".join(item.get("plain_text", "") for item in rich_text_list)


def _extract_block_content(block_type: str, block_data: dict) -> tuple[str, dict]:
    """Return (plain_text_content, extra_metadata) for a Notion block dict."""
    content = ""
    extra: dict = {}

    type_obj = block_data.get(block_type, {})
    if not isinstance(type_obj, dict):
        return content, extra

    rich = type_obj.get("rich_text", [])
    if rich:
        content = _plain_text(rich)

    if block_type == "code":
        extra["language"] = type_obj.get("language", "")
        caption = _plain_text(type_obj.get("caption", []))
        if caption:
            extra["caption"] = caption

    elif block_type in ("heading_1", "heading_2", "heading_3", "heading_4"):
        level_map = {"heading_1": 1, "heading_2": 2, "heading_3": 3, "heading_4": 4}
        extra["level"] = level_map.get(block_type, 0)
        extra["is_toggleable"] = type_obj.get("is_toggleable", False)

    elif block_type == "to_do":
        extra["checked"] = type_obj.get("checked", False)

    elif block_type == "table":
        extra["table_width"] = type_obj.get("table_width", 0)
        extra["has_column_header"] = type_obj.get("has_column_header", False)
        extra["has_row_header"] = type_obj.get("has_row_header", False)

    elif block_type == "table_row":
        cells = [
            _plain_text(cell) for cell in type_obj.get("cells", [])
        ]
        content = " | ".join(cells)
        extra["cells"] = cells

    elif block_type == "child_page":
        content = type_obj.get("title", "")

    elif block_type == "child_database":
        content = type_obj.get("title", "")

    elif block_type in ("image", "video", "audio", "file", "pdf"):
        file_obj = type_obj.get("external") or type_obj.get("file") or {}
        content = file_obj.get("url", "")
        caption = _plain_text(type_obj.get("caption", []))
        if caption:
            extra["caption"] = caption

    elif block_type == "bookmark":
        content = type_obj.get("url", "")

    elif block_type == "embed":
        content = type_obj.get("url", "")

    elif block_type == "equation":
        content = type_obj.get("expression", "")

    return content, extra


# ---------------------------------------------------------------------------
# NotionClient
# ---------------------------------------------------------------------------

class NotionClient:
    """Minimal Notion REST API client for the sync pipeline.

    Args:
        token : Notion integration secret (Bearer token).
    """

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("Notion integration token is required.")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Notion-Version": _NOTION_VERSION,
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    def test_connection(self) -> dict:
        """Call GET /v1/users/me to verify the token is valid.

        Returns the raw API response dict on success.
        Raises requests.HTTPError on authentication failure (HTTP 401/403).
        """
        result = self._get("/users/me")
        log.info(
            "Notion token OK — bot name: %s  id: %s",
            result.get("name", "?"),
            result.get("id", "?"),
        )
        return result

    # ------------------------------------------------------------------
    # Low-level HTTP with retry
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{_BASE_URL}{path}"
        for attempt in range(1, _MAX_RETRIES + 1):
            resp = self._session.get(url, params=params or {})
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "2"))
                log.warning("Rate limited — retrying in %ds (attempt %d)", wait, attempt)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"GET {url} failed after {_MAX_RETRIES} retries (rate limit)")

    def _post(self, path: str, body: dict | None = None) -> dict:
        url = f"{_BASE_URL}{path}"
        for attempt in range(1, _MAX_RETRIES + 1):
            resp = self._session.post(url, json=body or {})
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "2"))
                log.warning("Rate limited — retrying in %ds (attempt %d)", wait, attempt)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"POST {url} failed after {_MAX_RETRIES} retries (rate limit)")

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def get_database(self, database_id: str) -> dict:
        """Fetch a Notion database object (GET /v1/databases/{id}).

        Returns the raw API response dict, which includes the ``data_sources``
        array needed to obtain a ``data_source_id`` for page queries.
        """
        return self._get(f"/databases/{database_id}")

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def iter_all_pages(
        self,
        start_cursor: str | None = None,
    ) -> Iterator[tuple[dict, str | None]]:
        """Yield (raw_page_dict, next_cursor) pairs via /v1/search.

        Yields one tuple per API response page (each containing up to 100
        Notion pages).  *next_cursor* is None on the last page.

        Args:
            start_cursor : Resume from this cursor for incremental sync.
        """
        cursor = start_cursor
        while True:
            body: dict = {
                "filter": {"value": "page", "property": "object"},
                "sort":   {"direction": "descending", "timestamp": "last_edited_time"},
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor

            data = self._post("/search", body)
            results = data.get("results", [])
            next_cursor = data.get("next_cursor")

            yield results, next_cursor

            if not data.get("has_more") or not next_cursor:
                break
            cursor = next_cursor

    def get_page_raw(self, notion_page_id: str) -> dict:
        """Fetch a single Notion page object."""
        return self._get(f"/pages/{notion_page_id}")

    def get_page_markdown(self, page_id: str) -> str:
        """Fetch a page's full content as Markdown (GET /v1/pages/{id}/markdown).

        Returns the markdown string from the ``markdown`` field of the response.
        Raises ``requests.HTTPError`` if the page is not found or access is denied.
        """
        data = self._get(f"/pages/{page_id}/markdown")
        return data.get("markdown", "")

    def iter_data_source_pages(
        self,
        data_source_id: str,
        start_cursor: str | None = None,
    ) -> Iterator[tuple[list[dict], str | None]]:
        """Yield (raw_page_list, next_cursor) pairs from a data source query.

        Calls POST /v1/data_sources/{data_source_id}/query and follows
        pagination via ``next_cursor`` until all pages are retrieved.

        Args:
            data_source_id : The data source UUID (from the database object).
            start_cursor   : Resume pagination from this cursor value.
        """
        cursor = start_cursor
        while True:
            body: dict = {}
            if cursor:
                body["start_cursor"] = cursor

            data = self._post(f"/data_sources/{data_source_id}/query", body)
            results = data.get("results", [])
            next_cursor = data.get("next_cursor")

            yield results, next_cursor

            if not data.get("has_more") or not next_cursor:
                break
            cursor = next_cursor

    # ------------------------------------------------------------------
    # Blocks
    # ------------------------------------------------------------------

    def iter_block_children(
        self, block_id: str
    ) -> Iterator[list[dict]]:
        """Yield lists of raw block dicts for *block_id* (paginated).

        Automatically follows `next_cursor` to retrieve all children.
        """
        cursor: str | None = None
        while True:
            params: dict = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            data = self._get(f"/blocks/{block_id}/children", params=params)
            yield data.get("results", [])
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

    def get_all_blocks(self, page_id: str) -> list[dict]:
        """Recursively fetch all blocks for *page_id* (depth-first)."""
        blocks: list[dict] = []
        self._collect_blocks(page_id, blocks)
        return blocks

    def _collect_blocks(
        self,
        block_id: str,
        out: list[dict],
        depth: int = 0,
    ) -> None:
        """Depth-first collection of all blocks under *block_id*."""
        for children in self.iter_block_children(block_id):
            for blk in children:
                blk["_depth"] = depth
                out.append(blk)
                if blk.get("has_children") and blk.get("type") not in (
                    "child_database",
                ):
                    self._collect_blocks(blk["id"], out, depth + 1)

    # ------------------------------------------------------------------
    # Conversion: raw Notion dicts → domain models
    # ------------------------------------------------------------------

    @staticmethod
    def raw_page_to_model(
        raw: dict,
        workspace_id: str,
        workspace_name: str = "",
    ) -> Page:
        """Convert a raw Notion page dict to a Page domain model."""
        notion_id = raw["id"]
        props = raw.get("properties", {})

        # Extract title — Notion stores it in a "title" type property
        title = ""
        for prop in props.values():
            if prop.get("type") == "title":
                title = _plain_text(prop.get("title", []))
                break
        if not title:
            # Fallback: child_page / child_database carry title at top level
            title = (
                raw.get("child_page", {}).get("title")
                or raw.get("child_database", {}).get("title")
                or notion_id
            )

        source_url = raw.get("url", f"https://notion.so/{notion_id.replace('-', '')}")

        created = raw.get("created_time", datetime.now(timezone.utc).isoformat())
        updated = raw.get("last_edited_time", created)

        # Tags — look for the 'Tags' property explicitly first (by key name,
        # case-insensitive), then fall back to the first multi_select found.
        # This is robust against pages that have multiple multi_select
        # properties (e.g. Priority, Status-as-multi-select, etc.).
        # To support additional multi_select properties in the future, add
        # more named lookups here before the fallback.
        tags: list[str] = []
        _tag_prop = (
            props.get("Tags")
            or props.get("tags")
            or next(
                (p for p in props.values() if p.get("type") == "multi_select"),
                None,
            )
        )
        if _tag_prop and _tag_prop.get("type") == "multi_select":
            tags = [opt["name"] for opt in _tag_prop.get("multi_select", [])]

        return Page(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, source_url)),
            workspace_id=workspace_id,
            title=title,
            source=source_url,
            document_type="notion",
            tags=tags,
            notion_page_id=notion_id,
            parent_page_id=None,
            created_time=datetime.fromisoformat(created.replace("Z", "+00:00")),
            updated_time=datetime.fromisoformat(updated.replace("Z", "+00:00")),
            metadata={"notion_raw": raw},
        )

    @staticmethod
    def raw_blocks_to_models(
        raw_blocks: list[dict],
        page_id: str,
    ) -> list[Block]:
        """Convert a flat list of raw Notion block dicts to Block models.

        Blocks are assigned a depth-first ``order`` index based on their
        position in *raw_blocks* (which is already depth-first from
        ``get_all_blocks``).
        """
        models: list[Block] = []
        for order, raw in enumerate(raw_blocks):
            notion_id = raw.get("id", "")
            btype_str = raw.get("type", "unsupported")
            btype = _block_type(btype_str)
            content, extra = _extract_block_content(btype_str, raw)

            # parent block id
            parent = raw.get("parent", {})
            parent_block_id: str | None = None
            if parent.get("type") == "block_id":
                parent_block_id = parent.get("block_id")

            block = Block(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{page_id}:{notion_id}")),
                page_id=page_id,
                block_type=btype,
                content=content,
                order=order,
                parent_block_id=parent_block_id,
                notion_block_id=notion_id,
                metadata=extra,
            )
            models.append(block)
        return models
