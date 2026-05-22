"""
Notion API endpoint verification script.

Tests each API endpoint used by the sync pipeline in order:
    1. GET  /v1/users/me                   — token validity
    2. GET  /v1/databases/{id}             — database metadata + data_source_id
    3. POST /v1/data_sources/{id}/query    — page listing
    4. GET  /v1/pages/{id}/markdown        — page content as Markdown
    5. GET  /v1/blocks/{id}/children       — first level of block tree

IDs are resolved in this priority order:
    CLI argument  >  etc/config.yaml  >  auto-resolved from a previous step

Responses are saved to ./data/ for offline inspection and test fixtures.

Usage:
    python testing/testing_notion_api.py [database_id [data_source_id [page_id [block_id]]]]
"""

import json
import sys
from pathlib import Path

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.ingest.notion.client import NotionClient

_DATA_DIR = Path("data")
_PREVIEW_LEN = 400


def _save(filename: str, data: dict | list | str) -> None:
    """Pretty-print *data* as JSON and write it to ./data/{filename}."""
    _DATA_DIR.mkdir(exist_ok=True)
    path = _DATA_DIR / filename
    if isinstance(data, (dict, list)):
        text = json.dumps(data, ensure_ascii=False, indent=4)
    else:
        text = str(data)
    path.write_text(text, encoding="utf-8")
    print(f"    saved → {path}")


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str, exc: Exception) -> None:
    print(f"  [FAIL] {msg}: {exc}")


def main() -> None:
    config = AppConfig()
    AppLogger.setup(level="WARNING")  # suppress INFO logs for cleaner output

    token = config.notion_token
    if not token:
        print("ERROR: notion_token is not set in etc/config.yaml")
        sys.exit(1)

    # IDs: CLI args take priority over config; later steps auto-fill from responses
    database_id    = sys.argv[1] if len(sys.argv) > 1 else config.notion_database_id
    data_source_id = sys.argv[2] if len(sys.argv) > 2 else config.notion_data_source_id
    page_id        = sys.argv[3] if len(sys.argv) > 3 else ""
    block_id       = sys.argv[4] if len(sys.argv) > 4 else ""

    print(f"token          : {token[:12]}...")
    print(f"database_id    : {database_id or '(not set)'}")
    print(f"data_source_id : {data_source_id or '(not set)'}")

    client = NotionClient(token)
    failures: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: Token — GET /v1/users/me
    # ------------------------------------------------------------------
    _section("Step 1: token  GET /v1/users/me")
    try:
        me = client.test_connection()
        _save("me.json", me)
        bot   = me.get("bot", {})
        owner = bot.get("owner", {})
        print(f"  type : {me.get('type', '?')}")
        print(f"  id   : {me.get('id', '?')}")
        print(f"  name : {me.get('name', '?')}")
        print(f"  owner: {owner.get('type', '?')}")
        _ok("token is valid")
    except Exception as exc:
        _fail("token check", exc)
        sys.exit(1)  # no point continuing without a valid token

    # ------------------------------------------------------------------
    # Step 2: Database — GET /v1/databases/{id}
    # ------------------------------------------------------------------
    _section("Step 2: database  GET /v1/databases/{id}")
    if not database_id:
        print("  SKIP — notion_database_id not set in config or CLI")
    else:
        try:
            db = client.get_database(database_id)
            _save("database.json", db)
            title   = "".join(t.get("plain_text", "") for t in db.get("title", []))
            sources = db.get("data_sources", [])
            print(f"  title       : {title!r}")
            print(f"  data_sources: {[s['id'] for s in sources]}")
            # Auto-fill data_source_id from the first entry when not already set
            if not data_source_id and sources:
                data_source_id = sources[0]["id"]
                print(f"  → resolved data_source_id: {data_source_id}")
            _ok("database retrieved")
        except Exception as exc:
            _fail("database retrieval", exc)
            failures.append("database")

    # ------------------------------------------------------------------
    # Step 3: Data source query — POST /v1/data_sources/{id}/query
    # ------------------------------------------------------------------
    _section("Step 3: data source query  POST /v1/data_sources/{id}/query")
    if not data_source_id:
        print("  SKIP — data_source_id not available")
    else:
        try:
            batch, next_cursor = next(iter(client.iter_data_source_pages(data_source_id)))
            _save("data_source.json", {
                "object":      "list",
                "results":     batch,
                "next_cursor": next_cursor,
                "has_more":    next_cursor is not None,
            })
            print(f"  pages in first batch: {len(batch)}")
            for p in batch:
                props      = p.get("properties", {})
                page_title = ""
                for prop in props.values():
                    if prop.get("type") == "title":
                        page_title = "".join(
                            t.get("plain_text", "") for t in prop.get("title", [])
                        )
                        break
                print(f"    - {p['id']}  {page_title!r}")
            # Auto-fill page_id from first result when not already set
            if not page_id and batch:
                page_id = batch[0]["id"]
                print(f"  → using page_id: {page_id}")
            _ok(f"data source returned {len(batch)} page(s)")
        except Exception as exc:
            _fail("data source query", exc)
            failures.append("data_source")

    # ------------------------------------------------------------------
    # Step 4: Page markdown — GET /v1/pages/{id}/markdown
    # ------------------------------------------------------------------
    _section("Step 4: page markdown  GET /v1/pages/{id}/markdown")
    if not page_id:
        print("  SKIP — page_id not available (provide as CLI arg or set data_source_id)")
    else:
        try:
            markdown = client.get_page_markdown(page_id)
            _save("page_markdown.json", {"page_id": page_id, "markdown": markdown})
            preview = markdown[:_PREVIEW_LEN].replace("\n", " ")
            print(f"  page_id : {page_id}")
            print(f"  chars   : {len(markdown)}")
            print(f"  preview : {preview}...")
            # Auto-fill block_id from page_id (pages are valid block IDs)
            if not block_id:
                block_id = page_id
            _ok("markdown retrieved")
        except Exception as exc:
            _fail("markdown retrieval", exc)
            failures.append("markdown")

    # ------------------------------------------------------------------
    # Step 5: Block children — GET /v1/blocks/{id}/children
    # ------------------------------------------------------------------
    _section("Step 5: block children  GET /v1/blocks/{id}/children")
    if not block_id:
        print("  SKIP — block_id not available")
    else:
        try:
            first_batch = next(iter(client.iter_block_children(block_id)), [])
            _save("block_children.json", {"block_id": block_id, "results": first_batch})
            print(f"  block_id : {block_id}")
            print(f"  children : {len(first_batch)}")
            for blk in first_batch[:5]:
                print(f"    - [{blk.get('type', '?'):25s}]  {blk.get('id', '?')}")
            if len(first_batch) > 5:
                print(f"    ... and {len(first_batch) - 5} more")
            _ok(f"block children retrieved ({len(first_batch)} items)")
        except Exception as exc:
            _fail("block children retrieval", exc)
            failures.append("block_children")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if failures:
        print(f"FAILED steps: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All steps passed.")


if __name__ == "__main__":
    main()
