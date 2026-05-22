"""
Step 0.3.5 — Verify Notion sync pipeline.

Runs NotionSyncPipeline end-to-end and checks that synced pages are stored
correctly in the RawStore with their Markdown content.

Usage:
    python testing/testing_sync.py

Config (etc/config.yaml):
    notion_token        : Notion integration secret
    notion_workspace_id : Logical workspace name (free string)
    notion_database_id  : Notion database UUID (used to resolve data_source_id)
    raw_db_path         : Path to the SQLite raw store (default: ./my_db/raw.db)
"""

import sys
import pytest

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.knowledge.models import Workspace
from rag.knowledge.store import RawStore
from rag.ingest.notion.client import NotionClient
from rag.ingest.notion.sync import NotionSyncPipeline

log = AppLogger.get(__name__)

# Pages expected to be present after a successful sync
EXPECTED_TITLES = [
    "Scylla consistency levels",
    "golang source code tracking - Memory allocation",
]

_PREVIEW_LEN = 300


def _check(condition: bool, msg: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    if not condition:
        pytest.fail("Integration test failed")


def main() -> None:
    config = AppConfig()

    AppLogger.setup(level=config.log_level)

    token = config.notion_token
    database_id = config.notion_database_id
    workspace_name = config.notion_workspace_id or "test_workspace"
    raw_db_path = config.raw_db_path

    if not token:
        print("ERROR: notion_token is not set in etc/config.yaml")
        pytest.fail("Integration test failed")
    if not database_id:
        print("ERROR: notion_database_id is not set in etc/config.yaml")
        pytest.fail("Integration test failed")

    print(f"notion_token     : {token[:12]}...")
    print(f"notion_database_id: {database_id}")
    print(f"workspace_name   : {workspace_name}")
    print(f"raw_db_path      : {raw_db_path}")

    # ------------------------------------------------------------------
    # Step 1: resolve data_source_id
    # ------------------------------------------------------------------
    print("\n--- Step 1: resolve data_source_id ---")
    client = NotionClient(token)

    # Use data_source_id directly from config when available; otherwise
    # derive it from the database object (requires a valid database UUID).
    data_source_id = config.notion_data_source_id
    if data_source_id:
        print(f"  data_source_id  : {data_source_id}  (from config)")
    else:
        if not database_id:
            print("ERROR: neither notion_data_source_id nor notion_database_id is set")
            pytest.fail("Integration test failed")
        db = client.get_database(database_id)
        data_sources = db.get("data_sources", [])
        if not data_sources:
            print("FAIL: database has no data_sources — cannot query pages via data source")
            pytest.fail("Integration test failed")
        data_source_id = data_sources[0]["id"]
        data_source_name = data_sources[0].get("name", "?")
        print(f"  data_source_id  : {data_source_id}  (resolved from database)")
        print(f"  data_source_name: {data_source_name}")

    # ------------------------------------------------------------------
    # Step 2: run full sync into a fresh RawStore
    # ------------------------------------------------------------------
    print("\n--- Step 2: run NotionSyncPipeline (full_sync=True) ---")
    store = RawStore(raw_db_path)
    workspace = Workspace.new(workspace_name)

    pipeline = NotionSyncPipeline(
        token=token,
        workspace=workspace,
        store=store,
        data_source_id=data_source_id,
        full_sync=True,
    )
    result = pipeline.sync()
    print(f"  pages_seen    : {result.pages_seen}")
    print(f"  pages_updated : {result.pages_updated}")
    print(f"  pages_skipped : {result.pages_skipped}")
    print(f"  errors        : {result.errors}")
    if result.error_ids:
        print(f"  error_ids     : {result.error_ids}")

    _check(result.errors == 0, "sync completed with no errors")
    _check(result.pages_seen > 0, f"at least one page synced (got {result.pages_seen})")

    # ------------------------------------------------------------------
    # Step 3: verify expected pages are stored with content
    # ------------------------------------------------------------------
    print("\n--- Step 3: verify stored pages ---")
    pages = store.list_pages(workspace_id=workspace.id)
    stored_titles = {p.title for p in pages}

    print(f"  total pages in store: {len(pages)}")
    for page in pages:
        blocks = store.get_blocks(page.id)
        content_len = sum(len(b.content) for b in blocks)
        print(f"  - {page.title!r}  blocks={len(blocks)}  content_chars={content_len}")

    for expected in EXPECTED_TITLES:
        _check(expected in stored_titles, f"page stored: {expected!r}")

    # ------------------------------------------------------------------
    # Step 4: print markdown preview for each expected page
    # ------------------------------------------------------------------
    print("\n--- Step 4: markdown content preview ---")
    for page in pages:
        if page.title not in EXPECTED_TITLES:
            continue
        blocks = store.get_blocks(page.id)
        markdown = blocks[0].content if blocks else ""
        preview = markdown[:_PREVIEW_LEN].replace("\n", " ")
        print(f"\n  Page : {page.title!r}")
        print(f"  ID   : {page.notion_page_id}")
        print(f"  Tags : {page.tags}")
        print(f"  Chars: {len(markdown)}")
        print(f"  Preview: {preview}...")
        _check(len(markdown) > 0, f"markdown content is non-empty for {page.title!r}")

    print("\nAll checks passed.")


@pytest.mark.integration
def test_notion_sync():
    main()


if __name__ == "__main__":
    main()
