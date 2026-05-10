"""
Step 0.3.6 — Verify Notion block reading and data structure.

Tests:
    1. get_all_blocks() — recursive depth-first block fetch
    2. raw_blocks_to_models() — Notion raw dicts → Block domain models
    3. Block-level assertions:
         - All expected block types are correctly mapped
         - table_of_contents: content is empty (no rich_text)
         - has_children blocks: child blocks are fetched (depth > 0 present)
         - Nested blocks: parent_block_id is set (not None)
         - plain_text extraction handles mention-type rich_text items
         - Block order is strictly sequential (0, 1, 2, …)
    4. Print a structured tree of all fetched blocks for visual inspection

Usage:
    python testing/testing_blocks.py [page_id]

    page_id defaults to the first page returned by the configured data source.
"""

import sys

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.knowledge.models import BlockType
from rag.ingest.notion.client import NotionClient

_TREE_INDENT = "  "

# Block types observed in the test data — assert all are mapped (not UNSUPPORTED).
_EXPECTED_TYPES = {
    "paragraph",
    "heading_1",
    "heading_2",
    "heading_3",
    "table_of_contents",
    "numbered_list_item",
    "bulleted_list_item",
}


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _check(condition: bool, msg: str) -> None:
    if condition:
        _ok(msg)
    else:
        _fail(msg)


def _resolve_page_id(client: NotionClient, config: AppConfig) -> str:
    """Return page_id from CLI arg, or auto-resolve from the data source."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    data_source_id = config.notion_data_source_id
    if not data_source_id:
        print("ERROR: notion_data_source_id not set — provide page_id as CLI arg")
        sys.exit(1)
    batch, _ = next(iter(client.iter_data_source_pages(data_source_id)))
    if not batch:
        print("ERROR: data source returned no pages")
        sys.exit(1)
    page_id = batch[1]["id"]   # use second page (golang) — richer nested content
    print(f"  auto-resolved page_id: {page_id}  ({batch[1].get('id')})")
    return page_id


def _print_block_tree(raw_blocks: list[dict]) -> None:
    """Print a compact depth-indented tree of the fetched blocks."""
    for blk in raw_blocks:
        depth  = blk.get("_depth", 0)
        btype  = blk.get("type", "?")
        bid    = blk.get("id", "?")
        has_ch = "▼" if blk.get("has_children") else " "
        # Extract a short content preview from plain_text
        type_obj  = blk.get(btype, {})
        rich      = type_obj.get("rich_text", []) if isinstance(type_obj, dict) else []
        preview   = "".join(i.get("plain_text", "") for i in rich)[:60]
        prefix    = _TREE_INDENT * depth
        print(f"  {prefix}{has_ch}[{btype:25s}] {bid[:8]}  {preview!r}")


def main() -> None:
    config = AppConfig()
    AppLogger.setup(level="WARNING")

    token = config.notion_token
    if not token:
        print("ERROR: notion_token is not set in etc/config.yaml")
        sys.exit(1)

    client = NotionClient(token)

    # ------------------------------------------------------------------
    # Step 1: resolve page_id and fetch all blocks recursively
    # ------------------------------------------------------------------
    _section("Step 1: resolve page_id and fetch blocks recursively")
    page_id = _resolve_page_id(client, config)
    print(f"  page_id: {page_id}")

    raw_blocks = client.get_all_blocks(page_id)
    print(f"  total blocks fetched (recursive): {len(raw_blocks)}")
    _check(len(raw_blocks) > 0, f"at least one block fetched (got {len(raw_blocks)})")

    # ------------------------------------------------------------------
    # Step 2: convert to Block domain models
    # ------------------------------------------------------------------
    _section("Step 2: raw_blocks_to_models()")
    # Use a dummy page_id UUID for model conversion (not stored)
    import uuid
    dummy_page_id = str(uuid.uuid4())
    blocks = NotionClient.raw_blocks_to_models(raw_blocks, dummy_page_id)
    print(f"  block models created: {len(blocks)}")
    _check(len(blocks) == len(raw_blocks), "model count matches raw block count")

    # ------------------------------------------------------------------
    # Step 3: structural assertions
    # ------------------------------------------------------------------
    _section("Step 3: structural assertions")

    # 3a. Block order is sequential 0..N-1
    orders = [b.order for b in blocks]
    _check(orders == list(range(len(blocks))), "block order is sequential")

    # 3b. All block types are resolved (no UNSUPPORTED for expected types)
    raw_types_seen = {blk.get("type") for blk in raw_blocks}
    print(f"  block types seen: {sorted(raw_types_seen)}")
    unsupported_expected = [
        t for t in (_EXPECTED_TYPES & raw_types_seen)
        if NotionClient.raw_blocks_to_models(
            [b for b in raw_blocks if b.get("type") == t][:1], dummy_page_id
        )[0].block_type == BlockType.UNSUPPORTED
    ]
    _check(
        len(unsupported_expected) == 0,
        f"all expected block types are mapped (unsupported: {unsupported_expected})",
    )

    # 3c. table_of_contents has empty content
    toc_blocks = [b for b in blocks if b.block_type == BlockType.TABLE_OF_CONTENTS]
    if toc_blocks:
        _check(
            all(b.content == "" for b in toc_blocks),
            f"table_of_contents content is empty ({len(toc_blocks)} found)",
        )
    else:
        print("  [SKIP] no table_of_contents block in this page")

    # 3d. Recursive fetch: blocks with has_children=True in raw have child blocks
    #     fetched (i.e., blocks with _depth > 0 exist)
    has_children_raw = [b for b in raw_blocks if b.get("has_children")]
    depth_gt0 = [b for b in raw_blocks if b.get("_depth", 0) > 0]
    if has_children_raw:
        _check(
            len(depth_gt0) > 0,
            f"nested blocks fetched: {len(depth_gt0)} blocks at depth > 0"
            f" (from {len(has_children_raw)} has_children parents)",
        )
    else:
        print("  [SKIP] no has_children blocks in this page")

    # 3e. Nested blocks have parent_block_id set
    nested_models = [
        b for b, r in zip(blocks, raw_blocks) if r.get("_depth", 0) > 0
    ]
    if nested_models:
        missing_parent = [b for b in nested_models if b.parent_block_id is None]
        _check(
            len(missing_parent) == 0,
            f"all nested blocks have parent_block_id set"
            f" ({len(nested_models)} nested, {len(missing_parent)} missing)",
        )
    else:
        print("  [SKIP] no nested blocks to check parent_block_id")

    # 3f. Mention-type rich_text: plain_text is extracted (not empty)
    mention_raws = [
        b for b in raw_blocks
        if any(
            item.get("type") == "mention"
            for item in (b.get(b.get("type", ""), {}) or {}).get("rich_text", [])
        )
    ]
    if mention_raws:
        mention_models = [
            blocks[i] for i, r in enumerate(raw_blocks) if r in mention_raws
        ]
        _check(
            all(b.content != "" for b in mention_models),
            f"mention-type rich_text yields non-empty content ({len(mention_models)} block(s))",
        )
    else:
        print("  [SKIP] no mention-type rich_text in this page")

    # ------------------------------------------------------------------
    # Step 4: block tree (visual inspection)
    # ------------------------------------------------------------------
    _section("Step 4: block tree")
    _print_block_tree(raw_blocks)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
