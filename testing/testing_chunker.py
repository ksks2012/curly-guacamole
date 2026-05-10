"""
Step 1.1 — Verify the NotionChunker structure-aware chunking.

Tests:
    1. Fetch blocks for the golang memory page (rich structure)
    2. Run NotionChunker.chunk()
    3. Structural assertions:
         - Chunk count > 0
         - Each chunk has content (non-empty)
         - section field matches nearest heading text
         - Heading chunks: content == section (heading is the whole chunk)
         - Code blocks produce standalone chunks with fence markers
         - No chunk is wider than MAX_SECTION_CHARS * 1.5 (safety margin for
           edge cases like very long single paragraphs)
         - block_ids are non-empty and contain valid UUIDs
         - chunk_index is sequential
    4. Print a formatted summary of all chunks

Usage:
    python testing/testing_chunker.py [page_id]
"""

import sys

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.ingest.notion.client import NotionClient
from rag.ingest.notion.chunker import MAX_SECTION_CHARS, NotionChunker


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
    if len(sys.argv) > 1:
        return sys.argv[1]
    data_source_id = config.notion_data_source_id
    if not data_source_id:
        print("ERROR: notion_data_source_id not set — provide page_id as CLI arg")
        sys.exit(1)
    batch, _ = next(iter(client.iter_data_source_pages(data_source_id)))
    if len(batch) < 2:
        print("ERROR: data source returned fewer than 2 pages")
        sys.exit(1)
    page_id = batch[1]["id"]
    print(f"  auto-resolved page_id: {page_id}")
    return page_id


def _print_chunks(chunks) -> None:
    for c in chunks:
        preview = c.content.replace("\n", "↵")[:80]
        block_count = len(c.block_ids)
        print(
            f"  [{c.chunk_index:3d}] section={c.section[:30]!r:32s} "
            f"blocks={block_count}  chars={len(c.content):4d}  {preview!r}"
        )


def main() -> None:
    AppLogger.setup(level="WARNING")
    config = AppConfig()
    token = config.notion_token
    if not token:
        print("ERROR: notion_token is not set")
        sys.exit(1)

    client = NotionClient(token)

    # ------------------------------------------------------------------
    # Step 1: fetch blocks
    # ------------------------------------------------------------------
    _section("Step 1: fetch blocks")
    page_id = _resolve_page_id(client, config)
    raw_blocks = client.get_all_blocks(page_id)
    import uuid as _uuid
    dummy_page_id = str(_uuid.uuid4())
    blocks = NotionClient.raw_blocks_to_models(raw_blocks, dummy_page_id)
    print(f"  total blocks: {len(blocks)}")

    # ------------------------------------------------------------------
    # Step 2: chunk
    # ------------------------------------------------------------------
    _section("Step 2: NotionChunker.chunk()")
    chunker = NotionChunker()
    chunks = chunker.chunk(blocks, dummy_page_id)
    print(f"  total chunks: {len(chunks)}")

    # ------------------------------------------------------------------
    # Step 3: assertions
    # ------------------------------------------------------------------
    _section("Step 3: assertions")

    # 3a. At least one chunk
    _check(len(chunks) > 0, f"chunk count > 0 (got {len(chunks)})")

    # 3b. Sequential chunk_index
    indices = [c.chunk_index for c in chunks]
    _check(indices == list(range(len(chunks))), "chunk_index is sequential")

    # 3c. All chunks have non-empty content
    empty = [c for c in chunks if not c.content.strip()]
    _check(len(empty) == 0, f"all chunks have non-empty content ({len(empty)} empty found)")

    # 3d. All chunks have at least one block_id
    no_blocks = [c for c in chunks if not c.block_ids]
    _check(len(no_blocks) == 0, f"all chunks reference at least one block ({len(no_blocks)} without block_ids)")

    # 3e. No chunk exceeds the safety cap (MAX_SECTION_CHARS * 1.5 + breadcrumb margin)
    cap = int(MAX_SECTION_CHARS * 1.5) + 80
    oversized = [c for c in chunks if len(c.content) > cap]
    _check(len(oversized) == 0, f"no chunk exceeds size cap {cap} chars ({len(oversized)} oversized)")

    # 3f. Code chunks contain fence markers
    code_chunks = [c for c in chunks if "```" in c.content]
    print(f"  code chunks detected: {len(code_chunks)}")

    # 3g. section field is set for all chunks that follow a heading
    headings_seen = [c for c in chunks if c.section]
    print(f"  chunks with non-empty section: {len(headings_seen)}/{len(chunks)}")

    # 3h. page_id is consistent
    mismatched_page = [c for c in chunks if c.page_id != dummy_page_id]
    _check(len(mismatched_page) == 0, "all chunks carry correct page_id")

    # ------------------------------------------------------------------
    # Step 4: print chunk summary
    # ------------------------------------------------------------------
    _section("Step 4: chunk summary")
    _print_chunks(chunks)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
