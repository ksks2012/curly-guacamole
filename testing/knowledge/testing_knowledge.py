"""
Smoke test for Step 0.2 + 0.3:
  - ChunkMetadata serialisation round-trip
  - RawStore CRUD (workspace, page, blocks, version, cursor)
  - NotionClient.raw_page_to_model / raw_blocks_to_models
"""

import os
import tempfile
import uuid
from datetime import datetime, timezone

from rag.knowledge.metadata import ChunkMetadata, _join, _split
from rag.knowledge.models import (
    Block, BlockType, ChangeType, DocumentVersion, Page, Workspace,
)
from rag.knowledge.store import RawStore
from rag.ingest.notion.client import NotionClient

# ── helpers ─────────────────────────────────────────────────────────────────

def iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Step 0.2: ChunkMetadata ──────────────────────────────────────────────────

def test_metadata():
    ws = Workspace.new("Test WS")
    pg = Page.new(ws.id, "My Page", "/tmp/x.md",
                  document_type="markdown", tags=["ai", "rag"])

    meta = ChunkMetadata.from_page(
        pg, ws,
        chunk_id=3,
        section="Intro",
        heading_path=["Overview", "Intro"],
        block_id="b-001",
        block_type="paragraph",
        embedding_version="text-embedding-ada-002",
        chunk_version="heading-v1",
        content_type="prose",
    )

    chroma = meta.to_chroma()

    # All values must be Chroma-safe scalars
    for k, v in chroma.items():
        assert isinstance(v, (str, int, float, bool)), (
            f"Non-scalar value for key {k!r}: {type(v)}"
        )

    # Legacy aliases present
    assert chroma["doc_id"] == pg.id
    assert chroma["source_id"] == pg.id
    assert chroma["title"] == pg.title
    assert chroma["workspace"] == ws.name

    # Lists serialised as comma-joined strings
    assert chroma["tags"] == "ai,rag"
    assert chroma["heading_path"] == "Overview,Intro"

    # New v1 provenance fields
    assert chroma["source_type"] == "local"
    assert chroma["content_type"] == "prose"
    assert chroma["embedding_version"] == "text-embedding-ada-002"
    assert chroma["chunk_version"] == "heading-v1"

    # Aliases present
    assert chroma["created_at"] == chroma["created_time"]
    assert chroma["updated_at"] == chroma["last_edited_time"]
    assert chroma["source_path"] == chroma["source_url"]

    # Round-trip
    restored = ChunkMetadata.from_chroma(chroma)
    assert restored.tags == ["ai", "rag"]
    assert restored.heading_path == ["Overview", "Intro"]
    assert restored.chunk_id == 3
    assert restored.section == "Intro"
    assert restored.source_type == "local"
    assert restored.content_type == "prose"
    assert restored.embedding_version == "text-embedding-ada-002"
    assert restored.chunk_version == "heading-v1"

    print("  ChunkMetadata: OK")


# ── Step 0.3: RawStore ───────────────────────────────────────────────────────

def test_raw_store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = RawStore(db_path)

        # Workspace
        ws = Workspace.new("WS-A")
        store.upsert_workspace(ws)
        ws2 = store.get_workspace(ws.id)
        assert ws2 is not None and ws2.name == "WS-A"

        # Page
        pg = Page.new(ws.id, "Page Alpha", "https://notion.so/abc",
                      document_type="notion", tags=["x"])
        pg.notion_page_id = "notion-abc"
        store.upsert_page(pg, content_hash="aaa")
        pg2 = store.get_page(pg.id)
        assert pg2 is not None and pg2.title == "Page Alpha"
        assert store.get_content_hash(pg.id) == "aaa"

        # get_page_by_notion_id
        pg3 = store.get_page_by_notion_id("notion-abc")
        assert pg3 is not None and pg3.id == pg.id

        # Blocks
        b1 = Block.new(pg.id, BlockType.HEADING_1, "Hello World", 0)
        b2 = Block.new(pg.id, BlockType.PARAGRAPH, "Some text.", 1)
        store.upsert_blocks([b1, b2])
        blocks = store.get_blocks(pg.id)
        assert len(blocks) == 2
        assert blocks[0].block_type == BlockType.HEADING_1

        # delete_blocks
        store.delete_blocks(pg.id)
        assert store.get_blocks(pg.id) == []

        # DocumentVersion
        ver = DocumentVersion.new(pg.id, 1, "hash-001", ChangeType.CREATED)
        store.record_version(ver)
        latest = store.latest_version(pg.id)
        assert latest is not None and latest.version == 1
        assert store.next_version_number(pg.id) == 2

        # Sync cursor
        store.set_cursor("notion_sync:ws", "cursor-xyz")
        assert store.get_cursor("notion_sync:ws") == "cursor-xyz"
        store.set_cursor("notion_sync:ws", None)
        assert store.get_cursor("notion_sync:ws") is None

        # Stats
        stats = store.stats()
        assert stats["workspaces"] == 1
        assert stats["pages"] == 1

        print("  RawStore: OK")
    finally:
        os.unlink(db_path)


# ── Step 0.3: NotionClient model conversion ──────────────────────────────────

def test_notion_conversion():
    ws = Workspace.new("Notion WS")

    fake_page = {
        "id": "11111111-1111-1111-1111-111111111111",
        "object": "page",
        "url": "https://notion.so/test-page",
        "created_time": "2024-01-01T00:00:00.000Z",
        "last_edited_time": "2024-06-01T00:00:00.000Z",
        "properties": {
            "Name": {
                "type": "title",
                "title": [{"plain_text": "My Notion Page"}],
            },
            "Tags": {
                "type": "multi_select",
                "multi_select": [
                    {"name": "ai"},
                    {"name": "llm"},
                ],
            },
        },
    }

    page = NotionClient.raw_page_to_model(fake_page, ws.id, ws.name)
    assert page.title == "My Notion Page"
    assert page.document_type == "notion"
    assert page.notion_page_id == "11111111-1111-1111-1111-111111111111"
    assert "ai" in page.tags

    fake_blocks = [
        {
            "id": "aaaa",
            "type": "heading_1",
            "heading_1": {"rich_text": [{"plain_text": "Introduction"}]},
            "has_children": False,
            "parent": {"type": "page_id", "page_id": fake_page["id"]},
            "_depth": 0,
        },
        {
            "id": "bbbb",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"plain_text": "This is a paragraph."}]},
            "has_children": False,
            "parent": {"type": "page_id", "page_id": fake_page["id"]},
            "_depth": 0,
        },
        {
            "id": "cccc",
            "type": "code",
            "code": {
                "rich_text": [{"plain_text": "print('hello')"}],
                "language": "python",
                "caption": [],
            },
            "has_children": False,
            "parent": {"type": "page_id", "page_id": fake_page["id"]},
            "_depth": 0,
        },
    ]

    blocks = NotionClient.raw_blocks_to_models(fake_blocks, page.id)
    assert len(blocks) == 3
    assert blocks[0].block_type == BlockType.HEADING_1
    assert blocks[0].content == "Introduction"
    assert blocks[2].block_type == BlockType.CODE
    assert blocks[2].metadata.get("language") == "python"

    # Verify content hash is stable
    h1 = DocumentVersion.compute_hash(blocks)
    h2 = DocumentVersion.compute_hash(list(reversed(blocks)))
    assert h1 == h2, "Hash must be order-independent"

    print("  NotionClient conversion: OK")



