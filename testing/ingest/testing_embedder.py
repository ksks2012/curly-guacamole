"""
Step 1.2 — Verify the Notion Embedding Pipeline end-to-end.

Flow tested:
    NotionSyncPipeline.sync()   (ensure pages are in RawStore)
        → NotionEmbedder.embed_workspace()
            → get_all_blocks()  [live API]
            → NotionChunker.chunk()
            → Chunk.to_document()
            → Indexer.run()     [Chroma + RecordManager]
        → Chroma similarity_search() to confirm vectors are queryable

Steps 1-3 verify the pipeline up to the Indexer.
Steps 4-6 (Chroma query) require the local embedding server to be running.
When the embedding server is unavailable, Steps 4-6 are skipped with a note.

Usage:
    python testing/testing_embedder.py
"""

import socket
import sys
from urllib.parse import urlparse

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.knowledge.models import Workspace
from rag.knowledge.store import RawStore
from rag.ingest.notion.client import NotionClient
from rag.ingest.notion.sync import NotionSyncPipeline
from rag.ingest.notion.embedder import NotionEmbedder


def _embed_server_available(base_url: str) -> bool:
    """Return True if the embedding server TCP port is reachable."""
    try:
        parsed = urlparse(base_url)
        host   = parsed.hostname or "localhost"
        port   = parsed.port or 80
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


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


def main() -> None:
    AppLogger.setup(level="INFO")
    config = AppConfig()

    token           = config.notion_token
    workspace_name  = config.notion_workspace_id or "RAG Ingest Space"
    data_source_id  = config.notion_data_source_id

    if not token:
        print("ERROR: notion_token not set")
        sys.exit(1)

    embed_available = _embed_server_available(config.embed_base)
    if not embed_available:
        print(
            f"  [NOTE] Embedding server not reachable at {config.embed_base!r}.\n"
            "         Steps 1-3 (sync + chunk + document conversion) will run.\n"
            "         Steps 4-6 (Chroma index + query) will be skipped."
        )

    store  = RawStore(config.raw_db_path)
    client = NotionClient(token)

    # ------------------------------------------------------------------
    # Step 1: ensure pages are synced into RawStore
    # ------------------------------------------------------------------
    _section("Step 1: sync pages into RawStore")

    # Reuse existing workspace if present, otherwise create one.
    existing_workspaces = store.list_workspaces()
    workspace = next(
        (w for w in existing_workspaces if w.name == workspace_name), None
    )
    if workspace is None:
        workspace = Workspace.new(workspace_name)

    pipeline = NotionSyncPipeline(
        token=token,
        workspace=workspace,
        store=store,
        data_source_id=data_source_id,
    )
    sync_result = pipeline.sync()
    print(f"  sync result: {sync_result.as_dict()}")
    _check(sync_result.pages_seen > 0, f"at least one page synced ({sync_result.pages_seen} seen)")

    pages = store.list_pages(workspace.id)
    print(f"  pages in RawStore: {len(pages)}")
    for p in pages:
        print(f"    {p.title!r}  notion_id={p.notion_page_id}")

    # ------------------------------------------------------------------
    # Step 2: verify chunk + document conversion (pre-embedding)
    # ------------------------------------------------------------------
    _section("Step 2: verify chunk + document conversion")

    from rag.ingest.notion.chunker import NotionChunker
    chunker = NotionChunker()

    all_docs = []
    for page in pages:
        if not page.notion_page_id:
            continue
        workspace = store.get_workspace(page.workspace_id)
        raw_blocks = client.get_all_blocks(page.notion_page_id)
        blocks     = NotionClient.raw_blocks_to_models(raw_blocks, page.id)
        chunks     = chunker.chunk(blocks, page.id)
        docs       = [c.to_document(page, workspace) for c in chunks]
        all_docs.extend(docs)
        print(
            f"  {page.title!r}: {len(blocks)} blocks → {len(chunks)} chunks → {len(docs)} docs"
        )

    _check(len(all_docs) > 0, f"documents produced from all pages ({len(all_docs)} total)")

    # Verify Document metadata schema
    required_keys = {"source_id", "doc_id", "title", "workspace", "section",
                     "chunk_id", "document_type", "tags"}
    for doc in all_docs[:3]:
        missing = required_keys - set(doc.metadata.keys())
        _check(len(missing) == 0,
               f"required metadata keys present (missing: {missing})")

    # source_id is page.id for all docs
    known_page_ids = {p.id for p in pages}
    bad_source = [d for d in all_docs if d.metadata.get("source_id") not in known_page_ids]
    _check(len(bad_source) == 0,
           f"all docs have valid source_id ({len(bad_source)} invalid)")

    # ------------------------------------------------------------------
    # Steps 3-5: Chroma embed + query (requires embedding server)
    # ------------------------------------------------------------------
    if not embed_available:
        print("\n  [SKIP] Steps 3-5: embedding server not running.")
        print(f"         Start it at {config.embed_base!r} and re-run to test Chroma indexing.")
        print("\nPipeline architecture verified (chunks + docs). Embedding pending server.")
        return

    # ------------------------------------------------------------------
    # Step 3: first embed run — expect chunks to be added
    # ------------------------------------------------------------------
    _section("Step 3: first embed run")

    embedder = NotionEmbedder(config, store, client)
    result1  = embedder.embed_workspace(workspace.id)
    print(f"  {result1}")

    _check(result1.pages_embedded == len(pages),
           f"all pages embedded ({result1.pages_embedded}/{len(pages)})")
    _check(result1.errors == 0,
           f"no errors during embedding (got {result1.errors})")
    total_chunks = result1.chunks_added + result1.chunks_updated + result1.chunks_skipped
    _check(total_chunks > 0,
           f"at least one chunk processed (added={result1.chunks_added}, "
           f"updated={result1.chunks_updated}, skipped={result1.chunks_skipped})")

    # ------------------------------------------------------------------
    # Step 4: second embed run — incremental dedup (all chunks skipped)
    # ------------------------------------------------------------------
    _section("Step 4: second embed run (incremental dedup)")

    result2 = embedder.embed_workspace(workspace.id)
    print(f"  {result2}")

    _check(result2.chunks_added == 0,
           f"no new chunks added on second run (got {result2.chunks_added})")
    _check(result2.chunks_skipped > 0,
           f"chunks were skipped (dedup working) — skipped={result2.chunks_skipped}")

    # ------------------------------------------------------------------
    # Step 5: Chroma similarity search
    # ------------------------------------------------------------------
    _section("Step 5: Chroma similarity search")

    # Access Chroma directly from the embedder's internal store.
    chroma  = embedder._chroma
    results = chroma.similarity_search("memory allocation", k=5)
    print(f"  results returned: {len(results)}")
    _check(len(results) > 0, "similarity_search returned results")

    print("\n  Top results:")
    for doc in results:
        meta    = doc.metadata
        preview = doc.page_content.replace("\n", "↵")[:70]
        print(
            f"    source_id={meta.get('source_id', '?')[:8]}  "
            f"section={meta.get('section', '')[:30]!r}  "
            f"content={preview!r}"
        )

    # Check metadata fields exist in returned documents
    required_keys = {"source_id", "doc_id", "title", "workspace", "section"}
    for doc in results:
        missing = required_keys - set(doc.metadata.keys())
        _check(
            len(missing) == 0,
            f"all required metadata keys present (missing: {missing})"
        )

    # Check all results belong to known pages
    known_page_ids = {p.id for p in pages}
    for doc in results:
        sid = doc.metadata.get("source_id", "")
        _check(sid in known_page_ids,
               f"source_id {sid[:8]} belongs to a known page")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
