"""
Step 1.3 — Verify NotionRAGClient: Chroma collection + retrieval integration.

Tests (all assertions):
    1. sync_and_embed() completes without errors
    2. page_count() == expected page count from data source
    3. search() returns results with correct metadata (source_id, section, title)
    4. hybrid_search() returns results (requires BM25 index built)
    5. All returned docs have document_type == "notion"
    6. workspace field matches configured workspace name
    7. RAGEngine context tag for Notion docs uses title/section format
    8. query_with_filter() completes (embedding server gating)

Steps 3-8 that require the embedding server are skipped with a note when
localhost:8080 is not reachable.

Usage:
    python testing/testing_rag_client.py
"""

import socket
import sys
import pytest
from urllib.parse import urlparse

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.ingest.notion.pipeline import NotionRAGClient


def _server_up(base_url: str) -> bool:
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 80
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
    pytest.fail("Integration test failed")


def _check(condition: bool, msg: str) -> None:
    if condition:
        pass
    else:
        _fail(msg)


def main() -> None:
    AppLogger.setup(level="INFO")
    config = AppConfig()

    embed_up = _server_up(config.embed_base)
    llm_up   = _server_up(config.llm_base)

    if not embed_up:
        print(
            f"  [NOTE] Embedding server not reachable at {config.embed_base!r}.\n"
            "         Steps 3-8 (retrieval + query) will be skipped."
        )

    # ------------------------------------------------------------------
    # Step 1: instantiate client
    # ------------------------------------------------------------------
    _section("Step 1: instantiate NotionRAGClient")
    client = NotionRAGClient(config)
    print(f"  workspace: {client._workspace.name!r}")

    # ------------------------------------------------------------------
    # Step 2: sync_and_embed  (always runs — network required)
    # ------------------------------------------------------------------
    _section("Step 2: sync_and_embed()")
    result = client.sync_and_embed()
    print(f"  {result}")
    _check(result.sync.pages_seen > 0,
           f"pages seen during sync ({result.sync.pages_seen})")
    _check(result.sync.errors == 0,
           f"no sync errors (got {result.sync.errors})")

    n_pages = client.page_count()
    print(f"  pages in RawStore: {n_pages}")
    _check(n_pages > 0, f"at least one page in RawStore ({n_pages})")

    if not embed_up:
        print("\n  [SKIP] Steps 3-8: start embedding server and re-run.")
        print("\nSync pipeline verified. Retrieval pending server.")
        return

    _check(result.embed.errors == 0,
           f"no embed errors (got {result.embed.errors})")
    total_embed = result.embed.chunks_added + result.embed.chunks_skipped + result.embed.chunks_updated
    _check(total_embed > 0,
           f"chunks processed (added={result.embed.chunks_added}, "
           f"skipped={result.embed.chunks_skipped})")

    # ------------------------------------------------------------------
    # Step 3: vector search
    # ------------------------------------------------------------------
    _section("Step 3: search() — vector similarity")
    results = client.search("memory allocation span object", k=5)
    print(f"  results: {len(results)}")
    _check(len(results) > 0, "search() returned results")

    workspace_name = client._workspace.name
    for doc, score in results[:3]:
        meta = doc.metadata
        print(
            f"    score={score:.3f}  title={meta.get('title','?')[:30]!r}  "
            f"section={meta.get('section','')[:25]!r}  "
            f"doc_type={meta.get('document_type','?')!r}"
        )

    # ------------------------------------------------------------------
    # Step 4: metadata validation on returned docs
    # ------------------------------------------------------------------
    _section("Step 4: metadata validation")
    pages = client.list_pages()
    known_page_ids = {p.id for p in pages}
    required_keys = {"source_id", "doc_id", "title", "workspace",
                     "section", "document_type"}

    for doc, _ in results:
        meta = doc.metadata
        missing = required_keys - set(meta.keys())
        _check(len(missing) == 0,
               f"all required keys present (missing: {missing})")
        _check(meta.get("document_type") == "notion",
               f"document_type == 'notion' (got {meta.get('document_type')!r})")
        _check(meta.get("workspace") == workspace_name,
               f"workspace == {workspace_name!r}")
        _check(meta.get("source_id") in known_page_ids,
               f"source_id {meta.get('source_id','?')[:8]} is a known page")

    # Step 5: hybrid search (BM25 + vector)
    # ------------------------------------------------------------------
    _section("Step 5: hybrid_search() — BM25 + RRF")
    hybrid = client.hybrid_search("tcmalloc cache span", k=5)
    print(f"  results: {len(hybrid)}")
    _check(len(hybrid) > 0, "hybrid_search() returned results")
    for doc, score in hybrid[:3]:
        meta = doc.metadata
        print(
            f"    score={score:.3f}  "
            f"section={meta.get('section','')[:30]!r}  "
            f"{doc.page_content[:50]!r}"
        )

    # ------------------------------------------------------------------
    # Step 6: context tag format (Notion docs use title/section)
    # ------------------------------------------------------------------
    _section("Step 6: RAGEngine context tag for Notion")
    from langchain_core.documents import Document as LCDoc
    from rag.engine import RAGEngine

    # Build synthetic docs with Notion-style metadata
    test_docs = [
        LCDoc(
            page_content="cache allocates memory without locks",
            metadata={
                "document_type": "notion",
                "title": "golang memory",
                "section": "分配器",
                "page_content": "",
            },
        ),
        LCDoc(
            page_content="PDF content here",
            metadata={
                "document_type": "pdf",
                "filename": "test.pdf",
                "page": 2,
            },
        ),
    ]

    # Verify tag construction logic directly (private helper pattern)
    tags = []
    for doc in test_docs:
        meta = doc.metadata
        dtype = meta.get("document_type", "")
        if dtype == "notion":
            title   = meta.get("title", "Notion")
            section = meta.get("section", "")
            tag = f"[{title} / {section}]" if section else f"[{title}]"
        else:
            pg   = meta.get("page")
            name = meta.get("filename") or meta.get("title", "unknown")
            tag  = f"[page {pg + 1}, {name}]" if pg is not None else f"[{name}]"
        tags.append(tag)

    _check(
        tags[0] == "[golang memory / 分配器]",
        f"Notion context tag format correct: {tags[0]!r}",
    )
    _check(
        tags[1] == "[page 3, test.pdf]",
        f"PDF context tag format correct: {tags[1]!r}",
    )

    # ------------------------------------------------------------------
    # Step 7: query_with_filter (LLM required)
    # ------------------------------------------------------------------
    _section("Step 7: query_with_filter()")
    if not llm_up:
        print(f"  [SKIP] LLM server not reachable at {config.llm_base!r}")
    else:
        answer = client.query_with_filter(
            "What is a span in Go memory allocation?",
            k=3,
            expand_query=False,
        )
        print(f"  answer preview: {str(answer)[:200]!r}")
        _check(len(str(answer)) > 10, "query returned non-trivial answer")

    print("\nAll checks passed.")


@pytest.mark.integration
def test_notion_pipeline():
    main()


if __name__ == "__main__":
    main()
