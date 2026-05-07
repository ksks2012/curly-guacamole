"""
document_ingester.py — unified entry point for the ingestion pipeline.

Usage::

    ingester = DocumentIngester()
    chunks = ingester.ingest(
        path="/path/to/file.pdf",   # or .md / .txt
        doc_id="my-doc",
        chunk_size=500,
        chunk_overlap=100,
    )

Supported extensions: .pdf  .md  .markdown  .txt  .text

Pipeline::

    file path
        │
        ▼
    Parser  (pdf / markdown / text)
        │  produces Documents with source-specific metadata
        │  (page, heading, hierarchy, …)
        ▼
    Chunker  (RecursiveCharacterTextSplitter)
        │  injects standard fields: source_id, doc_id, chunk_id
        ▼
    list[Document]  → ready for Indexer.run()
"""

import os
from langchain_core.documents import Document

from .chunker import chunk_documents
from .parsers import parse_pdf, parse_markdown, parse_text
from .schema import make_document_meta
from utils.logger import AppLogger

log = AppLogger.get(__name__)

# Map of lowercase extension → parser function
_PARSERS = {
    ".pdf":      parse_pdf,
    ".md":       parse_markdown,
    ".markdown": parse_markdown,
    ".txt":      parse_text,
    ".text":     parse_text,
}

SUPPORTED_EXTENSIONS = tuple(_PARSERS.keys())


class DocumentIngester:
    """Dispatches a file to the appropriate parser and runs the chunker.

    This class is stateless; create one instance per application or per call.
    """

    def ingest(
        self,
        path: str,
        doc_id: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        title: str = "",
        tags: list[str] | None = None,
        workspace: str = "",
        importance: float = 0.0,
    ) -> list[Document]:
        """Parse *path* and return enriched, chunked Documents.

        Args:
            path         : Absolute or relative path to the document.
            doc_id       : Grouping / filter key stored in chunk metadata.
                           Defaults to the filename stem when not provided.
            chunk_size   : Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
            title        : Human-readable document title (defaults to doc_id).
            tags         : Tag strings attached to every chunk.
            workspace    : Logical workspace or project label.
            importance   : Float priority stamp (0.0 = neutral).

        Returns:
            List of Documents ready to be passed to ``Indexer.run()``.

        Raises:
            ValueError        : if the file extension is not supported.
            FileNotFoundError : if the file does not exist.
        """
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")

        ext = os.path.splitext(abs_path)[1].lower()
        parser = _PARSERS.get(ext)
        if parser is None:
            raise ValueError(
                f"Unsupported file type: {ext!r}. "
                f"Supported: {', '.join(sorted(_PARSERS))}"
            )

        log.info(
            "DocumentIngester.ingest: %s  ext=%s  doc_id=%r  chunk_size=%d  overlap=%d",
            os.path.basename(abs_path), ext, doc_id, chunk_size, chunk_overlap,
        )

        docs = parser(abs_path)
        log.debug("  parser produced %d document(s)", len(docs))

        doc_meta = make_document_meta(
            abs_path,
            doc_id=doc_id or "",
            title=title,
            tags=tags,
            workspace=workspace,
            importance=importance,
        )

        chunks = chunk_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            doc_meta=doc_meta,
        )
        log.info("  chunked → %d chunks", len(chunks))
        return chunks
