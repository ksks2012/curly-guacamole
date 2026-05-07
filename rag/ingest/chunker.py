"""
chunker.py — content-aware text splitter for the ingestion pipeline.

Uses RecursiveCharacterTextSplitter with configurable size/overlap and stamps
every chunk with the canonical metadata schema via schema.finalise_chunk.

When *doc_meta* is supplied (pre-built by DocumentIngester via
schema.make_document_meta), it is merged into every chunk.  When omitted a
minimal fallback dict is constructed from existing chunk metadata so that
callers which bypass DocumentIngester still work correctly.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schema import finalise_chunk, make_document_meta


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    doc_id: str | None = None,
    doc_meta: dict | None = None,
) -> list[Document]:
    """Split *docs* into chunks and stamp each with the canonical metadata schema.

    Args:
        docs         : Documents produced by a parser.
        chunk_size   : Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        doc_id       : Fallback grouping key used only when *doc_meta* is None.
        doc_meta     : Pre-built document-level metadata from
                       schema.make_document_meta.  When provided, *doc_id* is
                       ignored.

    Returns:
        List of Documents with fully-populated schema metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    if doc_meta is None:
        # Backward-compat: derive a minimal doc_meta from the first chunk.
        first_meta = chunks[0].metadata if chunks else {}
        source = first_meta.get("source", "")
        fallback_doc_id = (
            (doc_id or "").strip()
            or first_meta.get("doc_id", "")
            or (source.rsplit("/", 1)[-1] if source else "unknown")
        )
        doc_meta = make_document_meta(source or "unknown", doc_id=fallback_doc_id)

    for idx, chunk in enumerate(chunks):
        finalise_chunk(chunk, idx, doc_meta)

    return chunks
