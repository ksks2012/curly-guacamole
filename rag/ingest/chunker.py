"""
chunker.py — content-aware text splitter for the ingestion pipeline.

Uses RecursiveCharacterTextSplitter with configurable size/overlap and injects
standardised metadata onto every chunk:

  source     : absolute path of the source file
  filename   : basename of the source file
  source_id  : grouping key for LangChain index() incremental cleanup (= doc_id)
  doc_id     : caller-supplied identifier used for retrieval-time filtering
  chunk_id   : sequential index across all chunks (0-based)

Preserves existing metadata set by the parser (e.g. page, heading, hierarchy).
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    doc_id: str | None = None,
) -> list[Document]:
    """Split *docs* into chunks and inject standard metadata.

    Args:
        docs         : list of Documents produced by a parser.
        chunk_size   : maximum characters per chunk.
        chunk_overlap: overlap between consecutive chunks.
        doc_id       : document-level identifier; falls back to the ``filename``
                       metadata field, then to the ``source`` field basename.

    Returns:
        List of Documents with fully-populated metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata
        source = meta.get("source", "")
        filename = meta.get("filename", "") or (source.split("/")[-1] if source else "")
        resolved_doc_id = (doc_id or "").strip() or meta.get("doc_id", "") or filename

        # Build a clean, deterministic metadata dict.
        # We keep parser-supplied keys (page, heading, hierarchy, …) but
        # overwrite the five standard fields every downstream component depends on.
        clean_meta: dict = {
            k: v
            for k, v in meta.items()
            if k not in {"source_id", "doc_id", "chunk_id"}
        }
        clean_meta.update(
            {
                "source": source,
                "filename": filename,
                "source_id": resolved_doc_id,
                "doc_id": resolved_doc_id,
                "chunk_id": idx,
            }
        )
        chunk.metadata = clean_meta

    return chunks
