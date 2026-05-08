"""
chunker.py — content-aware text splitter for the ingestion pipeline.

Supports three chunking strategies (see strategies.py for details):

  recursive     Standard RecursiveCharacterTextSplitter — good all-rounder.
  heading       Sentence-boundary splitting that keeps heading sections whole
                when they fit within chunk_size.  Best for Markdown notes.
  semantic      Embedding-based SemanticChunker from langchain-experimental.
                Requires an embeddings model; finds natural topic-shift points.

When *doc_meta* is supplied (pre-built by DocumentIngester via
schema.make_document_meta), it is merged into every chunk via
schema.finalise_chunk.  When omitted, a minimal fallback dict is derived
from existing chunk metadata for backward compatibility.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schema import finalise_chunk, make_document_meta
from .strategies import ChunkStrategy, auto_strategy, resolve_strategy

# ---------------------------------------------------------------------------
# Strategy implementations (private)
# ---------------------------------------------------------------------------

# Separators tried in order for RecursiveCharacterTextSplitter.
_RECURSIVE_SEPARATORS = ["\n\n", "\n", " ", ""]

# Sentence-boundary-aware separators used by the heading-aware strategy.
# Trying to cut at: paragraph → sentence → clause → word → char.
_HEADING_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _recursive_chunk(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Standard recursive character splitting across all input documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_RECURSIVE_SEPARATORS,
    )
    return splitter.split_documents(docs)


def _heading_aware_chunk(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Sentence-boundary splitting that respects parser-produced heading sections.

    Each input Document (one per heading section from the Markdown parser) is
    processed independently:
      - Section fits within chunk_size → kept as a single chunk.
      - Section exceeds chunk_size    → split at sentence boundaries first,
                                        then line / word / char as fallback.

    Processing docs individually ensures that heading metadata (heading,
    hierarchy, level) is never blurred across section boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_HEADING_SEPARATORS,
    )
    result: list[Document] = []
    for doc in docs:
        result.extend(splitter.split_documents([doc]))
    return result


def _semantic_chunk(
    docs: list[Document],
    embeddings,
    chunk_size: int,
) -> list[Document]:
    """Embedding-based topic-shift chunking via SemanticChunker.

    All input documents are joined into a single text body.  SemanticChunker
    finds semantic breakpoints by computing embedding distances between adjacent
    sentences and inserting boundaries above a percentile threshold.

    Note: heading/section metadata is not preserved (chunks may cross section
    boundaries) because semantic coherence takes priority over structure.

    Args:
        docs       : Documents from any parser.
        embeddings : LangChain-compatible embeddings model (required).
        chunk_size : Passed as ``buffer_size`` hint to the splitter.

    Raises:
        ImportError : when langchain-experimental is not installed.
        ValueError  : when *embeddings* is None.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        raise ImportError(
            "The 'semantic' strategy requires langchain-experimental. "
            "Install it with:  pip install langchain-experimental"
        )
    if embeddings is None:
        raise ValueError(
            "The 'semantic' strategy requires an embeddings model. "
            "Ensure DocumentIngester is constructed with embeddings=<model>."
        )

    # Merge all section content, keeping paragraph spacing.
    full_text = "\n\n".join(d.page_content for d in docs if d.page_content.strip())

    # Use the base metadata from the first input doc for all resulting chunks.
    base_meta: dict = docs[0].metadata.copy() if docs else {}

    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
    )
    raw = chunker.create_documents([full_text])

    # Propagate source/filename metadata to semantic chunks.
    for chunk in raw:
        chunk.metadata = {**base_meta, **chunk.metadata}

    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    doc_id: str | None = None,
    doc_meta: dict | None = None,
    strategy: str | ChunkStrategy | None = None,
    embeddings=None,
) -> list[Document]:
    """Split *docs* into chunks and stamp each with the canonical metadata schema.

    Args:
        docs         : Documents produced by a parser.
        chunk_size   : Maximum characters per chunk (or buffer hint for semantic).
        chunk_overlap: Overlap between consecutive chunks (unused for semantic).
        doc_id       : Fallback grouping key when *doc_meta* is None.
        doc_meta     : Pre-built document-level metadata from
                       schema.make_document_meta.  When provided, *doc_id* is
                       ignored.
        strategy     : Chunking strategy.  None / "auto" → inferred from
                       doc_meta["document_type"].  See ChunkStrategy enum.
        embeddings   : Embeddings model; required only for strategy="semantic".

    Returns:
        List of Documents with fully-populated schema metadata.
    """
    if not docs:
        return []

    # Resolve strategy -------------------------------------------------------
    resolved = resolve_strategy(strategy)
    if resolved is None:
        # Auto-select from document_type when we have doc_meta
        if doc_meta:
            resolved = auto_strategy(doc_meta.get("document_type", "text"))
        else:
            resolved = ChunkStrategy.RECURSIVE

    # Dispatch to strategy implementation ------------------------------------
    if resolved == ChunkStrategy.SEMANTIC:
        raw = _semantic_chunk(docs, embeddings, chunk_size)
    elif resolved == ChunkStrategy.HEADING_AWARE:
        raw = _heading_aware_chunk(docs, chunk_size, chunk_overlap)
    else:
        raw = _recursive_chunk(docs, chunk_size, chunk_overlap)

    # Build doc_meta fallback (backward compat when called without doc_meta) -
    if doc_meta is None:
        first_meta = raw[0].metadata if raw else {}
        source = first_meta.get("source", "")
        fallback_doc_id = (
            (doc_id or "").strip()
            or first_meta.get("doc_id", "")
            or (source.rsplit("/", 1)[-1] if source else "unknown")
        )
        doc_meta = make_document_meta(source or "unknown", doc_id=fallback_doc_id)

    # Stamp each chunk with the canonical schema -----------------------------
    for idx, chunk in enumerate(raw):
        finalise_chunk(chunk, idx, doc_meta)

    return raw
