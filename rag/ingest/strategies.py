"""
strategies.py — chunking strategy enum and auto-selection logic.

Three strategies are supported:

  recursive     Standard RecursiveCharacterTextSplitter.
                Splits at paragraph → line → word boundaries.
                Good all-rounder; best for dense, unstructured text.

  heading       Heading-aware, sentence-boundary splitting.
                Each section produced by the Markdown parser is kept as one
                chunk when it fits within chunk_size.  Oversized sections are
                split at sentence boundaries (". " / "! " / "? ") before
                falling back to line and word breaks.
                Best for Markdown notes where headings delimit topics.

  semantic      Embedding-based topic-shift detection via SemanticChunker.
                Groups sentences by semantic similarity and inserts a chunk
                boundary whenever the embedding distance crosses a threshold.
                Requires langchain-experimental and an embeddings model.
                Best for long-form prose where topic changes are gradual.

Auto-selection (strategy="auto") maps document_type → strategy:
  pdf      → recursive
  markdown → heading
  text     → recursive
"""

from enum import Enum


class ChunkStrategy(str, Enum):
    RECURSIVE     = "recursive"
    HEADING_AWARE = "heading"
    SEMANTIC      = "semantic"


# Default strategy per document_type (from schema.doc_type_from_path)
_DEFAULT: dict[str, ChunkStrategy] = {
    "pdf":      ChunkStrategy.RECURSIVE,
    "markdown": ChunkStrategy.HEADING_AWARE,
    "text":     ChunkStrategy.RECURSIVE,
}


def auto_strategy(document_type: str) -> ChunkStrategy:
    """Return the recommended ChunkStrategy for *document_type*."""
    return _DEFAULT.get(document_type, ChunkStrategy.RECURSIVE)


def resolve_strategy(value: str | ChunkStrategy | None) -> ChunkStrategy | None:
    """Normalise a caller-supplied strategy value.

    Returns:
        None        when value is None or "auto" (let DocumentIngester decide)
        ChunkStrategy otherwise
    Raises:
        ValueError  for unknown string values
    """
    if value is None or value == "auto":
        return None
    if isinstance(value, ChunkStrategy):
        return value
    try:
        return ChunkStrategy(value)
    except ValueError:
        valid = ", ".join(s.value for s in ChunkStrategy)
        raise ValueError(f"Unknown chunk strategy {value!r}. Valid: {valid}")
