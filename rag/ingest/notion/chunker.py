"""
Structure-aware chunker for Notion pages.

Converts a flat, ordered list of Block objects (from a single Notion page) into
Chunk objects that are ready for embedding.

Chunking rules
--------------
Each Chunk corresponds to one *section* — a heading and its body content.
The algorithm is a single-pass aggregation:

  1. A heading block (H1/H2/H3/H4) always starts a NEW chunk.
  2. Body blocks (paragraph, list items, code, quote, callout, …) are
     aggregated into the current section's content.
  3. A consecutive run of same-type list items (bulleted or numbered) is
     kept together — they form a single logical list.
  4. A code block is always kept as an independent chunk because it is
     independently understandable and embedding it as a unit gives better
     retrieval precision.
  5. If the aggregated body text of a section exceeds MAX_SECTION_CHARS, the
     body is split into sub-chunks of roughly TARGET_CHUNK_CHARS each,
     respecting paragraph / sentence boundaries.

Each produced Chunk stores:
  - content     : plain text joined from the contributing Blocks
  - section     : text of the nearest preceding heading (empty if none)
  - block_ids   : ordered list of Block UUIDs that make up the Chunk
  - chunk_index : 0-based position within the Page

Context injection
-----------------
When a section body is split or a code/list chunk has no heading of its own,
the nearest heading is prepended as a breadcrumb so the chunk remains
independently understandable:

    "[Section: <heading>]\n\n<content>"

This breadcrumb is not added when the chunk already starts with the heading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.knowledge.models import Block, BlockType, Chunk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# A section body exceeding this character count will be split further.
MAX_SECTION_CHARS: int = 1200

# Target size for sub-chunks produced when splitting an oversized body.
TARGET_CHUNK_CHARS: int = 600

# Block types that are treated as heading markers (start a new section).
_HEADING_TYPES: frozenset[BlockType] = frozenset(
    {
        BlockType.HEADING_1,
        BlockType.HEADING_2,
        BlockType.HEADING_3,
        BlockType.HEADING_4,
        BlockType.LOCAL_HEADING,
    }
)

# Block types whose content contributes to chunk text.
_TEXT_BEARING: frozenset[BlockType] = frozenset(
    {
        BlockType.PARAGRAPH,
        BlockType.BULLETED_LIST_ITEM,
        BlockType.NUMBERED_LIST_ITEM,
        BlockType.TO_DO,
        BlockType.TOGGLE,
        BlockType.CODE,
        BlockType.QUOTE,
        BlockType.CALLOUT,
        BlockType.EQUATION,
        BlockType.TABLE_ROW,
        BlockType.LOCAL_PARAGRAPH,
        BlockType.LOCAL_CODE,
        BlockType.LOCAL_TABLE,
        BlockType.LOCAL_PAGE_TEXT,
    }
)

# Code-like block types — always emitted as independent chunks.
_CODE_TYPES: frozenset[BlockType] = frozenset(
    {BlockType.CODE, BlockType.LOCAL_CODE}
)

# List item types — consecutive runs are merged.
_LIST_TYPES: frozenset[BlockType] = frozenset(
    {BlockType.BULLETED_LIST_ITEM, BlockType.NUMBERED_LIST_ITEM}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    """Accumulator for one heading section."""

    heading_block: Block | None = None
    body_blocks:   list[Block]  = field(default_factory=list)

    @property
    def heading_text(self) -> str:
        return self.heading_block.content if self.heading_block else ""

    def body_text(self) -> str:
        """Concatenate body block contents with appropriate separators."""
        parts: list[str] = []
        prev_type: BlockType | None = None
        for blk in self.body_blocks:
            txt = blk.content.strip()
            if not txt:
                continue
            # Use a blank line when transitioning between different block types
            # to preserve visual / semantic separation.
            if prev_type is not None and blk.block_type != prev_type:
                parts.append("")
            parts.append(txt)
            prev_type = blk.block_type
        return "\n".join(parts)


def _breadcrumb(heading: str) -> str:
    return f"[Section: {heading}]\n\n" if heading else ""


def _split_body(text: str, target: int) -> list[str]:
    """Split *text* into segments of roughly *target* characters.

    Tries to split at paragraph boundaries first, then sentence boundaries,
    then falls back to hard splitting.
    """
    if len(text) <= target:
        return [text]

    segments: list[str] = []
    paragraphs = text.split("\n\n")

    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para).lstrip() if current else para
        if len(candidate) <= target:
            current = candidate
        else:
            if current:
                segments.append(current)
            # Para itself may be too long — try sentence split.
            if len(para) <= target:
                current = para
            else:
                # Hard split at sentence boundaries.
                for sep in (". ", "! ", "? ", "; ", "\n"):
                    parts = para.split(sep)
                    if len(parts) > 1:
                        cur = ""
                        for part in parts:
                            c = (cur + sep + part).lstrip() if cur else part
                            if len(c) <= target:
                                cur = c
                            else:
                                if cur:
                                    segments.append(cur)
                                cur = part
                        if cur:
                            current = cur
                        break
                else:
                    # No suitable separator — hard cut.
                    segments.extend(
                        para[i: i + target] for i in range(0, len(para), target)
                    )
                    current = ""
    if current:
        segments.append(current)
    return segments if segments else [text]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NotionChunker:
    """Convert a flat ordered list of Block objects into Chunks.

    Usage::

        chunks = NotionChunker().chunk(blocks, page_id)
    """

    def __init__(
        self,
        max_section_chars: int = MAX_SECTION_CHARS,
        target_chunk_chars: int = TARGET_CHUNK_CHARS,
    ) -> None:
        self._max_section = max_section_chars
        self._target = target_chunk_chars

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def chunk(self, blocks: list[Block], page_id: str) -> list[Chunk]:
        """Produce a list of Chunks from *blocks* belonging to *page_id*.

        Args:
            blocks   : Ordered list of Block objects (get_all_blocks output).
            page_id  : The UUID of the parent Page (written into each Chunk).

        Returns:
            list[Chunk] ordered by chunk_index.
        """
        sections = self._collect_sections(blocks)
        chunks: list[Chunk] = []
        idx = 0

        for section in sections:
            new_chunks = self._section_to_chunks(section, page_id, idx)
            idx += len(new_chunks)
            chunks.extend(new_chunks)

        return chunks

    # ------------------------------------------------------------------
    # Section collection
    # ------------------------------------------------------------------

    def _collect_sections(self, blocks: list[Block]) -> list[_Section]:
        """Group blocks into sections bounded by heading blocks.

        A heading block starts a new section and is stored as the section's
        ``heading_block``.  Non-heading blocks accumulate in the current
        section's ``body_blocks``.

        Structural / non-text blocks (dividers, table of contents, images,
        bookmarks, …) are skipped.
        """
        sections: list[_Section] = []
        current = _Section()

        for blk in blocks:
            if blk.block_type in _HEADING_TYPES:
                # Flush the existing section if it has any content.
                if current.heading_block is not None or current.body_blocks:
                    sections.append(current)
                current = _Section(heading_block=blk)
            elif blk.block_type in _TEXT_BEARING:
                current.body_blocks.append(blk)
            # else: structural block — skip

        if current.heading_block is not None or current.body_blocks:
            sections.append(current)

        return sections

    # ------------------------------------------------------------------
    # Section → Chunk(s)
    # ------------------------------------------------------------------

    def _section_to_chunks(
        self,
        section: _Section,
        page_id: str,
        start_idx: int,
    ) -> list[Chunk]:
        """Convert one section into one or more Chunks.

        Strategy:
          1. Emit the heading itself as a standalone chunk (if present).
          2. Walk body blocks:
             - Code blocks → independent chunk.
             - List runs    → merge consecutive same-type items, emit when type
                              changes or section ends.
             - Paragraphs   → accumulate; split if > max_section_chars.
        """
        chunks: list[Chunk] = []
        idx = start_idx
        heading = section.heading_text

        # --- Emit heading chunk ---
        if section.heading_block is not None:
            chunks.append(
                Chunk.new(
                    page_id=page_id,
                    content=heading,
                    chunk_index=idx,
                    section=heading,
                    block_ids=[section.heading_block.id],
                )
            )
            idx += 1

        # --- Walk body ---
        body_acc:   list[str]  = []
        body_ids:   list[str]  = []
        list_acc:   list[str]  = []
        list_ids:   list[str]  = []
        list_type:  BlockType | None = None

        def flush_list() -> None:
            nonlocal list_acc, list_ids, list_type, idx
            if not list_acc:
                return
            raw = "\n".join(list_acc)
            content = _breadcrumb(heading) + raw
            chunks.append(
                Chunk.new(
                    page_id=page_id,
                    content=content,
                    chunk_index=idx,
                    section=heading,
                    block_ids=list(list_ids),
                )
            )
            idx += 1
            list_acc = []
            list_ids = []
            list_type = None

        def flush_body() -> None:
            nonlocal body_acc, body_ids, idx
            if not body_acc:
                return
            raw = "\n\n".join(body_acc)
            if len(raw) > self._max_section:
                segments = _split_body(raw, self._target)
                for seg in segments:
                    content = _breadcrumb(heading) + seg
                    chunks.append(
                        Chunk.new(
                            page_id=page_id,
                            content=content,
                            chunk_index=idx,
                            section=heading,
                            block_ids=list(body_ids),
                        )
                    )
                    idx += 1
            else:
                content = _breadcrumb(heading) + raw
                chunks.append(
                    Chunk.new(
                        page_id=page_id,
                        content=content,
                        chunk_index=idx,
                        section=heading,
                        block_ids=list(body_ids),
                    )
                )
                idx += 1
            body_acc = []
            body_ids = []

        for blk in section.body_blocks:
            if blk.block_type in _CODE_TYPES:
                # Code always gets its own chunk.
                flush_list()
                flush_body()
                lang = blk.metadata.get("language", "")
                lang_tag = f"```{lang}\n" if lang else "```\n"
                content = _breadcrumb(heading) + lang_tag + blk.content + "\n```"
                chunks.append(
                    Chunk.new(
                        page_id=page_id,
                        content=content,
                        chunk_index=idx,
                        section=heading,
                        block_ids=[blk.id],
                    )
                )
                idx += 1

            elif blk.block_type in _LIST_TYPES:
                # Flush body accumulator before starting or continuing a list.
                flush_body()
                if list_type is not None and blk.block_type != list_type:
                    flush_list()
                list_type = blk.block_type
                # Indent nested items (depth tracked via parent_block_id chain).
                # For simplicity, all items at depth > 0 get a single indent.
                indent = "  " if blk.parent_block_id else ""
                prefix = "- " if blk.block_type == BlockType.BULLETED_LIST_ITEM else "• "
                list_acc.append(f"{indent}{prefix}{blk.content.strip()}")
                list_ids.append(blk.id)

            else:
                # Paragraph / quote / callout / toggle / to-do / equation.
                flush_list()
                txt = blk.content.strip()
                if txt:
                    body_acc.append(txt)
                    body_ids.append(blk.id)

        flush_list()
        flush_body()

        return chunks
