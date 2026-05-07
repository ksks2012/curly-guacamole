"""
parsers/markdown.py — Markdown parser for the ingestion pipeline.

Parses a .md file and splits it into Documents at heading boundaries
(# / ## / ###), preserving the heading hierarchy in metadata.

Metadata schema per Document:
    source     : absolute path of the .md file
    filename   : basename of the .md file
    heading    : nearest heading text above this block (empty string if none)
    hierarchy  : list of ancestor headings from H1 → current level
                 e.g. ["Chapter 1", "Section 1.2"]
    level      : heading depth of this block (0 = before first heading)
"""

import os
import re
from langchain_core.documents import Document

# Matches ATX headings: one or more # followed by a space and text
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def parse_markdown(path: str) -> list[Document]:
    """Load a Markdown file and return one Document per heading section.

    Sections are defined as the text between two consecutive headings (any level).
    Content before the first heading is returned as a single Document with
    empty heading metadata.
    """
    abs_path = os.path.abspath(path)
    filename = os.path.basename(abs_path)

    with open(abs_path, encoding="utf-8") as f:
        text = f.read()

    # Collect all heading positions and their depths
    heading_spans: list[tuple[int, int, str]] = []  # (start, depth, text)
    for m in _HEADING_RE.finditer(text):
        depth = len(m.group(1))
        heading_text = m.group(2).strip()
        heading_spans.append((m.start(), depth, heading_text))

    # Build sections: slice text between heading boundaries
    section_starts = [0] + [pos for pos, _, _ in heading_spans]
    section_ends = [pos for pos, _, _ in heading_spans] + [len(text)]

    docs: list[Document] = []

    # Maintain a stack to track heading hierarchy
    hierarchy_stack: list[tuple[int, str]] = []  # (depth, heading_text)

    for i, (start, end) in enumerate(zip(section_starts, section_ends)):
        content = text[start:end].strip()
        if not content:
            continue

        # The heading associated with this section is heading_spans[i-1] (i>0)
        if i == 0:
            # Preamble: content before the first heading
            heading = ""
            level = 0
            hierarchy: list[str] = []
        else:
            _, depth, heading = heading_spans[i - 1]
            level = depth
            # Trim stack to current depth
            hierarchy_stack = [(d, h) for d, h in hierarchy_stack if d < depth]
            hierarchy_stack.append((depth, heading))
            hierarchy = [h for _, h in hierarchy_stack]

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": abs_path,
                    "filename": filename,
                    "heading": heading,
                    "hierarchy": hierarchy,
                    "level": level,
                },
            )
        )

    return docs
