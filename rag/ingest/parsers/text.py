"""
parsers/text.py — Plain-text parser for the ingestion pipeline.

Loads a .txt file as a single Document.

Metadata schema:
    source   : absolute path of the text file
    filename : basename of the text file
"""

import os
from langchain_core.documents import Document


def parse_text(path: str) -> list[Document]:
    """Load a plain-text file and return a single Document."""
    abs_path = os.path.abspath(path)
    filename = os.path.basename(abs_path)

    with open(abs_path, encoding="utf-8") as f:
        content = f.read()

    return [
        Document(
            page_content=content,
            metadata={
                "source": abs_path,
                "filename": filename,
            },
        )
    ]
