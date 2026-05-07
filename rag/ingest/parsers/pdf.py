"""
parsers/pdf.py — PDF parser for the ingestion pipeline.

Wraps PyPDFLoader and normalises metadata to the standard schema:
  source, filename, page (0-based int), total_pages
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def parse_pdf(path: str) -> list[Document]:
    """Load a PDF and return one Document per page with normalised metadata.

    Metadata schema:
        source      : absolute path of the PDF
        filename    : basename of the PDF
        page        : 0-based page index (int)
        total_pages : total number of pages (int)
    """
    abs_path = os.path.abspath(path)
    loader = PyPDFLoader(abs_path)
    pages = loader.load()
    total_pages = len(pages)
    filename = os.path.basename(abs_path)

    normalised: list[Document] = []
    for doc in pages:
        normalised.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": abs_path,
                    "filename": filename,
                    "page": int(doc.metadata.get("page", 0)),
                    "total_pages": total_pages,
                },
            )
        )
    return normalised
