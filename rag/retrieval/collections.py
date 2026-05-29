"""Collection naming rules shared across retrieval/storage layers."""

from __future__ import annotations


DEFAULT_RAG_COLLECTION = "rag_collection"
QA_COLLECTION_SUFFIX = "_qa"
CODE_BLOCK_COLLECTION = "code_block"


def resolve_doc_collection_name(configured: str | None) -> str:
    """Return the canonical document collection name for the app."""
    name = str(configured or "").strip()
    return name or DEFAULT_RAG_COLLECTION


def resolve_qa_collection_name(doc_collection_name: str) -> str:
    """Return the QA collection name paired with a document collection."""
    return f"{doc_collection_name}{QA_COLLECTION_SUFFIX}"
