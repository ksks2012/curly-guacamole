"""Filtering utilities for code retrieval result rows."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.code.path_rules import is_test_path


class CodeResultFilter:
    """Composition-friendly helper for filtering code retrieval outputs."""

    def is_test_metadata(self, meta: dict) -> bool:
        """Return True when metadata indicates test-only code."""
        if bool((meta or {}).get("is_test", False)):
            return True
        file_path = str((meta or {}).get("file_path", "") or "")
        return is_test_path(file_path)

    def filter_scored_documents(
        self,
        rows: list[tuple[Document, float]],
        *,
        exclude_tests: bool = True,
    ) -> list[tuple[Document, float]]:
        """Filter ``(Document, score)`` rows using metadata heuristics."""
        if not exclude_tests:
            return list(rows)
        return [
            (doc, score)
            for doc, score in rows
            if not self.is_test_metadata(dict(doc.metadata or {}))
        ]

    def filter_content_rows(
        self,
        rows: list[dict],
        *,
        exclude_tests: bool = True,
    ) -> list[dict]:
        """Filter browse rows where each item is ``{content, metadata}``."""
        if not exclude_tests:
            return list(rows)
        out: list[dict] = []
        for row in rows:
            meta = dict((row or {}).get("metadata") or {})
            if self.is_test_metadata(meta):
                continue
            out.append({"content": row.get("content", ""), "metadata": meta})
        return out
