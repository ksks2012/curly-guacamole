"""
filters.py — retrieval-time filter builder for the RAG pipeline.

SearchFilter is a plain dataclass that holds optional constraints across the
five supported dimensions.  Call to_chroma() to obtain a Chroma-compatible
``where`` dict that can be passed directly to similarity_search* methods.

Supported dimensions
--------------------
source_id      : exact match on ``doc_id`` field  (= source_id in schema)
workspace      : exact match on ``workspace`` field
document_type  : exact match on ``document_type`` field  (pdf/markdown/text)
tag            : substring match inside the comma-joined ``tags`` field
created_after  : ISO-8601 date/datetime string — lower bound on ``created_time``
created_before : ISO-8601 date/datetime string — upper bound on ``created_time``

Chroma where-clause rules
--------------------------
- Single condition  : ``{"field": {"$op": value}}``
- Multiple conditions: ``{"$and": [cond1, cond2, …]}``
- Substring match   : ``{"$contains": value}`` (works for string fields)
- Date comparison   : ISO-8601 strings compare lexicographically, which is
                      correct as long as timezone suffix is consistent.

Usage::

    f = SearchFilter(workspace="work", tag="llm")
    where = f.to_chroma()
    # → {"$and": [{"workspace": {"$eq": "work"}}, {"tags": {"$contains": "llm"}}]}
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SearchFilter:
    """Immutable-style filter spec; set only the fields you want to restrict."""

    # """Exact doc_id match.  Maps to the ``doc_id`` Chroma metadata field."""
    source_id: str | None = None

    # """Exact workspace label match."""
    workspace: str | None = None

    # """Exact document type: 'pdf', 'markdown', or 'text'."""
    document_type: str | None = None

    # """Substring match inside the comma-joined ``tags`` field."""
    tag: str | None = None

    """Lower bound on ``created_time``.  Format: 'YYYY-MM-DD' or ISO-8601."""
    created_after: str | None = None

    """Upper bound on ``created_time``.  Format: 'YYYY-MM-DD' or ISO-8601."""
    created_before: str | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """Return True when no dimension is constrained."""
        return not any(dataclasses.asdict(self).values())

    def to_chroma(self) -> dict | None:
        """Build a Chroma-compatible ``where`` dict.

        Returns None when all fields are unset (no filtering).
        Returns a single condition dict when exactly one field is set.
        Returns ``{"$and": [...]}`` when multiple fields are set.
        """
        conditions: list[dict] = []

        if self.source_id:
            conditions.append({"doc_id": {"$eq": self.source_id}})
        if self.workspace:
            conditions.append({"workspace": {"$eq": self.workspace}})
        if self.document_type:
            conditions.append({"document_type": {"$eq": self.document_type}})
        if self.tag:
            conditions.append({"tags": {"$contains": self.tag}})
        if self.created_after:
            # Pad to full ISO-8601 for reliable string comparison
            after = _pad_date(self.created_after)
            conditions.append({"created_time": {"$gte": after}})
        if self.created_before:
            before = _pad_date(self.created_before, end_of_day=True)
            conditions.append({"created_time": {"$lte": before}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def summary(self) -> str:
        """Return a short human-readable description of active constraints."""
        parts: list[str] = []
        if self.source_id:
            parts.append(f"source={self.source_id}")
        if self.workspace:
            parts.append(f"workspace={self.workspace}")
        if self.document_type:
            parts.append(f"type={self.document_type}")
        if self.tag:
            parts.append(f"tag={self.tag}")
        if self.created_after:
            parts.append(f"after={self.created_after}")
        if self.created_before:
            parts.append(f"before={self.created_before}")
        return "  ·  ".join(parts) if parts else "off"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pad_date(value: str, end_of_day: bool = False) -> str:
    """Ensure a bare 'YYYY-MM-DD' string becomes a full ISO timestamp.

    Chroma stores created_time as 'YYYY-MM-DDTHH:MM:SS+00:00'.  Comparing
    a bare date string against that would miss records within the same day,
    so we pad start-of-day to 'T00:00:00+00:00' and end-of-day to
    'T23:59:59+00:00'.
    """
    value = value.strip()
    if "T" in value:
        return value
    suffix = "T23:59:59+00:00" if end_of_day else "T00:00:00+00:00"
    return value + suffix
