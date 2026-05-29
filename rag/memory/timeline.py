"""
Stage C.3 — Knowledge Timeline

Tracks what the user has been working on over time by recording a daily
activity entry for every Q-A turn.  Entries are aggregated per calendar day
so repeated questions in the same day accumulate into one row.

Each daily entry stores:
  - topics    : all topic tags that surfaced on that day (deduped)
  - doc_ids   : all documents referenced in retrieved chunks on that day
  - question_count : total turns on that day

This enables queries like:
  "What was I working on in April 2026?"
  "Show me everything about RAG systems across all months"

Storage: shared ``MemoryStore`` → ``timeline_entries`` table.

Usage::

    tl = KnowledgeTimeline(store=mem_store)
    tl.record_activity(["RAG Architecture", "Chunking"], doc_ids=["my_doc"])
    block = tl.build_timeline_block(days=7)
    summary = tl.get_yearly_summary(2026)
    # {"RAG Architecture": 12, "Knowledge Graph": 5, ...}
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

from utils.logger import AppLogger

if TYPE_CHECKING:
    from rag.memory.store import MemoryStore

log = AppLogger.get(__name__)


class KnowledgeTimeline:
    """Daily knowledge activity log with topic and document tracking.

    Args
    ----
    store : Shared ``MemoryStore`` instance (provides ``timeline_entries`` table).
    """

    def __init__(self, store: "MemoryStore") -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_activity(
        self,
        topics:   list[str],
        doc_ids:  list[str]   = [],
        date_str: str | None  = None,
    ) -> None:
        """Record a Q-A turn's topics and referenced documents.

        Merges into the existing entry for *date_str* (today by default)
        so multiple turns on the same day accumulate.

        Args
        ----
        topics   : Topic tags from the current turn.
        doc_ids  : Document IDs of retrieved chunks (may be empty).
        date_str : ``YYYY-MM-DD`` override; defaults to today (UTC).
        """
        if date_str is None:
            date_str = date.today().isoformat()

        # Filter blanks
        clean_topics  = [t.strip() for t in topics  if t.strip()]
        clean_doc_ids = [d.strip() for d in doc_ids if d.strip()]

        if not clean_topics and not clean_doc_ids:
            return  # nothing to record

        self._store.upsert_timeline_entry(date_str, clean_topics, clean_doc_ids)
        log.debug("KnowledgeTimeline: recorded %d topics on %s", len(clean_topics), date_str)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_recent(self, days: int = 30) -> list[dict]:
        """Return entries for the last *days* calendar days, oldest first.

        Each entry: ``{"date", "year", "month", "topics", "doc_ids", "question_count"}``.
        """
        end   = date.today().isoformat()
        start = (date.today() - timedelta(days=days)).isoformat()
        return self._store.get_timeline(start, end)

    def get_period(self, year: int, month: int | None = None) -> list[dict]:
        """Return all entries for *year* (optionally filtered to *month*).

        Each entry: ``{"date", "year", "month", "topics", "doc_ids", "question_count"}``.
        """
        rows = self._store.get_timeline_by_year(year)
        if month is not None:
            rows = [r for r in rows if r["month"] == month]
        return rows

    def get_yearly_summary(self, year: int) -> dict[str, int]:
        """Return a topic-frequency map for *year*.

        Returns
        -------
        ``{"topic": total_count}`` ordered by count descending.
        The count is the number of *days* the topic appeared (not total turns).
        """
        entries = self._store.get_timeline_by_year(year)
        freq: dict[str, int] = {}
        for entry in entries:
            for topic in entry["topics"]:
                freq[topic] = freq.get(topic, 0) + 1
        return dict(sorted(freq.items(), key=lambda kv: kv[1], reverse=True))
    
    def clear(self) -> None:
        """Delete all timeline entries."""
        self._store.clear_timeline()
        log.info("KnowledgeTimeline: all entries cleared")

    # ------------------------------------------------------------------
    # Prompt integration
    # ------------------------------------------------------------------

    def build_timeline_block(self, days: int = 7) -> str:
        """Return a compact block of recent activity for prompt injection.

        Shows up to 5 most-recent active days.  Returns an empty string when
        there is no recorded activity in the window.

        Example output::

            Recent Activity:
              2026-05-17: RAG Architecture, Vector Search (3 questions)
              2026-05-16: Topic Clustering, Cross-Document Linking (5 questions)
        """
        entries = self.get_recent(days=days)
        if not entries:
            return ""

        lines = ["Recent Activity:"]
        for entry in entries[-5:]:   # most-recent 5 active days
            topics_str = ", ".join(entry["topics"][:5])
            q = entry["question_count"]
            q_str = f"({q} question{'s' if q != 1 else ''})"
            lines.append(f"  {entry['date']}: {topics_str} {q_str}")

        return "\n".join(lines)
