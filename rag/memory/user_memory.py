"""
Stage C.2 — Semantic User Memory

Builds a long-term profile of the user's research interests by tracking which
topics appear across all sessions.  Topics are scored with a recency-weighted
accumulator so frequently-discussed *and* recently-active areas float to the top.

Weight update rule (applied on every occurrence of a topic):
    new_weight = old_weight * 0.9 + 1.0

This is equivalent to an EMA where each occurrence contributes 1.0 and the
existing score decays by 10 % between updates.  A topic discussed 10 times
converges near 10 (if evenly spaced); a topic discussed once and then ignored
gradually decays toward zero relative to active topics.

Storage: shared ``MemoryStore`` → ``user_interests`` table.

Usage::

    user_mem = UserMemoryManager(store=mem_store)
    user_mem.update_from_topics(["RAG Architecture", "Vector Search"])
    block = user_mem.build_profile_block(n=5)
    # "Frequent Research Areas: RAG Architecture, Vector Search, ..."
    profile = user_mem.get_profile()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from utils.logger import AppLogger

if TYPE_CHECKING:
    from rag.memory.store import MemoryStore

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """Snapshot of the user's current interest profile.

    Attributes
    ----------
    top_interests     : List of ``{"topic", "count", "weight"}`` dicts,
                        ordered by weight descending.
    total_topics_seen : Total number of distinct topics ever recorded.
    """

    top_interests:     list[dict] = field(default_factory=list)
    total_topics_seen: int        = 0


# ---------------------------------------------------------------------------
# UserMemoryManager
# ---------------------------------------------------------------------------

class UserMemoryManager:
    """Maintains a recency-weighted topic interest profile across all sessions.

    Args
    ----
    store : Shared ``MemoryStore`` instance (provides ``user_interests`` table).
    """

    def __init__(self, store: "MemoryStore") -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def update_from_topics(self, topics: list[str]) -> None:
        """Record each topic in *topics*, updating weights in the store.

        Silently skips blank or whitespace-only strings.
        """
        for topic in topics:
            t = topic.strip()
            if t:
                self._store.upsert_interest(t)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_top_interests(self, n: int = 10) -> list[dict]:
        """Return top *n* interests sorted by recency-weighted score.

        Each entry: ``{"topic", "count", "weight", "first_seen", "last_seen"}``.
        """
        return self._store.get_top_interests(n)

    def get_profile(self) -> UserProfile:
        """Return a ``UserProfile`` snapshot with the top 20 interests."""
        interests = self._store.get_top_interests(20)
        return UserProfile(
            top_interests=interests,
            total_topics_seen=len(interests),
        )

    # ------------------------------------------------------------------
    # Prompt integration
    # ------------------------------------------------------------------

    def build_profile_block(self, n: int = 5) -> str:
        """Return a one-line summary suitable for injecting into a RAG prompt.

        Returns an empty string when no interests have been recorded yet.

        Example output::

            Frequent Research Areas: RAG Architecture, Vector Search, Knowledge Graph
        """
        top = self.get_top_interests(n)
        if not top:
            return ""
        interests = ", ".join(item["topic"] for item in top)
        return f"Frequent Research Areas: {interests}"

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Delete all stored interest records (irreversible)."""
        self._store.clear_interests()
        log.info("UserMemoryManager: all interests cleared")
