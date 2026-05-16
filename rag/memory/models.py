"""
Data models for Stage C.1 — Conversation Memory.

These are plain Python dataclasses with no framework dependencies.
Persistence is handled by MemoryStore (store.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationTurn:
    """One Q&A exchange within a session.

    Attributes
    ----------
    session_id     : Parent session identifier.
    question       : The user's question verbatim.
    answer_summary : First 500 chars of the LLM response, stored for context.
    topics         : 1-5 topic tags extracted from this turn (may be empty).
    timestamp      : ISO-8601 UTC string.
    seq            : Sequence number within the session (1-based).
    id             : Database row id (assigned on insert, None before save).
    """

    session_id:     str
    question:       str
    answer_summary: str         = ""
    topics:         list[str]   = field(default_factory=list)
    timestamp:      str         = ""
    seq:            int         = 0
    id:             int | None  = None


@dataclass
class SessionState:
    """Current state of a conversation session.

    Attributes
    ----------
    session_id       : Unique session identifier.
    active_project   : Inferred or manually set project / goal (free text).
    current_topics   : Rolling list of recent topics, most-recent first,
                       trimmed to ``max_topics``.
    recent_questions : Last ``max_recent`` ConversationTurn objects.
    created_at       : ISO-8601 UTC string.
    updated_at       : ISO-8601 UTC string.
    metadata         : Arbitrary JSON-serialisable dict for future extension.
    """

    session_id:       str
    active_project:   str                    = ""
    current_topics:   list[str]              = field(default_factory=list)
    recent_questions: list[ConversationTurn] = field(default_factory=list)
    created_at:       str                    = ""
    updated_at:       str                    = ""
    metadata:         dict[str, Any]         = field(default_factory=dict)
