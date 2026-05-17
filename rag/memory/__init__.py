"""
Stage C — Conversation Memory, User Memory, Knowledge Timeline

Public surface:
    ConversationMemory   — session manager (C.1)
    MemoryStore          — raw SQLite persistence (C.1/C.2/C.3)
    SessionState         — dataclass: active_project, current_topics, recent_questions (C.1)
    ConversationTurn     — dataclass: one Q&A exchange (C.1)
    UserMemoryManager    — recency-weighted topic interest profile (C.2)
    UserProfile          — dataclass: top_interests, total_topics_seen (C.2)
    KnowledgeTimeline    — daily activity log (C.3)
"""

from rag.memory.models      import ConversationTurn, SessionState
from rag.memory.store       import MemoryStore
from rag.memory.manager     import ConversationMemory
from rag.memory.user_memory import UserMemoryManager, UserProfile
from rag.memory.timeline    import KnowledgeTimeline

__all__ = [
    "ConversationMemory",
    "MemoryStore",
    "SessionState",
    "ConversationTurn",
    "UserMemoryManager",
    "UserProfile",
    "KnowledgeTimeline",
]
