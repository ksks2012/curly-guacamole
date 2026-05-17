"""
Stage C.1 — Conversation Memory

Public surface:
    ConversationMemory   — session manager (use this)
    MemoryStore          — raw SQLite persistence
    SessionState         — dataclass: active_project, current_topics, recent_questions
    ConversationTurn     — dataclass: one Q&A exchange
"""

from rag.memory.models  import ConversationTurn, SessionState
from rag.memory.store   import MemoryStore
from rag.memory.manager import ConversationMemory

__all__ = [
    "ConversationMemory",
    "MemoryStore",
    "SessionState",
    "ConversationTurn",
]
