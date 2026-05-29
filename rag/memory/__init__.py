"""
Stage C — Conversation Memory, User Memory, Knowledge Timeline,
           and Research Session Tracking

Public surface:
    ConversationMemory      — session manager (C.1)
    MemoryStore             — raw SQLite persistence (C.1/C.2/C.3/C.4)
    SessionState            — dataclass: active_project, current_topics (C.1)
    ConversationTurn        — dataclass: one Q&A exchange (C.1)
    UserMemoryManager       — recency-weighted topic interest profile (C.2)
    UserProfile             — dataclass: top_interests (C.2)
    KnowledgeTimeline       — daily activity log (C.3)
    ResearchSessionManager  — named research sessions (C.4)
    ResearchSession         — dataclass: name, queries, doc_ids, notes (C.4)
    ResearchNote            — dataclass: note content + source docs (C.4)
"""

from rag.memory.models           import ConversationTurn, SessionState
from rag.memory.gateway          import (
    IMemoryManager,
    ConversationHistoryMemory,
    UserProfileMemory as UserProfileMemoryGateway,
    TimelineMemory,
    ResearchContextMemory,
    MemoryGateway,
    build_memory_gateway,
)
from rag.memory.store            import MemoryStore
from rag.memory.manager          import ConversationMemory
from rag.memory.user_memory      import UserMemoryManager, UserProfile
from rag.memory.timeline         import KnowledgeTimeline
from rag.memory.research_session import ResearchSessionManager, ResearchSession, ResearchNote

__all__ = [
    "ConversationMemory",
    "IMemoryManager",
    "ConversationHistoryMemory",
    "UserProfileMemoryGateway",
    "TimelineMemory",
    "ResearchContextMemory",
    "MemoryGateway",
    "build_memory_gateway",
    "MemoryStore",
    "SessionState",
    "ConversationTurn",
    "UserMemoryManager",
    "UserProfile",
    "KnowledgeTimeline",
    "ResearchSessionManager",
    "ResearchSession",
    "ResearchNote",
]
