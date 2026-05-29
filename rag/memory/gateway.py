"""Unified memory gateway and adapters for conversation, profile, timeline, and research state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class IMemoryManager(Protocol):
    """Minimal memory interface shared by gateway adapters."""

    def store(self, key: str, value: object) -> object | None:
        ...

    def retrieve(self, key: str) -> object:
        ...

    def clear(self) -> None:
        ...


@dataclass
class ConversationHistoryMemory(IMemoryManager):
    """Adapter exposing session-scoped conversation memory through a uniform API."""

    memory: object

    def store(self, key: str, value: object) -> object | None:
        if key == "active_project":
            self.memory.set_active_project(str(value))
            return None
        if key == "session_id":
            return self.memory.ensure_session(str(value))
        raise KeyError(f"Unsupported conversation store key: {key}")

    def retrieve(self, key: str) -> object:
        if key == "state":
            return self.memory.get_state()
        if key == "sessions":
            return self.memory.list_sessions()
        if key == "active_project":
            return self.memory.get_state().active_project
        if key == "current_topics":
            return list(self.memory.get_state().current_topics)
        raise KeyError(f"Unsupported conversation retrieve key: {key}")

    def clear(self) -> None:
        self.memory.clear_session()


@dataclass
class UserProfileMemory(IMemoryManager):
    """Adapter exposing long-term user interest state through a uniform API."""

    user_memory: object

    def store(self, key: str, value: object) -> object | None:
        if key == "topics":
            topics = list(value) if isinstance(value, list) else [str(value)]
            self.user_memory.update_from_topics(topics)
            return None
        raise KeyError(f"Unsupported user_profile store key: {key}")

    def retrieve(self, key: str) -> object:
        if key == "top_interests":
            return self.user_memory.get_top_interests(10)
        if key == "profile":
            return self.user_memory.get_profile()
        if key == "profile_block":
            return self.user_memory.build_profile_block(n=5)
        raise KeyError(f"Unsupported user_profile retrieve key: {key}")

    def clear(self) -> None:
        self.user_memory.reset()


@dataclass
class TimelineMemory(IMemoryManager):
    """Adapter exposing recent activity history through a uniform API."""

    timeline: object

    def store(self, key: str, value: object) -> object | None:
        if key != "activity" or not isinstance(value, dict):
            raise KeyError(f"Unsupported timeline store key: {key}")
        self.timeline.record_activity(
            topics=list(value.get("topics", [])),
            doc_ids=list(value.get("doc_ids", [])),
            date_str=value.get("date_str"),
        )
        return None

    def retrieve(self, key: str) -> object:
        if key == "recent":
            return self.timeline.get_recent(days=30)
        if key == "timeline_block":
            return self.timeline.build_timeline_block(days=7)
        raise KeyError(f"Unsupported timeline retrieve key: {key}")

    def clear(self) -> None:
        self.timeline.clear()

    def get_period(self, year: int, month: int | None = None) -> list[dict]:
        return self.timeline.get_period(year, month)

    def get_yearly_summary(self, year: int) -> dict[str, int]:
        return self.timeline.get_yearly_summary(year)


@dataclass
class ResearchContextMemory(IMemoryManager):
    """Adapter exposing long-lived research session state through a uniform API."""

    research: object

    def store(self, key: str, value: object) -> object | None:
        if key == "active_session":
            self.research.set_active(str(value))
            return None
        if key == "note":
            if not isinstance(value, dict):
                raise ValueError("Research note payload must be a dict")
            return self.research.add_note(
                content=str(value.get("content", "")),
                session_id=value.get("session_id"),
                source_doc_ids=list(value.get("source_doc_ids", [])),
            )
        if key == "session":
            if not isinstance(value, dict):
                raise ValueError("Research session payload must be a dict")
            session = self.research.create(
                str(value.get("name", "")),
                tags=list(value.get("tags", [])),
            )
            if value.get("set_active", True):
                self.research.set_active(session.session_id)
            return session
        raise KeyError(f"Unsupported research store key: {key}")

    def retrieve(self, key: str) -> object:
        if key == "active_session":
            return self.research.get_active_session()
        if key == "active_sessions":
            return self.research.list_active()
        if key == "archived_sessions":
            return self.research.list_archived()
        if key == "notes":
            return self.research.get_notes()
        if key == "session_block":
            return self.research.build_session_block()
        raise KeyError(f"Unsupported research retrieve key: {key}")

    def clear(self) -> None:
        self.research.clear_active()

    def archive(self, session_id: str | None = None) -> None:
        sid = session_id or self.research.active_session_id
        if sid:
            self.research.archive(sid)

    def get(self, session_id: str) -> object:
        return self.research.get(session_id)


@dataclass
class MemoryGateway:
    """Single memory entry point that exposes clear domain boundaries."""

    conversation: ConversationHistoryMemory
    user_profile: UserProfileMemory
    timeline: TimelineMemory
    research: ResearchContextMemory

    def store(self, scope: str, key: str, value: object) -> object | None:
        return self._resolve_scope(scope).store(key, value)

    def retrieve(self, scope: str, key: str) -> object:
        return self._resolve_scope(scope).retrieve(key)

    def clear(self, scope: str) -> None:
        self._resolve_scope(scope).clear()

    def _resolve_scope(self, scope: str) -> IMemoryManager:
        mapping = {
            "conversation": self.conversation,
            "user_profile": self.user_profile,
            "timeline": self.timeline,
            "research": self.research,
        }
        try:
            return mapping[scope]
        except KeyError as exc:
            raise KeyError(f"Unknown memory scope: {scope}") from exc


def build_memory_gateway(
    *,
    memory: object,
    user_memory: object,
    timeline: object,
    research: object,
) -> MemoryGateway:
    """Create the application's unified memory gateway from existing managers."""
    return MemoryGateway(
        conversation=ConversationHistoryMemory(memory=memory),
        user_profile=UserProfileMemory(user_memory=user_memory),
        timeline=TimelineMemory(timeline=timeline),
        research=ResearchContextMemory(research=research),
    )