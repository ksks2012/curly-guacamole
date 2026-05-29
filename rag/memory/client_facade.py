"""Compatibility facade for memory-related client operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.memory.research_session import ResearchSession


@dataclass
class ClientMemoryFacade:
    """Groups memory, user-profile, timeline, and research-session helpers."""

    memory: object
    user_memory: object
    timeline: object
    research: object

    def set_active_project(self, project: str) -> None:
        self.memory.set_active_project(project)

    def infer_project(self) -> str:
        return self.memory.infer_project()

    def get_memory_state(self):
        return self.memory.get_state()

    def list_sessions(self) -> list[dict]:
        return self.memory.list_sessions()

    def clear_memory_session(self) -> None:
        self.memory.clear_session()

    def switch_memory_session(self, session_id: str) -> None:
        self.memory.ensure_session(session_id)

    def get_user_interests(self, n: int = 10) -> list[dict]:
        return self.user_memory.get_top_interests(n)

    def get_user_profile(self):
        return self.user_memory.get_profile()

    def get_timeline_recent(self, days: int = 30) -> list[dict]:
        return self.timeline.get_recent(days)

    def get_timeline_period(self, year: int, month: int | None = None) -> list[dict]:
        return self.timeline.get_period(year, month)

    def get_yearly_summary(self, year: int) -> dict[str, int]:
        return self.timeline.get_yearly_summary(year)

    def start_research_session(
        self,
        name: str,
        tags: list[str] | None = None,
        set_active: bool = True,
    ) -> "ResearchSession":
        session = self.research.create(name, tags=list(tags or []))
        if set_active:
            self.research.set_active(session.session_id)
        return session

    def get_research_session(self, session_id: str | None = None):
        if session_id:
            return self.research.get(session_id)
        return self.research.get_active_session()

    def list_research_sessions(self, archived: bool = False) -> list:
        return self.research.list_archived() if archived else self.research.list_active()

    def archive_research_session(self, session_id: str | None = None) -> None:
        sid = session_id or self.research.active_session_id
        if sid:
            self.research.archive(sid)

    def add_research_note(
        self,
        content: str,
        source_doc_ids: list[str] | None = None,
        session_id: str | None = None,
    ):
        return self.research.add_note(
            content,
            session_id=session_id,
            source_doc_ids=list(source_doc_ids or []),
        )

    def get_research_notes(self, session_id: str | None = None) -> list:
        return self.research.get_notes(session_id)