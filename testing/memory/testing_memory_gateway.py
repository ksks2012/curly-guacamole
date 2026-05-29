from types import SimpleNamespace

from rag.memory.gateway import build_memory_gateway


class _FakeConversationMemory:
    def __init__(self):
        self.active_project = ""
        self.sessions = [{"session_id": "default"}]
        self.current_topics = ["rag"]
        self.cleared = False

    def set_active_project(self, project: str) -> None:
        self.active_project = project

    def ensure_session(self, session_id: str):
        self.sessions.append({"session_id": session_id})
        return {"session_id": session_id}

    def get_state(self):
        return SimpleNamespace(active_project=self.active_project, current_topics=self.current_topics)

    def list_sessions(self):
        return list(self.sessions)

    def clear_session(self):
        self.cleared = True


class _FakeUserMemory:
    def __init__(self):
        self.topics = []
        self.reset_called = False

    def update_from_topics(self, topics):
        self.topics.extend(topics)

    def get_top_interests(self, n=10):
        return [{"topic": topic} for topic in self.topics[:n]]

    def get_profile(self):
        return SimpleNamespace(total_topics_seen=len(self.topics))

    def build_profile_block(self, n=5):
        return ", ".join(self.topics[:n])

    def reset(self):
        self.reset_called = True


class _FakeTimeline:
    def __init__(self):
        self.activities = []

    def record_activity(self, topics, doc_ids=None, date_str=None):
        self.activities.append({"topics": list(topics), "doc_ids": list(doc_ids or []), "date_str": date_str})

    def get_recent(self, days=30):
        return list(self.activities)

    def build_timeline_block(self, days=7):
        return "timeline-block"

    def get_period(self, year, month=None):
        return [{"year": year, "month": month}]

    def get_yearly_summary(self, year):
        return {"rag": year}


class _FakeResearch:
    def __init__(self):
        self.active_session_id = None
        self.notes = []
        self.sessions = []
        self.archived = []

    def set_active(self, session_id):
        self.active_session_id = session_id

    def clear_active(self):
        self.active_session_id = None

    def create(self, name, tags=None):
        session = SimpleNamespace(session_id=f"sid-{len(self.sessions) + 1}", name=name, tags=list(tags or []))
        self.sessions.append(session)
        return session

    def add_note(self, content, session_id=None, source_doc_ids=None):
        note = SimpleNamespace(content=content, session_id=session_id or self.active_session_id, source_doc_ids=list(source_doc_ids or []))
        self.notes.append(note)
        return note

    def get_notes(self):
        return list(self.notes)

    def get_active_session(self):
        if self.active_session_id is None:
            return None
        return SimpleNamespace(session_id=self.active_session_id)

    def list_active(self):
        return list(self.sessions)

    def list_archived(self):
        return list(self.archived)

    def build_session_block(self):
        return "research-block"

    def archive(self, session_id):
        self.archived.append(session_id)

    def get(self, session_id):
        return SimpleNamespace(session_id=session_id)


def test_memory_gateway_routes_conversation_operations():
    gateway = build_memory_gateway(
        memory=_FakeConversationMemory(),
        user_memory=_FakeUserMemory(),
        timeline=_FakeTimeline(),
        research=_FakeResearch(),
    )

    gateway.store("conversation", "active_project", "LangChain")

    assert gateway.retrieve("conversation", "active_project") == "LangChain"
    assert gateway.retrieve("conversation", "sessions")[-1]["session_id"] == "default"

    gateway.clear("conversation")
    assert gateway.conversation.memory.cleared is True


def test_memory_gateway_routes_profile_timeline_and_research_operations():
    gateway = build_memory_gateway(
        memory=_FakeConversationMemory(),
        user_memory=_FakeUserMemory(),
        timeline=_FakeTimeline(),
        research=_FakeResearch(),
    )

    gateway.store("user_profile", "topics", ["RAG", "Memory"])
    gateway.store("timeline", "activity", {"topics": ["RAG"], "doc_ids": ["doc-1"]})
    session = gateway.store("research", "session", {"name": "Agentic RAG", "tags": ["rag"]})
    gateway.store("research", "note", {"content": "Investigate memory gateway"})

    assert [item["topic"] for item in gateway.retrieve("user_profile", "top_interests")] == ["RAG", "Memory"]
    assert gateway.retrieve("timeline", "recent")[0]["doc_ids"] == ["doc-1"]
    assert session.name == "Agentic RAG"
    assert gateway.retrieve("research", "active_session").session_id == session.session_id
    assert gateway.retrieve("research", "notes")[0].content == "Investigate memory gateway"


def test_memory_gateway_exposes_domain_specific_helpers():
    gateway = build_memory_gateway(
        memory=_FakeConversationMemory(),
        user_memory=_FakeUserMemory(),
        timeline=_FakeTimeline(),
        research=_FakeResearch(),
    )

    assert gateway.timeline.get_period(2026, 5)[0]["year"] == 2026
    assert gateway.timeline.get_yearly_summary(2026)["rag"] == 2026

    session = gateway.store("research", "session", {"name": "Timeline"})
    gateway.research.archive(session.session_id)

    assert gateway.retrieve("research", "archived_sessions") == [session.session_id]