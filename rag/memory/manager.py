"""
Stage C.1 — ConversationMemory

Maintains three pieces of persistent session state per conversation:

  active_project   — what the user is currently building / researching
  current_topics   — rolling list of recent discussion topics (most-recent first)
  recent_questions — last N Q&A turns (question + answer excerpt + topics)

These are injected into the RAG prompt as a compact context block so the LLM
can tailor its answers without being asked to repeat itself.

Typical usage (managed by LocalLlamaClient)::

    memory = ConversationMemory(store=MemoryStore(db_path), llm=chat_llm)
    memory.ensure_session("session-abc")

    # Called automatically by RAGEngine.answer():
    block = memory.build_context_block()    # injected into prompt
    ...
    memory.add_turn(question, answer_text)  # updated after each response

    # Manual project override from the UI:
    memory.set_active_project("Notion AI Knowledge Management System")

    # Read current state for display in a dashboard tab:
    state = memory.get_state()
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rag.memory.models import ConversationTurn, SessionState
from rag.memory.store  import MemoryStore
from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

log = AppLogger.get(__name__)

_DEFAULT_SESSION = "default"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_topics(
    existing: list[str],
    new:      list[str],
    max_topics: int,
) -> list[str]:
    """Prepend *new* topics to *existing*, deduplicating by lowercase, then trim."""
    merged: list[str] = []
    seen: set[str] = set()
    for t in new + existing:
        key = t.lower().strip()
        if key and key not in seen:
            merged.append(t)
            seen.add(key)
    return merged[:max_topics]


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Session-scoped conversation memory with SQLite persistence.

    Args
    ----
    store               : MemoryStore instance.
    llm                 : ChatOpenAI (optional) — used for topic extraction and
                          project inference.  When None, topics are not extracted.
    session_id          : Session identifier.  Defaults to ``"default"``.
    max_recent          : Maximum turns to keep in ``recent_questions``.
    max_topics          : Maximum entries in ``current_topics``.
    extract_topics      : Whether to call the LLM to extract topics after each turn.
    auto_infer_project  : Whether to periodically infer ``active_project`` via LLM.
    infer_project_every : Refresh ``active_project`` every N turns.
    """

    def __init__(
        self,
        store:              MemoryStore,
        llm:                "ChatOpenAI | None" = None,
        session_id:         str                 = _DEFAULT_SESSION,
        max_recent:         int                 = 20,
        max_topics:         int                 = 10,
        extract_topics:     bool                = True,
        auto_infer_project: bool                = True,
        infer_project_every: int                = 10,
    ) -> None:
        self._store               = store
        self._llm                 = llm
        self._max_recent          = max_recent
        self._max_topics          = max_topics
        self._extract_topics      = extract_topics and (llm is not None)
        self._auto_infer          = auto_infer_project and (llm is not None)
        self._infer_every         = infer_project_every
        self.session_id           = session_id
        self._state: SessionState | None = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def ensure_session(self, session_id: str | None = None) -> SessionState:
        """Load or create a session.  Call once before using the memory.

        Calling again with a different *session_id* switches the active session.
        """
        if session_id is not None:
            self.session_id = session_id

        state = self._store.get_or_create_session(self.session_id)
        # Populate recent_questions from DB
        state.recent_questions = self._store.get_recent_turns(
            self.session_id, limit=self._max_recent
        )
        self._state = state
        log.debug(
            "ConversationMemory: session=%r  turns=%d  topics=%d",
            self.session_id,
            len(state.recent_questions),
            len(state.current_topics),
        )
        return state

    def _require_state(self) -> SessionState:
        if self._state is None:
            return self.ensure_session()
        return self._state

    # ------------------------------------------------------------------
    # Manual overrides
    # ------------------------------------------------------------------

    def set_active_project(self, project: str) -> None:
        """Manually set the active project (from UI or explicit call)."""
        state = self._require_state()
        state.active_project = project.strip()
        self._store.update_session(
            self.session_id, state.active_project, state.current_topics
        )
        log.info("ConversationMemory: active_project set to %r", state.active_project)

    # ------------------------------------------------------------------
    # Core: add a turn
    # ------------------------------------------------------------------

    def add_turn(self, question: str, answer: str) -> ConversationTurn:
        """Record a Q&A exchange, update topics and optionally infer project.

        Args
        ----
        question : The user's question verbatim.
        answer   : Full LLM response text (stored truncated to 500 chars).

        Returns
        -------
        The saved ConversationTurn (with ``id`` and ``seq`` assigned).
        """
        state = self._require_state()

        # --- Extract topics ---
        topics: list[str] = []
        if self._extract_topics:
            try:
                topics = self._call_topic_extraction(question, answer)
            except Exception as exc:
                log.warning("Topic extraction failed: %s", exc)

        # --- Build and save turn ---
        turn = ConversationTurn(
            session_id=self.session_id,
            question=question,
            answer_summary=answer[:500],
            topics=topics,
            timestamp=_utcnow(),
        )
        turn = self._store.save_turn(turn)

        # --- Update rolling state ---
        state.current_topics = _merge_topics(
            state.current_topics, topics, self._max_topics
        )
        state.recent_questions = (state.recent_questions + [turn])[-self._max_recent:]
        self._store.update_session(
            self.session_id, state.active_project, state.current_topics
        )

        # --- Periodic project inference ---
        if self._auto_infer and len(state.recent_questions) % self._infer_every == 0:
            try:
                self._refresh_active_project()
            except Exception as exc:
                log.warning("Project inference failed: %s", exc)

        log.debug(
            "ConversationMemory: turn=%d  topics=%s",
            turn.seq, topics,
        )
        return turn

    # ------------------------------------------------------------------
    # Prompt integration
    # ------------------------------------------------------------------

    def build_context_block(self) -> str:
        """Return a compact context string for injection into the RAG prompt.

        Returns an empty string when the session has no meaningful state.
        """
        state = self._require_state()
        if not state.active_project and not state.current_topics and not state.recent_questions:
            return ""

        lines: list[str] = []

        if state.active_project:
            lines.append(f"Active Project: {state.active_project}")

        if state.current_topics:
            lines.append(f"Current Focus: {', '.join(state.current_topics)}")

        if state.recent_questions:
            lines.append("Recent Questions:")
            for t in state.recent_questions[-5:]:   # last 5 are enough for context
                lines.append(f"  • {t.question}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_state(self) -> SessionState:
        """Return a fresh copy of the current SessionState."""
        return self._require_state()

    def list_sessions(self) -> list[dict]:
        """Return summary dicts for all sessions in the store."""
        return self._store.list_sessions()

    def clear_session(self) -> None:
        """Delete all turns and reset state for the current session."""
        self._store.delete_session(self.session_id)
        self._state = None
        self.ensure_session(self.session_id)
        log.info("ConversationMemory: session %r cleared", self.session_id)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_topic_extraction(self, question: str, answer: str) -> list[str]:
        """Ask the LLM to extract 1-5 topic tags from this Q&A turn.

        Returns a (possibly empty) list of topic strings.
        """
        from rag.prompt import TOPIC_EXTRACTION_PROMPT

        prompt = TOPIC_EXTRACTION_PROMPT.format(
            question=question,
            answer_excerpt=answer[:600],
        )
        response = self._llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        result = json.loads(raw.strip())
        if isinstance(result, list):
            return [str(t).strip() for t in result if t][:5]
        return []

    def _refresh_active_project(self) -> None:
        """Infer active_project from recent questions and update session."""
        from rag.prompt import PROJECT_INFERENCE_PROMPT

        state = self._require_state()
        if not state.recent_questions:
            return

        recent_qs = "\n".join(
            f"  - {t.question}" for t in state.recent_questions[-10:]
        )
        topics_str = ", ".join(state.current_topics[:8]) or "(none)"
        prompt = PROJECT_INFERENCE_PROMPT.format(
            questions=recent_qs,
            topics=topics_str,
        )
        response = self._llm.invoke(prompt)
        project  = (response.content if hasattr(response, "content") else str(response)).strip()
        if project:
            project = project.strip('"').strip("'")
            state.active_project = project
            self._store.update_session(
                self.session_id, project, state.current_topics
            )
            log.info("ConversationMemory: active_project inferred → %r", project)

    def infer_project(self) -> str:
        """Manually trigger project inference and return the result."""
        self._refresh_active_project()
        return self._require_state().active_project
