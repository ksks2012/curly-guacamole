"""
SQLite persistence for Stage C.1 — Conversation Memory.

Schema
------
conversation_sessions
    session_id    TEXT PK
    created_at    TEXT
    updated_at    TEXT
    active_project TEXT
    current_topics TEXT   (JSON array of strings)
    metadata      TEXT   (JSON object)

conversation_turns
    id            INTEGER PK AUTOINCREMENT
    session_id    TEXT  FK → conversation_sessions
    seq           INTEGER   (1-based, per session)
    timestamp     TEXT
    question      TEXT
    answer_summary TEXT
    topics        TEXT   (JSON array of strings)

Uses SQLAlchemy Core (same pattern as rag/knowledge/store.py).
WAL mode enabled for concurrent read safety.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.engine import Connection, Engine

from rag.memory.models import ConversationTurn, SessionState
from utils.logger import AppLogger

log = AppLogger.get(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_metadata = MetaData()

sessions_table = Table(
    "conversation_sessions",
    _metadata,
    Column("session_id",     String,  primary_key=True),
    Column("created_at",     String,  nullable=False),
    Column("updated_at",     String,  nullable=False),
    Column("active_project", Text,    nullable=False, default=""),
    Column("current_topics", Text,    nullable=False, default="[]"),
    Column("metadata",       Text,    nullable=False, default="{}"),
)

turns_table = Table(
    "conversation_turns",
    _metadata,
    Column("id",             Integer, primary_key=True, autoincrement=True),
    Column("session_id",     String,  nullable=False),
    Column("seq",            Integer, nullable=False, default=0),
    Column("timestamp",      String,  nullable=False),
    Column("question",       Text,    nullable=False),
    Column("answer_summary", Text,    nullable=False, default=""),
    Column("topics",         Text,    nullable=False, default="[]"),
    Index("idx_turns_session_seq", "session_id", "seq"),
)


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """Low-level SQLite persistence for conversation sessions and turns.

    Args
    ----
    db_path : Absolute path to the SQLite file.
              Use ``:memory:`` for in-process testing.
    """

    def __init__(self, db_path: str) -> None:
        url = db_path if "://" in db_path else f"sqlite:///{db_path}"
        self._engine: Engine = create_engine(url, echo=False)
        self._enable_wal()
        self._ensure_schema()
        log.debug("MemoryStore ready — %s", db_path)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _enable_wal(self) -> None:
        @event.listens_for(self._engine, "connect")
        def _set_wal(conn, _record):
            conn.execute("PRAGMA journal_mode=WAL")

    def _ensure_schema(self) -> None:
        _metadata.create_all(self._engine, checkfirst=True)

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def get_or_create_session(
        self,
        session_id: str,
        active_project: str = "",
    ) -> SessionState:
        """Load an existing session or create a new one.

        Returns the current SessionState (without turns; call get_recent_turns
        separately when you need them).
        """
        with self._engine.begin() as conn:
            row = conn.execute(
                select(sessions_table).where(
                    sessions_table.c.session_id == session_id
                )
            ).fetchone()

            if row is None:
                now = _utcnow()
                conn.execute(
                    sessions_table.insert().values(
                        session_id=session_id,
                        created_at=now,
                        updated_at=now,
                        active_project=active_project,
                        current_topics="[]",
                        metadata="{}",
                    )
                )
                return SessionState(
                    session_id=session_id,
                    created_at=now,
                    updated_at=now,
                )

            return SessionState(
                session_id=row.session_id,
                active_project=row.active_project or "",
                current_topics=self._load_json(row.current_topics, []),
                created_at=row.created_at,
                updated_at=row.updated_at,
                metadata=self._load_json(row.metadata, {}),
            )

    def update_session(
        self,
        session_id:    str,
        active_project: str,
        current_topics: list[str],
        metadata:      dict[str, Any] | None = None,
    ) -> None:
        with self._engine.begin() as conn:
            values: dict = {
                "updated_at":     _utcnow(),
                "active_project": active_project,
                "current_topics": json.dumps(current_topics),
            }
            if metadata is not None:
                values["metadata"] = json.dumps(metadata)
            conn.execute(
                sessions_table.update()
                .where(sessions_table.c.session_id == session_id)
                .values(**values)
            )

    def list_sessions(self) -> list[dict]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(sessions_table).order_by(sessions_table.c.updated_at.desc())
            ).fetchall()
        return [
            {
                "session_id":     r.session_id,
                "active_project": r.active_project,
                "current_topics": self._load_json(r.current_topics, []),
                "created_at":     r.created_at,
                "updated_at":     r.updated_at,
            }
            for r in rows
        ]

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its turns."""
        with self._engine.begin() as conn:
            conn.execute(
                turns_table.delete().where(turns_table.c.session_id == session_id)
            )
            conn.execute(
                sessions_table.delete().where(sessions_table.c.session_id == session_id)
            )

    # ------------------------------------------------------------------
    # Turn CRUD
    # ------------------------------------------------------------------

    def save_turn(self, turn: ConversationTurn) -> ConversationTurn:
        """Persist a turn and return it with ``id`` and ``seq`` assigned."""
        with self._engine.begin() as conn:
            # Determine next seq for this session
            row = conn.execute(
                select(func.max(turns_table.c.seq)).where(
                    turns_table.c.session_id == turn.session_id
                )
            ).scalar()
            seq = (row or 0) + 1

            ts = turn.timestamp or _utcnow()
            result = conn.execute(
                turns_table.insert().values(
                    session_id=turn.session_id,
                    seq=seq,
                    timestamp=ts,
                    question=turn.question,
                    answer_summary=turn.answer_summary,
                    topics=json.dumps(turn.topics),
                )
            )
            turn.id        = result.inserted_primary_key[0]
            turn.seq       = seq
            turn.timestamp = ts
        return turn

    def get_recent_turns(
        self,
        session_id: str,
        limit:      int = 20,
    ) -> list[ConversationTurn]:
        """Return the most-recent *limit* turns, ordered oldest-first."""
        with self._engine.connect() as conn:
            # Fetch latest `limit` rows (ordered DESC), then reverse for
            # chronological display
            rows = conn.execute(
                select(turns_table)
                .where(turns_table.c.session_id == session_id)
                .order_by(turns_table.c.seq.desc())
                .limit(limit)
            ).fetchall()

        rows = list(reversed(rows))
        return [
            ConversationTurn(
                id=r.id,
                session_id=r.session_id,
                seq=r.seq,
                timestamp=r.timestamp,
                question=r.question,
                answer_summary=r.answer_summary,
                topics=self._load_json(r.topics, []),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(raw: str | None, default: Any) -> Any:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default
