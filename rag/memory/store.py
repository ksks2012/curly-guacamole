"""
SQLite persistence for Stage C — Conversation Memory, User Memory, Knowledge Timeline.

Schema
------
conversation_sessions      C.1 — session state
conversation_turns         C.1 — per-turn Q-A history
user_interests             C.2 — recency-weighted topic interest profile
timeline_entries           C.3 — daily knowledge activity log

Uses SQLAlchemy Core (same pattern as rag/knowledge/store.py).
WAL mode enabled for concurrent read safety.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
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

# C.2 — Semantic User Memory
user_interests_table = Table(
    "user_interests",
    _metadata,
    Column("topic",      String,  primary_key=True),
    Column("count",      Integer, nullable=False, default=1),
    Column("weight",     Float,   nullable=False, default=1.0),
    Column("first_seen", String,  nullable=False),
    Column("last_seen",  String,  nullable=False),
)

# C.3 — Knowledge Timeline  (one row per calendar day, aggregated across sessions)
timeline_table = Table(
    "timeline_entries",
    _metadata,
    Column("date",           String,  primary_key=True),   # YYYY-MM-DD
    Column("year",           Integer, nullable=False),
    Column("month",          Integer, nullable=False),
    Column("topics",         Text,    nullable=False, default="[]"),
    Column("doc_ids",        Text,    nullable=False, default="[]"),
    Column("question_count", Integer, nullable=False, default=1),
    Column("updated_at",     String,  nullable=False),
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

    # ------------------------------------------------------------------
    # C.2 — User Interests
    # ------------------------------------------------------------------

    def upsert_interest(self, topic: str) -> None:
        """Record one occurrence of *topic*, updating its recency-weighted score.

        Weight update rule: ``new_weight = old_weight * 0.9 + 1.0``

        This is an exponential moving average where each new occurrence adds
        1.0 to a score that decays by 10 % between updates.  Topics that
        appear frequently and recently will accumulate the highest scores.
        """
        now = _utcnow()
        with self._engine.begin() as conn:
            row = conn.execute(
                select(user_interests_table).where(
                    user_interests_table.c.topic == topic
                )
            ).fetchone()

            if row is None:
                conn.execute(
                    user_interests_table.insert().values(
                        topic=topic, count=1, weight=1.0,
                        first_seen=now, last_seen=now,
                    )
                )
            else:
                conn.execute(
                    user_interests_table.update()
                    .where(user_interests_table.c.topic == topic)
                    .values(
                        count=row.count + 1,
                        weight=round(row.weight * 0.9 + 1.0, 4),
                        last_seen=now,
                    )
                )

    def get_top_interests(self, n: int = 10) -> list[dict]:
        """Return top *n* interests ordered by weight descending."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(user_interests_table)
                .order_by(user_interests_table.c.weight.desc())
                .limit(n)
            ).fetchall()
        return [
            {
                "topic":      r.topic,
                "count":      r.count,
                "weight":     r.weight,
                "first_seen": r.first_seen,
                "last_seen":  r.last_seen,
            }
            for r in rows
        ]

    def clear_interests(self) -> None:
        """Delete all user interest records."""
        with self._engine.begin() as conn:
            conn.execute(user_interests_table.delete())

    # ------------------------------------------------------------------
    # C.3 — Knowledge Timeline
    # ------------------------------------------------------------------

    def upsert_timeline_entry(
        self,
        date:    str,
        topics:  list[str],
        doc_ids: list[str],
    ) -> None:
        """Upsert a daily timeline entry, merging topics and doc_ids.

        *date* must be a ``YYYY-MM-DD`` ISO date string.
        Calling this multiple times on the same date accumulates topics and
        increments the question count — it does NOT replace the existing row.
        """
        year  = int(date[:4])
        month = int(date[5:7])
        now   = _utcnow()

        with self._engine.begin() as conn:
            row = conn.execute(
                select(timeline_table).where(timeline_table.c.date == date)
            ).fetchone()

            if row is None:
                conn.execute(
                    timeline_table.insert().values(
                        date=date, year=year, month=month,
                        topics=json.dumps(topics),
                        doc_ids=json.dumps(doc_ids),
                        question_count=1,
                        updated_at=now,
                    )
                )
            else:
                # Merge: preserve order, deduplicate
                merged_topics = list(
                    dict.fromkeys(self._load_json(row.topics, []) + topics)
                )
                merged_docs = list(
                    dict.fromkeys(self._load_json(row.doc_ids, []) + doc_ids)
                )
                conn.execute(
                    timeline_table.update()
                    .where(timeline_table.c.date == date)
                    .values(
                        topics=json.dumps(merged_topics),
                        doc_ids=json.dumps(merged_docs),
                        question_count=row.question_count + 1,
                        updated_at=now,
                    )
                )

    def get_timeline(self, start_date: str, end_date: str) -> list[dict]:
        """Return daily entries in the closed interval [*start_date*, *end_date*].

        Dates are ``YYYY-MM-DD`` strings; ISO lexicographic order is used for
        comparison (safe for SQLite TEXT columns).
        Results are ordered by date ascending.
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(timeline_table)
                .where(
                    and_(
                        timeline_table.c.date >= start_date,
                        timeline_table.c.date <= end_date,
                    )
                )
                .order_by(timeline_table.c.date.asc())
            ).fetchall()
        return [
            {
                "date":           r.date,
                "year":           r.year,
                "month":          r.month,
                "topics":         self._load_json(r.topics, []),
                "doc_ids":        self._load_json(r.doc_ids, []),
                "question_count": r.question_count,
            }
            for r in rows
        ]

    def get_timeline_by_year(self, year: int) -> list[dict]:
        """Return all entries for the given *year*, ordered by date."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(timeline_table)
                .where(timeline_table.c.year == year)
                .order_by(timeline_table.c.date.asc())
            ).fetchall()
        return [
            {
                "date":           r.date,
                "year":           r.year,
                "month":          r.month,
                "topics":         self._load_json(r.topics, []),
                "doc_ids":        self._load_json(r.doc_ids, []),
                "question_count": r.question_count,
            }
            for r in rows
        ]
