"""
GCR2.1 — Dependency Graph Storage.

GraphStore is a SQLite-backed store for dependency edges extracted from
source code by ``PythonASTParser.parse_edges()``.

Schema
------
    edges — one row per unique directed dependency edge, keyed by edge_id
            (deterministic sha256 of src + type + dst + file).

Upserts are idempotent: inserting the same edge twice silently ignores
the duplicate.  Call ``delete_repo_edges()`` or ``delete_file_edges()``
before re-indexing to remove stale edges.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import (
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    event,
    func,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from rag.code.schema import DependencyEdge
from utils.logger import AppLogger

log = AppLogger.get(__name__)

# ---------------------------------------------------------------------------
# Table definition
# ---------------------------------------------------------------------------

_meta = MetaData()

_edges = Table(
    "edges", _meta,
    Column("edge_id",   String,  primary_key=True),
    Column("src_id",    String,  nullable=False),
    Column("dst_id",    String,  nullable=False),
    Column("edge_type", String,  nullable=False),
    Column("repo_id",   String,  nullable=False),
    Column("file_path", String,  nullable=False),
    Column("line_no",   Integer, nullable=False),
    Index("idx_edges_src",  "src_id"),
    Index("idx_edges_dst",  "dst_id"),
    Index("idx_edges_repo", "repo_id"),
    Index("idx_edges_type", "edge_type"),
)


def _make_engine(db_path: str) -> Engine:
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """SQLite-backed store for code dependency edges.

    Usage
    -----
        store  = GraphStore("./my_db/graph.db")
        parser = PythonASTParser()
        edges  = parser.parse_edges(source, "rag/engine.py", "my-repo")
        store.upsert_edges(edges)

        imports = store.get_edges(edge_type="IMPORTS", repo_id="my-repo")
        extends = store.get_edges(edge_type="EXTENDS", repo_id="my-repo")
    """

    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = _make_engine(str(path))
        _meta.create_all(self._engine)
        log.info("GraphStore ready at %s", path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_edges(self, edges: list[DependencyEdge]) -> int:
        """Insert edges, silently ignoring duplicates (same edge_id).

        Returns the number of newly inserted rows.
        """
        if not edges:
            return 0
        rows = [e.to_dict() for e in edges]
        stmt = sqlite_insert(_edges).on_conflict_do_nothing(
            index_elements=["edge_id"]
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt, rows)
        inserted = result.rowcount
        log.debug("upsert_edges: %d new / %d submitted", inserted, len(edges))
        return inserted

    def delete_repo_edges(self, repo_id: str) -> int:
        """Delete all edges for *repo_id*.  Returns the count deleted."""
        stmt = delete(_edges).where(_edges.c.repo_id == repo_id)
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
        count = result.rowcount
        log.debug("delete_repo_edges: %d removed for repo %s", count, repo_id)
        return count

    def delete_file_edges(self, repo_id: str, file_path: str) -> int:
        """Delete all edges whose source file is *file_path* in *repo_id*."""
        stmt = delete(_edges).where(
            (_edges.c.repo_id == repo_id) & (_edges.c.file_path == file_path)
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
        return result.rowcount

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_edges(
        self,
        *,
        src_id:    str | None = None,
        dst_id:    str | None = None,
        edge_type: str | None = None,
        repo_id:   str | None = None,
        file_path: str | None = None,
    ) -> list[DependencyEdge]:
        """Return edges matching all provided filters (AND semantics).

        Omit a parameter to skip that filter.
        """
        stmt = select(_edges)
        if src_id    is not None:
            stmt = stmt.where(_edges.c.src_id    == src_id)
        if dst_id    is not None:
            stmt = stmt.where(_edges.c.dst_id    == dst_id)
        if edge_type is not None:
            stmt = stmt.where(_edges.c.edge_type == edge_type)
        if repo_id   is not None:
            stmt = stmt.where(_edges.c.repo_id   == repo_id)
        if file_path is not None:
            stmt = stmt.where(_edges.c.file_path == file_path)
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [DependencyEdge(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._engine.connect() as conn:
            total = conn.execute(
                select(func.count()).select_from(_edges)
            ).scalar()
        return {"edges": total}
