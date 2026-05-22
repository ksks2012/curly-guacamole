"""
GCR2.1 — Dependency Graph Storage.
GCR2.2 — Symbol Evolution Storage.

GraphStore is a SQLite-backed store for:
  - dependency edges extracted from source code (GCR2.1)
  - symbol lifecycle records across git history (GCR2.2)

Schema
------
    edges             — one row per unique directed dependency edge.
    symbol_evolution  — one row per symbol per file, tracking when it was
                        introduced, modified, and deleted.

All upserts are idempotent.  Call delete helpers before re-indexing to
remove stale rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import (
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    event,
    func,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from rag.code.schema import DependencyEdge, SymbolEvolution
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

_symbol_evolution = Table(
    "symbol_evolution", _meta,
    Column("evolution_id",  String, primary_key=True),
    Column("symbol_name",   String, nullable=False),
    Column("repo_id",       String, nullable=False),
    Column("file_path",     String, nullable=False),
    Column("introduced_in", String, nullable=False, default=""),
    Column("modified_in",   Text,   nullable=False, default="[]"),  # JSON list
    Column("deleted_in",    String, nullable=False, default=""),
    Column("renamed_from",  Text,   nullable=False, default="[]"),  # JSON list
    Index("idx_evo_repo",    "repo_id"),
    Index("idx_evo_file",    "repo_id", "file_path"),
    Index("idx_evo_symbol",  "repo_id", "symbol_name"),
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
            total_edges = conn.execute(
                select(func.count()).select_from(_edges)
            ).scalar()
            total_evo = conn.execute(
                select(func.count()).select_from(_symbol_evolution)
            ).scalar()
        return {"edges": total_edges, "symbol_evolution": total_evo}

    # ------------------------------------------------------------------
    # Symbol evolution — write
    # ------------------------------------------------------------------

    def upsert_evolutions(self, evolutions: list[SymbolEvolution]) -> int:
        """Insert-or-replace symbol evolution records.

        The full record is replaced on conflict (caller owns the complete
        history; use delete helpers before re-building from scratch).

        Returns the number of rows inserted or replaced.
        """
        if not evolutions:
            return 0
        rows = [
            {
                "evolution_id":  e.evolution_id,
                "symbol_name":   e.symbol_name,
                "repo_id":       e.repo_id,
                "file_path":     e.file_path,
                "introduced_in": e.introduced_in,
                "modified_in":   json.dumps(e.modified_in),
                "deleted_in":    e.deleted_in,
                "renamed_from":  json.dumps(e.renamed_from),
            }
            for e in evolutions
        ]
        stmt = sqlite_insert(_symbol_evolution).on_conflict_do_update(
            index_elements=["evolution_id"],
            set_={
                c: sqlite_insert(_symbol_evolution).excluded[c]
                for c in ("introduced_in", "modified_in", "deleted_in", "renamed_from")
            },
        )
        with self._engine.begin() as conn:
            conn.execute(stmt, rows)
        log.debug("upsert_evolutions: %d records", len(evolutions))
        return len(evolutions)

    def delete_repo_evolutions(self, repo_id: str) -> int:
        """Delete all evolution records for *repo_id*."""
        stmt = delete(_symbol_evolution).where(
            _symbol_evolution.c.repo_id == repo_id
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
        return result.rowcount

    def delete_file_evolutions(self, repo_id: str, file_path: str) -> int:
        """Delete all evolution records for *file_path* in *repo_id*."""
        stmt = delete(_symbol_evolution).where(
            (_symbol_evolution.c.repo_id   == repo_id)
            & (_symbol_evolution.c.file_path == file_path)
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
        return result.rowcount

    # ------------------------------------------------------------------
    # Symbol evolution — read
    # ------------------------------------------------------------------

    def get_evolution(
        self, repo_id: str, file_path: str, symbol_name: str
    ) -> SymbolEvolution | None:
        """Return the evolution record for one specific symbol, or None."""
        stmt = select(_symbol_evolution).where(
            (_symbol_evolution.c.repo_id      == repo_id)
            & (_symbol_evolution.c.file_path  == file_path)
            & (_symbol_evolution.c.symbol_name == symbol_name)
        )
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return self._row_to_evolution(row) if row else None

    def get_evolutions(
        self,
        *,
        repo_id:   str | None = None,
        file_path: str | None = None,
        alive_only: bool = False,
    ) -> list[SymbolEvolution]:
        """Return evolution records matching all provided filters.

        Parameters
        ----------
        repo_id    : Filter by repository.
        file_path  : Filter by source file (use with repo_id).
        alive_only : When True, exclude symbols with a non-empty deleted_in.
        """
        stmt = select(_symbol_evolution)
        if repo_id    is not None:
            stmt = stmt.where(_symbol_evolution.c.repo_id   == repo_id)
        if file_path  is not None:
            stmt = stmt.where(_symbol_evolution.c.file_path == file_path)
        if alive_only:
            stmt = stmt.where(_symbol_evolution.c.deleted_in == "")
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [self._row_to_evolution(r) for r in rows]

    @staticmethod
    def _row_to_evolution(row) -> SymbolEvolution:
        return SymbolEvolution(
            evolution_id=row["evolution_id"],
            symbol_name=row["symbol_name"],
            repo_id=row["repo_id"],
            file_path=row["file_path"],
            introduced_in=row["introduced_in"] or "",
            modified_in=json.loads(row["modified_in"] or "[]"),
            deleted_in=row["deleted_in"] or "",
            renamed_from=json.loads(row["renamed_from"] or "[]"),
        )
