"""
HierarchicalCodeRetriever — repo-aware routing for code retrieval.

Performs a two-stage retrieval flow:
  1) Repository routing via RepoIndex (coarse level)
  2) Code retrieval via CodeRetriever with repo_id filter passthrough

If explicit ``repo_ids`` are supplied, routing is skipped and those IDs are
passed directly to CodeRetriever as a Chroma where-filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.retrieval.base import RetrievalResult
from utils.logger import AppLogger

if TYPE_CHECKING:
    from rag.code.repo_index import RepoIndex
    from rag.retrieval.code_retriever import CodeRetriever

log = AppLogger.get(__name__)


def _repo_filter(repo_ids: list[str]) -> dict:
    """Build Chroma repo_id filter for one or many repository IDs."""
    if len(repo_ids) == 1:
        return {"repo_id": {"$eq": repo_ids[0]}}
    return {"repo_id": {"$in": repo_ids}}


def _merge_filters(base: dict | None, extra: dict | None) -> dict | None:
    """Merge two Chroma where-filters using $and semantics.

    Returns
    -------
    - base when extra is None
    - extra when base is None
    - {"$and": [...]} when both are present
    """
    if not extra:
        return base
    if not base:
        return extra

    if "$and" in base and isinstance(base["$and"], list):
        return {"$and": [*base["$and"], extra]}
    return {"$and": [base, extra]}


def _normalize_repo_ids(repo_ids: list[str] | None) -> list[str]:
    """Normalize and deduplicate repo IDs while preserving order."""
    if not repo_ids:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for rid in repo_ids:
        rid2 = str(rid).strip()
        if not rid2 or rid2 in seen:
            continue
        seen.add(rid2)
        cleaned.append(rid2)
    return cleaned


class HierarchicalCodeRetriever:
    """Repo-aware wrapper around CodeRetriever.

    Parameters
    ----------
    repo_index     : RepoIndex used for coarse repository routing.
    code_retriever : CodeRetriever used for symbol/file/block retrieval.
    repo_top_k     : Number of candidate repositories to route to when
                     ``repo_ids`` is not explicitly supplied.
    """

    def __init__(
        self,
        repo_index: "RepoIndex",
        code_retriever: "CodeRetriever",
        *,
        repo_top_k: int = 3,
    ) -> None:
        self._repo_index = repo_index
        self._code_retriever = code_retriever
        self._repo_top_k = max(1, int(repo_top_k))

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        *,
        repo_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Search code with optional repository-level routing.

        Routing behavior
        ----------------
        - If ``repo_ids`` is provided, those IDs are passed through directly
          to CodeRetriever as a metadata filter and RepoIndex is not queried.
        - Otherwise, RepoIndex is queried first to infer candidate repositories,
          then CodeRetriever is called with the inferred repo_id filter.
        - If no repositories are inferred, fallback to CodeRetriever with the
          original ``filters`` unchanged.
        """
        explicit_repo_ids = _normalize_repo_ids(repo_ids)

        if explicit_repo_ids:
            merged = _merge_filters(filters, _repo_filter(explicit_repo_ids))
            return self._code_retriever.search(query, top_k=top_k, filters=merged)

        routed_docs = self._repo_index.search(query, top_k=self._repo_top_k)
        routed_repo_ids = _normalize_repo_ids([
            str(doc.metadata.get("repo_id", "")) for doc in routed_docs
        ])

        if not routed_repo_ids:
            log.debug(
                "HierarchicalCodeRetriever: no routed repos, fallback to unfiltered code search"
            )
            return self._code_retriever.search(query, top_k=top_k, filters=filters)

        merged = _merge_filters(filters, _repo_filter(routed_repo_ids))
        return self._code_retriever.search(query, top_k=top_k, filters=merged)
