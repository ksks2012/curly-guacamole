"""Soft scope conventions for code search queries.

Supported query hints:
- repo:<repo-id>
- path:<path-fragment>
- module:<module-prefix>

These hints influence ranking only. They never become a hard filter.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import shlex

from langchain_core.documents import Document


@dataclass(frozen=True)
class CodeQueryScope:
    """Parsed soft-scope hints extracted from a code search query."""

    raw_query: str
    semantic_query: str
    repo_terms: tuple[str, ...]
    path_terms: tuple[str, ...]
    module_terms: tuple[str, ...]

    @property
    def has_hints(self) -> bool:
        """Return True when at least one soft-scope hint is present."""
        return bool(self.repo_terms or self.path_terms or self.module_terms)


def parse_code_query_scope(query: str) -> CodeQueryScope:
    """Parse a code query into semantic text plus soft-scope hints."""
    raw_query = str(query or "").strip()
    if not raw_query:
        return CodeQueryScope(
            raw_query="",
            semantic_query="",
            repo_terms=(),
            path_terms=(),
            module_terms=(),
        )

    try:
        tokens = shlex.split(raw_query)
    except ValueError:
        tokens = raw_query.split()

    repo_terms: list[str] = []
    path_terms: list[str] = []
    module_terms: list[str] = []
    semantic_tokens: list[str] = []

    for token in tokens:
        lower_token = token.lower()
        if lower_token.startswith("repo:") and len(token) > 5:
            repo_terms.append(token[5:].strip())
            continue
        if lower_token.startswith("path:") and len(token) > 5:
            path_terms.append(token[5:].strip())
            continue
        if lower_token.startswith("module:") and len(token) > 7:
            module_terms.append(token[7:].strip())
            continue
        semantic_tokens.append(token)

    semantic_query = " ".join(semantic_tokens).strip() or raw_query
    return CodeQueryScope(
        raw_query=raw_query,
        semantic_query=semantic_query,
        repo_terms=tuple(_dedupe_normalized(repo_terms)),
        path_terms=tuple(_dedupe_normalized(path_terms)),
        module_terms=tuple(_dedupe_normalized(module_terms)),
    )


def rerank_code_rows_by_scope(
    rows: list[tuple[Document, float]],
    query_scope: CodeQueryScope,
) -> list[tuple[Document, float]]:
    """Apply soft-scope score boosts to code rows and return stable sorted results."""
    if not rows or not query_scope.has_hints:
        return rows

    rescored: list[tuple[int, Document, float]] = []
    for index, (doc, score) in enumerate(rows):
        boost = _scope_match_boost(query_scope, doc)
        adjusted = round(float(score) + boost, 4)
        rescored.append((index, doc, adjusted))

    rescored.sort(key=lambda item: (-item[2], item[0]))
    return [(doc, score) for _, doc, score in rescored]


def _scope_match_boost(query_scope: CodeQueryScope, doc: Document) -> float:
    metadata = dict(doc.metadata or {})
    repo_id = str(metadata.get("repo_id", "") or "").strip().lower()
    file_path = str(metadata.get("file_path", "") or "").strip().lower().replace("\\", "/")
    module_path = _module_path(file_path)
    chunk_name = str(metadata.get("name", "") or "").strip().lower()

    boost = 0.0
    for repo_term in query_scope.repo_terms:
        if repo_term in repo_id:
            boost += 0.18

    for path_term in query_scope.path_terms:
        normalized_path_term = path_term.replace("\\", "/")
        if normalized_path_term in file_path:
            boost += 0.10

    for module_term in query_scope.module_terms:
        if module_term in module_path or module_term == chunk_name:
            boost += 0.12

    return min(boost, 0.45)


def _module_path(file_path: str) -> str:
    normalized = str(file_path or "").strip().replace("\\", "/")
    if not normalized:
        return ""
    root, _ext = os.path.splitext(normalized)
    return root.replace("/", ".").lower()


def _dedupe_normalized(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out
