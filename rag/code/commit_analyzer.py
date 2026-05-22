"""
GCR2.4 — Commit Semantic Indexing: CommitAnalyzer.

CommitAnalyzer builds a CommitRecord from a CommitInfo and a list of
SymbolEvolution records.

Responsibilities
----------------
- Derive affected_symbols mechanically from SymbolEvolution.introduced_in /
  modified_in / deleted_in fields — no LLM required.
- Generate a one-sentence semantic summary via an LLM call, using the
  commit message and affected symbol list as context.
- Assemble and return a CommitRecord ready for CommitIndexer ingestion.

Design
------
- CommitAnalyzer is stateless except for the injected LLM.
- The LLM is any object exposing ``.invoke(str) -> obj`` where ``obj.content``
  is the reply text.  Compatible with LangChain BaseChatModel; mockable.
- Returns "" as summary on empty input or LLM error — callers store the
  commit message as fallback context in that case.
- ``derive_affected_symbols()`` is a standalone helper so it can be called
  without instantiating CommitAnalyzer (no LLM dependency).
"""

from __future__ import annotations

import hashlib

from utils.logger import AppLogger
from rag.code.schema import CommitInfo, CommitRecord, SymbolEvolution, _commit_record_id

log = AppLogger.get(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CONTEXT_CHARS: int = 3_000

_PROMPT_TEMPLATE = (
    "You are a code-change analyst.\n"
    "Commit {short_hash} by {author} on {date}.\n"
    "Commit message: {message}\n"
    "Affected symbols ({symbol_count}): {symbol_list}\n\n"
    "Write exactly one concise sentence (at most 20 words) describing the "
    "semantic intent of this commit — what behaviour or architecture changed, "
    "not how many files were modified.\n\n"
    "One-line summary:"
)


# ---------------------------------------------------------------------------
# Standalone helper — no LLM needed
# ---------------------------------------------------------------------------

def derive_affected_symbols(
    commit_hash: str,
    evolutions: list[SymbolEvolution],
) -> list[str]:
    """Return symbol names introduced, modified, or deleted in *commit_hash*.

    Pure function — performs no I/O.  Symbols are sorted and deduplicated.

    Parameters
    ----------
    commit_hash : The full commit hash to check against.
    evolutions  : SymbolEvolution records for the repository (or a file subset).
    """
    affected: set[str] = set()
    for evo in evolutions:
        if (
            evo.introduced_in == commit_hash
            or commit_hash in evo.modified_in
            or evo.deleted_in == commit_hash
        ):
            affected.add(evo.symbol_name)
    return sorted(affected)


# ---------------------------------------------------------------------------
# CommitAnalyzer
# ---------------------------------------------------------------------------

class CommitAnalyzer:
    """Build CommitRecord objects from CommitInfo + SymbolEvolution data.

    Parameters
    ----------
    llm : Any object exposing ``.invoke(str) -> object`` where the returned
          object has a ``.content`` attribute.  Pass None to skip LLM
          summarisation (summary will be "").
    max_context_chars : Hard cap on context characters sent to the LLM.
    """

    def __init__(self, llm=None, *, max_context_chars: int = _MAX_CONTEXT_CHARS) -> None:
        self._llm = llm
        self._max_context_chars = max_context_chars

    def build(
        self,
        commit_info: CommitInfo,
        evolutions: list[SymbolEvolution],
        repo_id: str,
    ) -> CommitRecord:
        """Build and return a CommitRecord.

        Parameters
        ----------
        commit_info : Git commit metadata (hash, author, date, message, files).
        evolutions  : All SymbolEvolution records for *repo_id*.  The method
                      filters to those affected by this commit internally.
        repo_id     : Logical repository identifier.
        """
        affected = derive_affected_symbols(commit_info.commit_hash, evolutions)
        summary  = self._summarize(commit_info, affected) if self._llm else ""
        fingerprint = summary if summary else commit_info.message
        content_hash = hashlib.sha256(fingerprint.encode()).hexdigest()

        return CommitRecord(
            commit_id=_commit_record_id(repo_id, commit_info.commit_hash),
            repo_id=repo_id,
            commit_hash=commit_info.commit_hash,
            author=commit_info.author,
            date=commit_info.date,
            message=commit_info.message,
            files_changed=commit_info.files_changed,
            affected_symbols=affected,
            summary=summary,
            content_hash=content_hash,
        )

    # ── Private ───────────────────────────────────────────────────────────

    def _summarize(
        self,
        commit_info: CommitInfo,
        affected_symbols: list[str],
    ) -> str:
        """Return a one-line LLM summary, or "" on failure."""
        symbol_list = ", ".join(affected_symbols[:30]) or "(none detected)"
        message_trunc = commit_info.message[: self._max_context_chars]
        prompt = _PROMPT_TEMPLATE.format(
            short_hash=commit_info.short_hash,
            author=commit_info.author,
            date=commit_info.date,
            message=message_trunc,
            symbol_count=len(affected_symbols),
            symbol_list=symbol_list,
        )
        try:
            response = self._llm.invoke(prompt)
            raw = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()
            summary = raw.splitlines()[0].strip().strip('"').strip("'") if raw else ""
            log.debug("CommitAnalyzer: %s → %r", commit_info.short_hash, summary[:80])
            return summary
        except Exception as exc:
            log.warning(
                "CommitAnalyzer._summarize failed for %s: %s",
                commit_info.short_hash, exc,
            )
            return ""
