"""
GCR2.3 — Diff Semantic Analysis.

DiffAnalyzer converts a unified git diff for a symbol into a one-line
semantic change summary using an LLM call.

Pipeline (caller-driven)
------------------------
    GitReader.diff_commit_file(commit, file_path)
        ↓
    DiffAnalyzer.summarize(symbol_name, diff_text, file_path)
        ↓
    SymbolEvolution.change_summary  →  GraphStore.upsert_evolutions()

Design
------
- DiffAnalyzer is stateless except for the injected LLM.
- The LLM is any object exposing ``.invoke(str) -> obj`` where ``obj.content``
  is the response text.  This is the standard LangChain BaseChatModel interface
  and is also trivially mockable in tests.
- Diff is truncated to ``max_diff_chars`` before the LLM call to bound token
  cost.  The default (3 000 chars) is enough for most function-level changes.
- Returns ``""`` on empty input or LLM error — callers may store ``""`` as a
  sentinel meaning "not yet analysed".
"""

from __future__ import annotations

from utils.logger import AppLogger

log = AppLogger.get(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DIFF_CHARS: int = 3_000

_PROMPT_TEMPLATE = (
    "You are a code-change analyst.\n"
    "The symbol \"{symbol_name}\" in \"{file_path}\" was modified.\n"
    "Review the unified diff below and write exactly one concise sentence "
    "(at most 15 words) describing the semantic change — what behaviour or "
    "structure changed, not how many lines changed.\n\n"
    "Unified diff (may be truncated):\n"
    "{diff_text}\n\n"
    "One-line summary:"
)


# ---------------------------------------------------------------------------
# DiffAnalyzer
# ---------------------------------------------------------------------------

class DiffAnalyzer:
    """Produce one-line semantic summaries of symbol-level code changes.

    Parameters
    ----------
    llm : Any object exposing ``.invoke(str) -> object`` where the returned
          object has a ``.content`` attribute containing the model's reply.
          Compatible with LangChain ``BaseChatModel``.
    max_diff_chars : Hard cap on diff characters sent to the LLM.
    """

    def __init__(self, llm, *, max_diff_chars: int = _MAX_DIFF_CHARS) -> None:
        self._llm = llm
        self._max_diff_chars = max_diff_chars

    def summarize(
        self,
        symbol_name: str,
        diff_text: str,
        file_path: str = "",
    ) -> str:
        """Return a one-line semantic summary of *diff_text* for *symbol_name*.

        Parameters
        ----------
        symbol_name : Fully-qualified symbol name (e.g. ``"MyClass.my_method"``).
        diff_text   : Unified diff text (output of ``git diff``).
        file_path   : Source file path used for context in the prompt.

        Returns
        -------
        A single-sentence string, or ``""`` when *diff_text* is empty or the
        LLM call fails.
        """
        if not diff_text or not diff_text.strip():
            return ""

        truncated = diff_text[: self._max_diff_chars]
        prompt = _PROMPT_TEMPLATE.format(
            symbol_name=symbol_name,
            file_path=file_path or "<unknown>",
            diff_text=truncated,
        )

        try:
            response = self._llm.invoke(prompt)
            raw = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()
            # Normalise: use only the first line and strip surrounding quotes.
            summary = raw.splitlines()[0].strip().strip('"').strip("'") if raw else ""
            log.debug("DiffAnalyzer: %s → %r", symbol_name, summary[:80])
            return summary
        except Exception as exc:
            log.warning("DiffAnalyzer.summarize failed for %s: %s", symbol_name, exc)
            return ""
