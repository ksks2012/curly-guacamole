"""
Tests for GCR2.3 — Diff Semantic Analysis.

Covers:
  SymbolEvolution schema
  - change_summary field defaults to ""
  - from_dict backward compat (missing change_summary key)
  - to_dict / from_dict roundtrip with change_summary

  DiffAnalyzer
  - empty diff returns ""
  - whitespace-only diff returns ""
  - mock LLM response used as summary
  - multi-line LLM response → only first line kept
  - quoted response → quotes stripped
  - LLM failure returns "" without raising
  - diff truncated to max_diff_chars

  GitReader.diff_commit_file
  - returns non-empty diff for a modified file
  - initial commit falls back to full file content
  - file not in commit returns ""

  GraphStore — change_summary
  - upsert stores change_summary
  - get_evolution returns correct change_summary
  - DO UPDATE replaces change_summary on conflict
  - backward compat: missing change_summary in row → ""
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

PASS: list[str] = []
FAIL: list[str] = []


def ok(name: str) -> None:
    print(f"{name}: OK")
    PASS.append(name)


def fail(name: str, exc: Exception) -> None:
    import traceback
    print(f"{name}: FAIL — {exc}")
    traceback.print_exc()
    FAIL.append(name)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REPO_ID   = "test-repo"
FILE_PATH = "rag/engine.py"
CA = "a" * 40
CB = "b" * 40


def _make_evolution(**kwargs):
    from rag.code.schema import SymbolEvolution, _evolution_id
    defaults = dict(
        evolution_id=_evolution_id(REPO_ID, FILE_PATH, "foo"),
        symbol_name="foo",
        repo_id=REPO_ID,
        file_path=FILE_PATH,
        introduced_in=CA,
    )
    defaults.update(kwargs)
    return SymbolEvolution(**defaults)


# ---------------------------------------------------------------------------
# SymbolEvolution — change_summary field
# ---------------------------------------------------------------------------

def test_change_summary_defaults_empty():
    evo = _make_evolution()
    assert evo.change_summary == ""
    ok("test_change_summary_defaults_empty")


def test_change_summary_from_dict_backward_compat():
    from rag.code.schema import SymbolEvolution, _evolution_id
    d = {
        "evolution_id": _evolution_id(REPO_ID, FILE_PATH, "foo"),
        "symbol_name": "foo",
        "repo_id": REPO_ID,
        "file_path": FILE_PATH,
        "introduced_in": CA,
        "modified_in": [],
        "deleted_in": "",
        "renamed_from": [],
        # deliberately no change_summary key
    }
    evo = SymbolEvolution.from_dict(d)
    assert evo.change_summary == ""
    ok("test_change_summary_from_dict_backward_compat")


def test_change_summary_roundtrip():
    evo = _make_evolution(change_summary="added retry logic on timeout")
    from rag.code.schema import SymbolEvolution
    evo2 = SymbolEvolution.from_dict(evo.to_dict())
    assert evo2.change_summary == "added retry logic on timeout"
    ok("test_change_summary_roundtrip")


# ---------------------------------------------------------------------------
# DiffAnalyzer
# ---------------------------------------------------------------------------

class _MockLLM:
    """Minimal LangChain-compatible mock that returns a fixed response."""

    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, _prompt: str):
        return type("Msg", (), {"content": self._response})()


class _FailingLLM:
    def invoke(self, _prompt: str):
        raise RuntimeError("LLM unavailable")


SAMPLE_DIFF = """\
--- a/rag/engine.py
+++ b/rag/engine.py
@@ -10,7 +10,9 @@ class RAGEngine:
-    def search(self, query: str) -> list:
-        return self._retriever.get(query)
+    def search(self, query: str, top_k: int = 5) -> list:
+        results = self._retriever.get(query, top_k=top_k)
+        return results
"""


def test_diff_analyzer_empty_diff_returns_empty():
    from rag.code.diff_analyzer import DiffAnalyzer
    da = DiffAnalyzer(_MockLLM("irrelevant"))
    assert da.summarize("foo", "") == ""
    ok("test_diff_analyzer_empty_diff_returns_empty")


def test_diff_analyzer_whitespace_diff_returns_empty():
    from rag.code.diff_analyzer import DiffAnalyzer
    da = DiffAnalyzer(_MockLLM("irrelevant"))
    assert da.summarize("foo", "   \n\t\n") == ""
    ok("test_diff_analyzer_whitespace_diff_returns_empty")


def test_diff_analyzer_returns_mock_response():
    from rag.code.diff_analyzer import DiffAnalyzer
    da = DiffAnalyzer(_MockLLM("added top_k parameter to search method"))
    result = da.summarize("RAGEngine.search", SAMPLE_DIFF, FILE_PATH)
    assert result == "added top_k parameter to search method"
    ok("test_diff_analyzer_returns_mock_response")


def test_diff_analyzer_multiline_response_first_line_only():
    from rag.code.diff_analyzer import DiffAnalyzer
    da = DiffAnalyzer(_MockLLM("added top_k parameter\n\nsome extra explanation"))
    result = da.summarize("RAGEngine.search", SAMPLE_DIFF)
    assert result == "added top_k parameter"
    ok("test_diff_analyzer_multiline_response_first_line_only")


def test_diff_analyzer_strips_surrounding_quotes():
    from rag.code.diff_analyzer import DiffAnalyzer
    da = DiffAnalyzer(_MockLLM('"added top_k parameter to search"'))
    result = da.summarize("RAGEngine.search", SAMPLE_DIFF)
    assert result == "added top_k parameter to search"
    ok("test_diff_analyzer_strips_surrounding_quotes")


def test_diff_analyzer_llm_failure_returns_empty():
    from rag.code.diff_analyzer import DiffAnalyzer
    da = DiffAnalyzer(_FailingLLM())
    result = da.summarize("foo", SAMPLE_DIFF)
    assert result == ""
    ok("test_diff_analyzer_llm_failure_returns_empty")


def test_diff_analyzer_truncates_diff():
    from rag.code.diff_analyzer import DiffAnalyzer

    captured: list[str] = []

    class CaptureLLM:
        def invoke(self, prompt: str):
            captured.append(prompt)
            return type("Msg", (), {"content": "truncated diff processed"})()

    big_diff = "+" + "x" * 10_000
    da = DiffAnalyzer(CaptureLLM(), max_diff_chars=500)
    da.summarize("big_func", big_diff)
    assert len(captured) == 1
    # The prompt should contain at most 500 chars of diff content
    assert big_diff[500:] not in captured[0]
    ok("test_diff_analyzer_truncates_diff")


def test_diff_analyzer_no_file_path_uses_unknown():
    from rag.code.diff_analyzer import DiffAnalyzer

    captured: list[str] = []

    class CaptureLLM:
        def invoke(self, prompt: str):
            captured.append(prompt)
            return type("Msg", (), {"content": "ok"})()

    da = DiffAnalyzer(CaptureLLM())
    da.summarize("foo", SAMPLE_DIFF)  # no file_path arg
    assert "<unknown>" in captured[0]
    ok("test_diff_analyzer_no_file_path_uses_unknown")


# ---------------------------------------------------------------------------
# GitReader.diff_commit_file
# ---------------------------------------------------------------------------

def _make_git_repo() -> Path:
    """Create a temporary git repository with two commits."""
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["git", "init", "-b", "main", str(tmp)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp), check=True, capture_output=True)

    src = tmp / "engine.py"
    src.write_text("def search(q):\n    return []\n")
    subprocess.run(["git", "add", "engine.py"], cwd=str(tmp), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp), check=True, capture_output=True)

    src.write_text("def search(q, top_k=5):\n    return [][:top_k]\n")
    subprocess.run(["git", "add", "engine.py"], cwd=str(tmp), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add top_k"], cwd=str(tmp), check=True, capture_output=True)

    return tmp


def _get_commits(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--format=%H", "--reverse"],
        cwd=str(repo), capture_output=True, text=True,
    )
    return [h.strip() for h in result.stdout.splitlines() if h.strip()]


def test_diff_commit_file_returns_diff():
    from rag.code.git_reader import GitReader
    repo = _make_git_repo()
    commits = _get_commits(repo)
    reader = GitReader(repo)
    diff = reader.diff_commit_file(commits[1], "engine.py")
    assert "top_k" in diff
    ok("test_diff_commit_file_returns_diff")


def test_diff_commit_file_initial_commit_fallback():
    from rag.code.git_reader import GitReader
    repo = _make_git_repo()
    commits = _get_commits(repo)
    reader = GitReader(repo)
    # First commit has no parent — should not return empty
    diff = reader.diff_commit_file(commits[0], "engine.py")
    assert diff.strip() != ""
    assert "search" in diff
    ok("test_diff_commit_file_initial_commit_fallback")


def test_diff_commit_file_untouched_file_returns_empty():
    from rag.code.git_reader import GitReader
    repo = _make_git_repo()
    commits = _get_commits(repo)
    reader = GitReader(repo)
    # "other.py" was never added to the repo
    diff = reader.diff_commit_file(commits[1], "other.py")
    assert diff == ""
    ok("test_diff_commit_file_untouched_file_returns_empty")


# ---------------------------------------------------------------------------
# GraphStore — change_summary
# ---------------------------------------------------------------------------

def _make_store():
    from rag.code.graph_store import GraphStore
    return GraphStore(f"{tempfile.mkdtemp()}/graph.db")


def test_graph_store_upsert_stores_change_summary():
    store = _make_store()
    evo = _make_evolution(change_summary="method signature extended with top_k")
    store.upsert_evolutions([evo])
    result = store.get_evolution(REPO_ID, FILE_PATH, "foo")
    assert result is not None
    assert result.change_summary == "method signature extended with top_k"
    ok("test_graph_store_upsert_stores_change_summary")


def test_graph_store_upsert_updates_change_summary():
    store = _make_store()
    evo = _make_evolution(change_summary="initial summary")
    store.upsert_evolutions([evo])

    updated = _make_evolution(change_summary="revised summary after rewrite")
    store.upsert_evolutions([updated])

    result = store.get_evolution(REPO_ID, FILE_PATH, "foo")
    assert result.change_summary == "revised summary after rewrite"
    ok("test_graph_store_upsert_updates_change_summary")


def test_graph_store_empty_change_summary_stored():
    store = _make_store()
    evo = _make_evolution()  # change_summary=""
    store.upsert_evolutions([evo])
    result = store.get_evolution(REPO_ID, FILE_PATH, "foo")
    assert result.change_summary == ""
    ok("test_graph_store_empty_change_summary_stored")


def test_graph_store_get_evolutions_includes_change_summary():
    store = _make_store()
    store.upsert_evolutions([
        _make_evolution(change_summary="foo changed"),
    ])
    evos = store.get_evolutions(repo_id=REPO_ID)
    assert evos[0].change_summary == "foo changed"
    ok("test_graph_store_get_evolutions_includes_change_summary")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_change_summary_defaults_empty,
    test_change_summary_from_dict_backward_compat,
    test_change_summary_roundtrip,
    test_diff_analyzer_empty_diff_returns_empty,
    test_diff_analyzer_whitespace_diff_returns_empty,
    test_diff_analyzer_returns_mock_response,
    test_diff_analyzer_multiline_response_first_line_only,
    test_diff_analyzer_strips_surrounding_quotes,
    test_diff_analyzer_llm_failure_returns_empty,
    test_diff_analyzer_truncates_diff,
    test_diff_analyzer_no_file_path_uses_unknown,
    test_diff_commit_file_returns_diff,
    test_diff_commit_file_initial_commit_fallback,
    test_diff_commit_file_untouched_file_returns_empty,
    test_graph_store_upsert_stores_change_summary,
    test_graph_store_upsert_updates_change_summary,
    test_graph_store_empty_change_summary_stored,
    test_graph_store_get_evolutions_includes_change_summary,
]

if __name__ == "__main__":
    for t in TESTS:
        try:
            t()
        except Exception as e:
            fail(t.__name__, e)

    print()
    if FAIL:
        print(f"FAIL  ({len(FAIL)} failed, {len(PASS)} passed)")
        for name in FAIL:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("PASS")
