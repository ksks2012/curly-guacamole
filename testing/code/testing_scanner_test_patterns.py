"""Tests for scanner test-path heuristics."""

from __future__ import annotations

from rag.code.path_rules import is_test_path
from rag.code.scanner import _is_test


def test_scanner_marks_testing_directory_as_test() -> None:
    assert _is_test("testing/ui/testing_code_graph_ui_smoke.py") is True


def test_scanner_marks_testing_prefix_filename_as_test() -> None:
    assert _is_test("src/testing_utils/testing_parser.py") is True


def test_scanner_keeps_normal_source_file_non_test() -> None:
    assert _is_test("src/app/service.py") is False


def test_shared_path_rule_matches_scanner_behavior() -> None:
    paths = [
        "testing/ui/testing_code_graph_ui_smoke.py",
        "src/testing_utils/testing_parser.py",
        "src/app/service.py",
    ]
    for path in paths:
        assert is_test_path(path) == _is_test(path)
