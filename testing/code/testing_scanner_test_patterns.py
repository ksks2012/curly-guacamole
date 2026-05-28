"""Tests for scanner test-path heuristics."""

from __future__ import annotations

from rag.code.scanner import _is_test


def test_scanner_marks_testing_directory_as_test() -> None:
    assert _is_test("testing/ui/testing_code_graph_ui_smoke.py") is True


def test_scanner_marks_testing_prefix_filename_as_test() -> None:
    assert _is_test("src/testing_utils/testing_parser.py") is True


def test_scanner_keeps_normal_source_file_non_test() -> None:
    assert _is_test("src/app/service.py") is False
