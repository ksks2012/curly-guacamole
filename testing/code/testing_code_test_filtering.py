"""Tests for hard filtering of test-oriented code paths in code search."""

from __future__ import annotations

from rag.client import LocalLlamaClient


def test_is_test_code_metadata_accepts_testing_directory() -> None:
    assert LocalLlamaClient._is_test_code_metadata({"file_path": "testing/foo/bar.py"}) is True


def test_is_test_code_metadata_accepts_testing_prefix_filename() -> None:
    assert LocalLlamaClient._is_test_code_metadata({"file_path": "src/testing_utils/testing_parser.py"}) is True


def test_is_test_code_metadata_accepts_existing_test_patterns() -> None:
    assert LocalLlamaClient._is_test_code_metadata({"file_path": "tests/unit/test_service.py"}) is True
    assert LocalLlamaClient._is_test_code_metadata({"file_path": "src/module/service_test.py"}) is True


def test_is_test_code_metadata_respects_explicit_flag() -> None:
    assert LocalLlamaClient._is_test_code_metadata({"file_path": "src/app/main.py", "is_test": True}) is True


def test_is_test_code_metadata_keeps_non_test_paths() -> None:
    assert LocalLlamaClient._is_test_code_metadata({"file_path": "src/app/service.py"}) is False
