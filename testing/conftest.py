"""Shared pytest configuration for the entire testing/ tree.

--integration flag
------------------
Pass ``pytest --integration`` to include tests that depend on a live server,
the Notion API, the local git repository, or any other external resource.
Without the flag, all tests marked ``@pytest.mark.integration`` are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the workspace root importable from every test file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# CLI option + marker
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (require live server, Notion API, git repo, etc.).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: depends on a live server, Notion API, git repo, or network "
        "(skipped by default; pass --integration to run).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if not config.getoption("--integration"):
        skip = pytest.mark.skip(reason="pass --integration to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
