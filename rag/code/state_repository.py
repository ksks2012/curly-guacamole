"""State repository for code orchestration run metadata.

This module persists:
- per-operation run-state payloads
- per-repo edge reindex commit state
"""

from __future__ import annotations

import json
from pathlib import Path


class CodeOrchestrationStateRepository:
    """Small filesystem-backed repository for orchestration state."""

    def __init__(self, code_rag_root: str) -> None:
        self._code_rag_root = Path(code_rag_root)

    def operation_state_path(self, operation_id: str) -> Path:
        return self._code_rag_root / "ops" / f"{operation_id}.json"

    def write_operation_state(self, operation_id: str, payload: dict) -> str:
        path = self.operation_state_path(operation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(path)

    def edge_reindex_state_path(self) -> Path:
        return self._code_rag_root / "ops" / "edge_reindex_state.json"

    def load_edge_reindex_state(self) -> dict[str, str]:
        path = self.edge_reindex_state_path()
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}

        state: dict[str, str] = {}
        for key, val in raw.items():
            if isinstance(key, str) and isinstance(val, str):
                state[key] = val
        return state

    def save_edge_reindex_state(self, state: dict[str, str]) -> None:
        path = self.edge_reindex_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(state, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
