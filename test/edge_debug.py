"""Debug utility for printing dependency edges stored in GraphStore.

This script is intentionally lightweight and lives under ``test/`` so it can
be run locally without affecting the main application entry points.
"""

from __future__ import annotations

import argparse
import json

from rag.code.graph_store import GraphStore
from utils.config import AppConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print GraphStore edges for a repo.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--repo-id", required=True, help="Logical repo id")
    parser.add_argument("--edge-type", default=None, help="Optional edge type filter")
    parser.add_argument("--file-path", default=None, help="Optional source file filter")
    parser.add_argument("--src-id", default=None, help="Optional source node filter")
    parser.add_argument("--dst-id", default=None, help="Optional destination node filter")
    parser.add_argument("--limit", type=int, default=500, help="Maximum rows to print")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = AppConfig(path=args.config) if args.config else AppConfig()
    store = GraphStore(config.graph_db_path)

    rows = store.get_edges(
        repo_id=args.repo_id,
        edge_type=args.edge_type,
        file_path=args.file_path,
        src_id=args.src_id,
        dst_id=args.dst_id,
    )[: max(0, int(args.limit))]

    if args.json:
        payload = {
            "repo_id": args.repo_id,
            "count": len(rows),
            "edges": [r.to_dict() for r in rows],
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0

    print(f"repo_id={args.repo_id} count={len(rows)} graph_db={config.graph_db_path}")
    for row in rows:
        print(
            f"{row.edge_id}\t{row.edge_type}\t{row.src_id}\t{row.dst_id}\t"
            f"{row.file_path}:{row.line_no}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())