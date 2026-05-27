"""Debug utility for printing dependency edges stored in GraphStore.

This script is intentionally lightweight and lives under ``debug_tools/`` so it
can be run locally without affecting the main application entry points.
"""

from __future__ import annotations

import argparse
import json

from langchain_openai import OpenAIEmbeddings

from rag.code.graph_store import GraphStore
from rag.code.indexer import CodeIndexer
from rag.embeddings import OpenRouterEmbeddings
from rag.retrieval.code_retriever import CodeRetriever
from rag.retrieval.related_code_retriever import RelatedCodeRetriever
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
    parser.add_argument("--code-query", default=None, help="Optional code graph search query")
    parser.add_argument(
        "--code-level",
        choices=["file", "symbol", "block"],
        default="symbol",
        help="Code collection level used for code graph search",
    )
    parser.add_argument("--code-top-k", type=int, default=5, help="Maximum code search results")
    parser.add_argument(
        "--code-max-related",
        type=int,
        default=5,
        help="Maximum GraphStore relations to show per code result",
    )
    parser.add_argument(
        "--code-max-nearby",
        type=int,
        default=2,
        help="Maximum nearby same-file relations to show per code result",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "summary"],
        default="text",
        help="Output format",
    )
    return parser.parse_args()


def _build_embeddings(config: AppConfig):
    if config.model_provider == "openrouter":
        return OpenRouterEmbeddings(
            model=config.embed_model,
            api_key=config.embed_api_key,
            base_url=config.embed_base,
            requests_per_minute=config.requests_rate_limit,
        )
    return OpenAIEmbeddings(
        openai_api_key=config.embed_api_key,
        openai_api_base=config.embed_base,
        model=config.embed_model,
    )


def _load_code_search_results(
    *,
    query: str,
    repo_id: str,
    level: str,
    top_k: int,
    max_related: int,
    max_nearby: int,
) -> list[object]:
    config = AppConfig()
    embeddings = _build_embeddings(config)
    indexer = CodeIndexer(config.code_rag_root, embeddings)
    base_retriever = CodeRetriever(indexer, level=level)
    retriever = RelatedCodeRetriever(
        base_retriever,
        GraphStore(config.graph_db_path),
        max_related=max_related,
        max_nearby=max_nearby,
    )
    filters = {"repo_id": {"$eq": repo_id}} if repo_id else None
    return retriever.search(query, top_k=top_k, filters=filters)


def _format_code_search_text(*, query: str, repo_id: str, level: str, rows: list[object]) -> str:
    lines = [f"query={query} repo_id={repo_id} level={level} count={len(rows)}"]
    for index, row in enumerate(rows, start=1):
        meta = dict(getattr(row, "metadata", {}) or {})
        chunk_id = str(meta.get("chunk_id", "")).strip()
        file_path = str(meta.get("file_path", "")).strip()
        chunk_type = str(meta.get("chunk_type", "")).strip()
        name = str(meta.get("name", "")).strip()
        start_line = int(meta.get("start_line", 0) or 0)
        end_line = int(meta.get("end_line", 0) or 0)
        related = list(meta.get("related_blocks", []) or [])

        if start_line > 0 and end_line >= start_line:
            span = f"{start_line}-{end_line}"
        elif start_line > 0:
            span = str(start_line)
        else:
            span = "?"

        lines.append(
            f"{index}\t{float(getattr(row, 'score', 0.0)):.4f}\t{chunk_id}\t{chunk_type}\t"
            f"{name}\t{file_path}:{span}"
        )
        for rel in related:
            rel_target = str(rel.get("target_id", "")).strip() or "?"
            rel_edge = str(rel.get("edge_type", "")).strip() or "?"
            rel_strategy = str(rel.get("mapping_strategy", "")).strip() or "?"
            rel_anchor = str(rel.get("source_anchor", "")).strip() or "unknown"
            rel_direction = str(rel.get("direction", "")).strip() or "?"
            lines.append(
                f"  related\t{rel_edge}\t{rel_strategy}\t{rel_anchor}\t{rel_direction}\t{rel_target}"
            )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    config = AppConfig(path=args.config) if args.config else AppConfig()

    code_query = str(getattr(args, "code_query", "") or "").strip()
    if code_query:
        rows = _load_code_search_results(
            query=code_query,
            repo_id=args.repo_id,
            level=args.code_level,
            top_k=args.code_top_k,
            max_related=args.code_max_related,
            max_nearby=args.code_max_nearby,
        )

        if args.output == "json":
            payload = {
                "query": code_query,
                "repo_id": args.repo_id,
                "level": args.code_level,
                "count": len(rows),
                "results": [
                    {
                        "score": float(getattr(row, "score", 0.0)),
                        "content": getattr(row, "content", ""),
                        "metadata": dict(getattr(row, "metadata", {}) or {}),
                    }
                    for row in rows
                ],
            }
            print(json.dumps(payload, ensure_ascii=True, indent=2))
            return 0

        print(
            _format_code_search_text(
                query=code_query,
                repo_id=args.repo_id,
                level=args.code_level,
                rows=rows,
            )
        )
        return 0

    store = GraphStore(config.graph_db_path)

    rows = store.get_edges(
        repo_id=args.repo_id,
        edge_type=args.edge_type,
        file_path=args.file_path,
        src_id=args.src_id,
        dst_id=args.dst_id,
    )[: max(0, int(args.limit))]

    if args.output == "json":
        payload = {
            "repo_id": args.repo_id,
            "count": len(rows),
            "edges": [r.to_dict() for r in rows],
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0

    if args.output == "summary":
        by_type: dict[str, int] = {}
        by_file: dict[str, int] = {}
        for row in rows:
            by_type[row.edge_type] = by_type.get(row.edge_type, 0) + 1
            by_file[row.file_path] = by_file.get(row.file_path, 0) + 1

        payload = {
            "repo_id": args.repo_id,
            "count": len(rows),
            "graph_db": config.graph_db_path,
            "by_type": dict(sorted(by_type.items())),
            "by_file": dict(sorted(by_file.items())),
            "filters": {
                "edge_type": args.edge_type,
                "file_path": args.file_path,
                "src_id": args.src_id,
                "dst_id": args.dst_id,
                "limit": int(args.limit),
            },
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