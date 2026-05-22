"""
Retrieval Evaluation CLI runner.

Loads a YAML dataset, runs all pipeline modes against the live Chroma collection,
and prints a comparison table of Recall@K, MRR, and NDCG@K.

Usage
-----
    python testing/testing_eval.py [--dataset PATH] [--k K] [--fetch-k FETCH_K]
                                   [--modes MODE [MODE ...]] [--per-query MODE]

Options
-------
    --dataset   PATH        Path to the YAML evaluation dataset.
                            Default: data/eval_dataset.yaml
    --k         INT         Rank cut-off for metrics.  Default: 5
    --fetch-k   INT         Candidate pool size for vector/hybrid.  Default: 20
    --modes     MODE ...    Pipeline modes to evaluate.
                            Choices: vector bm25 hybrid reranked
                            Default: all applicable modes
    --per-query MODE        Print a per-query breakdown for this mode.

Examples
--------
    # Run with defaults (all modes, K=5)
    python testing/testing_eval.py

    # Custom dataset and K
    python testing/testing_eval.py --dataset data/my_eval.yaml --k 10

    # Only vector and hybrid, plus per-query breakdown for hybrid
    python testing/testing_eval.py --modes vector hybrid --per-query hybrid
"""

import argparse
import sys

import pytest

from utils.config import AppConfig
from utils.logger import AppLogger
from rag.client import LocalLlamaClient
from rag.retrieval.eval import EvalDataset, RetrievalEvaluator

log = AppLogger.get(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation against the live Chroma collection."
    )
    parser.add_argument(
        "--dataset", default="data/eval_dataset.yaml",
        help="Path to the YAML evaluation dataset (default: data/eval_dataset.yaml)",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Rank cut-off for all metrics (default: 5)",
    )
    parser.add_argument(
        "--fetch-k", type=int, default=20, dest="fetch_k",
        help="Candidate pool size for vector/hybrid modes (default: 20)",
    )
    parser.add_argument(
        "--modes", nargs="+",
        choices=["vector", "bm25", "hybrid", "reranked"],
        default=None,
        help="Pipeline modes to evaluate (default: all applicable)",
    )
    parser.add_argument(
        "--per-query", dest="per_query", default=None,
        metavar="MODE",
        help="Print a per-query breakdown table for this mode after the summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    try:
        dataset = EvalDataset.from_yaml(args.dataset)
    except FileNotFoundError:
        print(f"ERROR: dataset file not found: {args.dataset}", file=sys.stderr)
        print(
            "  Create one at data/eval_dataset.yaml — see data/eval_dataset.yaml.example",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(dataset) == 0:
        print("WARNING: dataset is empty — nothing to evaluate.")
        return

    # ── Initialise client ─────────────────────────────────────────────────────
    config = AppConfig()
    print(f"Connecting to Chroma at {config.persist_directory} …")
    client = LocalLlamaClient(config)

    # ── Run evaluation ────────────────────────────────────────────────────────
    evaluator = RetrievalEvaluator(
        client,
        k=args.k,
        fetch_k=args.fetch_k,
        modes=args.modes,
    )
    report = evaluator.run(dataset)

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print(report.summary_table())

    # ── Optional per-query breakdown ──────────────────────────────────────────
    if args.per_query:
        mode = args.per_query
        if mode not in report.modes:
            print(f"\nWARNING: mode {mode!r} was not evaluated; skipping per-query table.")
        else:
            print()
            print(report.per_query_table(mode))


@pytest.mark.integration
def test_retrieval_eval():
    main()


if __name__ == "__main__":
    main()
