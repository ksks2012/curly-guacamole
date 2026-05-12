"""
Retrieval Evaluation — metrics and pipeline runner.

Metrics
-------
    Recall@K  — fraction of relevant items found in the top-K results.
    MRR       — Mean Reciprocal Rank; rewards finding a relevant item early.
    NDCG@K    — Normalised Discounted Cumulative Gain; position-weighted recall.

Relevance judgements
--------------------
Each EvalQuery contains a set of ``relevant_chunks`` (chunk_id values) and/or
``relevant_docs`` (doc_id values).  A retrieved chunk is considered relevant
when its ``chunk_id`` is in ``relevant_chunks`` OR its ``doc_id`` is in
``relevant_docs``.  This dual-level matching lets you write coarse (doc-level)
labels when you haven't yet pinned down exact chunks.

Pipeline modes evaluated
------------------------
    vector   — similarity_search_with_scores (MMR off, raw cosine/L2)
    bm25     — BM25Index.search
    hybrid   — RRF fusion of vector + BM25 (no reranker)
    reranked — hybrid results re-ordered by BaseReranker (when enabled)

Usage
-----
    from rag.client import LocalLlamaClient
    from rag.retrieval.eval import EvalDataset, RetrievalEvaluator

    dataset  = EvalDataset.from_yaml("data/eval_dataset.yaml")
    client   = LocalLlamaClient(config)
    evaluator = RetrievalEvaluator(client, k=5, fetch_k=20)
    report   = evaluator.run(dataset)
    print(report.summary_table())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from utils.logger import AppLogger

if TYPE_CHECKING:
    from rag.client import LocalLlamaClient

log = AppLogger.get(__name__)


# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """A single evaluation example.

    Attributes
    ----------
    query           : Natural-language question or search phrase.
    relevant_chunks : Set of ``chunk_id`` values (int or str) that are
                      considered relevant answers.  May be empty when only
                      doc-level labels are available.
    relevant_docs   : Set of ``doc_id`` values.  A retrieved chunk is
                      considered relevant when its doc_id is in this set,
                      even if its chunk_id is not in relevant_chunks.
    note            : Optional free-text note for the dataset author.
    """

    query:           str
    relevant_chunks: set[str] = field(default_factory=set)
    relevant_docs:   set[str] = field(default_factory=set)
    note:            str = ""

    def is_relevant(self, doc: Document) -> bool:
        """Return True when *doc* satisfies any relevance judgement."""
        meta = doc.metadata
        chunk_id = str(meta.get("chunk_id", ""))
        doc_id   = str(meta.get("doc_id", ""))
        return chunk_id in self.relevant_chunks or doc_id in self.relevant_docs


@dataclass
class EvalDataset:
    """Collection of EvalQuery objects loaded from a YAML file.

    YAML format
    -----------
    .. code-block:: yaml

        - query: "What is PSO?"
          relevant_docs:
            - "my-doc"
          relevant_chunks:
            - "3"
          note: "optional annotation"

        - query: "Explain gradient descent"
          relevant_docs:
            - "ml-notes"
    """

    queries: list[EvalQuery] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "EvalDataset":
        """Load an EvalDataset from a YAML file at *path*."""
        import yaml  # pyyaml — already in requirements
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or []
        queries = []
        for item in raw:
            queries.append(
                EvalQuery(
                    query=str(item.get("query", "")),
                    relevant_chunks={str(c) for c in item.get("relevant_chunks", [])},
                    relevant_docs={str(d) for d in item.get("relevant_docs", [])},
                    note=str(item.get("note", "")),
                )
            )
        log.info("EvalDataset loaded: %d queries from %s", len(queries), path)
        return cls(queries=queries)

    def __len__(self) -> int:
        return len(self.queries)


# ---------------------------------------------------------------------------
# Metric functions (stateless)
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved: list[Document], relevant_fn, k: int, n_relevant: int = 0
) -> float:
    """Recall@K — fraction of known-relevant items found in the top-K retrieved.

    Args:
        retrieved   : Ranked list of retrieved Documents (best first).
        relevant_fn : Callable(Document) -> bool — relevance oracle.
        k           : Cut-off rank.
        n_relevant  : Total number of known-relevant items (denominator).
                      Pass ``len(eq.relevant_docs) + len(eq.relevant_chunks)``
                      or any positive integer.  When 0 or unset, falls back to
                      the number of hits in the full *retrieved* list, which
                      gives Recall@K = 1.0 when all labelled items were found.

    Returns:
        Float in [0, 1].  Returns 0.0 when there are no relevant items.
    """
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if relevant_fn(d))
    denom = n_relevant if n_relevant > 0 else sum(1 for d in retrieved if relevant_fn(d))
    return hits / denom if denom > 0 else 0.0


def mrr(retrieved: list[Document], relevant_fn) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant item.

    Returns 0.0 when no relevant item is found in the retrieved list.
    """
    for rank, doc in enumerate(retrieved, start=1):
        if relevant_fn(doc):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[Document], relevant_fn, k: int) -> float:
    """Normalised Discounted Cumulative Gain @ K.

    Uses binary relevance (0 or 1).  The ideal DCG assumes all top-K slots
    are filled with relevant items up to the number of available relevant
    documents.

    Returns:
        Float in [0, 1].
    """
    top_k = retrieved[:k]
    dcg = sum(
        relevant_fn(doc) / math.log2(rank + 1)
        for rank, doc in enumerate(top_k, start=1)
    )
    # Ideal DCG: all hits at the top
    n_relevant = sum(1 for d in top_k if relevant_fn(d))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-query result
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Metrics for a single query across all pipeline modes."""

    query:   str
    k:       int

    # Per-mode scores — set after evaluation
    recall:  dict[str, float] = field(default_factory=dict)   # mode → score
    mrr_:    dict[str, float] = field(default_factory=dict)
    ndcg:    dict[str, float] = field(default_factory=dict)

    # Raw retrieved docs per mode (for inspection / debugging)
    retrieved: dict[str, list[Document]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Aggregate evaluation results across all queries.

    Attributes
    ----------
    k          : Rank cut-off used for all metrics.
    fetch_k    : Candidate pool size (for modes that use it).
    modes      : List of pipeline mode names evaluated.
    results    : Per-query QueryResult objects.
    avg_recall : Averaged Recall@K per mode.
    avg_mrr    : Averaged MRR per mode.
    avg_ndcg   : Averaged NDCG@K per mode.
    """

    k:         int
    fetch_k:   int
    modes:     list[str]
    results:   list[QueryResult] = field(default_factory=list)
    avg_recall: dict[str, float] = field(default_factory=dict)
    avg_mrr:    dict[str, float] = field(default_factory=dict)
    avg_ndcg:   dict[str, float] = field(default_factory=dict)

    def _compute_averages(self) -> None:
        """Compute per-mode averages from individual QueryResults."""
        for mode in self.modes:
            n = len(self.results)
            if n == 0:
                self.avg_recall[mode] = 0.0
                self.avg_mrr[mode]    = 0.0
                self.avg_ndcg[mode]   = 0.0
            else:
                self.avg_recall[mode] = sum(r.recall.get(mode, 0.0) for r in self.results) / n
                self.avg_mrr[mode]    = sum(r.mrr_.get(mode, 0.0) for r in self.results) / n
                self.avg_ndcg[mode]   = sum(r.ndcg.get(mode, 0.0) for r in self.results) / n

    def summary_table(self) -> str:
        """Return a human-readable ASCII table of averaged metrics."""
        col_w  = 12
        mode_w = 10
        header  = f"{'Mode':<{mode_w}}  {'Recall@K':>{col_w}}  {'MRR':>{col_w}}  {'NDCG@K':>{col_w}}"
        divider = "-" * len(header)
        lines   = [
            f"Retrieval Evaluation  (K={self.k}, fetch_k={self.fetch_k},"
            f" queries={len(self.results)})",
            divider,
            header,
            divider,
        ]
        for mode in self.modes:
            lines.append(
                f"{mode:<{mode_w}}  "
                f"{self.avg_recall.get(mode, 0.0):>{col_w}.4f}  "
                f"{self.avg_mrr.get(mode, 0.0):>{col_w}.4f}  "
                f"{self.avg_ndcg.get(mode, 0.0):>{col_w}.4f}"
            )
        lines.append(divider)
        return "\n".join(lines)

    def per_query_table(self, mode: str) -> str:
        """Return a per-query breakdown for a single *mode*."""
        lines = [f"Per-query results — mode: {mode}  K={self.k}", "-" * 60]
        for r in self.results:
            lines.append(
                f"  Q: {r.query[:50]!r:<52}"
                f"  R@K={r.recall.get(mode, 0.0):.3f}"
                f"  MRR={r.mrr_.get(mode, 0.0):.3f}"
                f"  NDCG={r.ndcg.get(mode, 0.0):.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """Run retrieval evaluation across all pipeline modes.

    Supported modes (evaluated in parallel per query):
        ``vector``   — raw vector similarity search
        ``bm25``     — BM25 keyword search
        ``hybrid``   — RRF fusion of vector + BM25
        ``reranked`` — hybrid results passed through the configured reranker
                       (only included when a reranker is available)

    Args
    ----
    client  : LocalLlamaClient instance (provides all retrieval methods).
    k       : Rank cut-off for all metrics.
    fetch_k : Candidate pool size for vector and hybrid search.
    modes   : Override the list of pipeline modes to evaluate.  Defaults to
              all applicable modes based on what the client has configured.
    """

    def __init__(
        self,
        client: "LocalLlamaClient",
        k: int = 5,
        fetch_k: int = 20,
        modes: list[str] | None = None,
    ) -> None:
        self._client  = client
        self.k        = k
        self.fetch_k  = fetch_k

        # Determine which modes to run
        has_reranker = client.reranker is not None
        default_modes = ["vector", "bm25", "hybrid"]
        if has_reranker:
            default_modes.append("reranked")
        self.modes = modes if modes is not None else default_modes

        log.info(
            "RetrievalEvaluator: k=%d  fetch_k=%d  modes=%s",
            self.k, self.fetch_k, self.modes,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, dataset: EvalDataset) -> EvalReport:
        """Evaluate all queries in *dataset* and return an EvalReport."""
        log.info("Evaluation start: %d queries", len(dataset))

        # Ensure BM25 is fresh before the evaluation loop
        if "bm25" in self.modes or "hybrid" in self.modes:
            self._client.rebuild_bm25()

        report = EvalReport(k=self.k, fetch_k=self.fetch_k, modes=self.modes)

        for idx, eq in enumerate(dataset.queries):
            log.info("  [%d/%d] %r", idx + 1, len(dataset), eq.query)
            qr = self._eval_query(eq)
            report.results.append(qr)

        report._compute_averages()
        log.info("Evaluation complete.\n%s", report.summary_table())
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _eval_query(self, eq: EvalQuery) -> QueryResult:
        """Run all pipeline modes for a single EvalQuery."""
        qr = QueryResult(query=eq.query, k=self.k)
        relevant   = eq.is_relevant
        n_relevant = len(eq.relevant_docs) + len(eq.relevant_chunks)

        for mode in self.modes:
            try:
                docs = self._retrieve(mode, eq.query)
            except Exception as exc:
                log.warning("mode=%s query=%r error: %s", mode, eq.query, exc)
                docs = []

            qr.retrieved[mode] = docs
            qr.recall[mode]    = recall_at_k(docs, relevant, self.k, n_relevant=n_relevant)
            qr.mrr_[mode]      = mrr(docs, relevant)
            qr.ndcg[mode]      = ndcg_at_k(docs, relevant, self.k)

        return qr

    def _retrieve(self, mode: str, query: str) -> list[Document]:
        """Run one retrieval pipeline mode and return a ranked Document list."""
        client = self._client

        if mode == "vector":
            pairs = client.similarity_search_with_scores(query, k=self.fetch_k)
            return [doc for doc, _ in pairs][: self.k]

        if mode == "bm25":
            pairs = client.bm25_index.search(query, k=self.fetch_k)
            return [doc for doc, _ in pairs][: self.k]

        if mode == "hybrid":
            _, _, fused = client.hybrid_search_with_scores(
                query, k=self.k, fetch_k=self.fetch_k
            )
            return [doc for doc, _ in fused][: self.k]

        if mode == "reranked":
            _, _, fused = client.hybrid_search_with_scores(
                query, k=self.k, fetch_k=self.fetch_k
            )
            candidates = [doc for doc, _ in fused]
            if client.reranker is not None and candidates:
                return client.reranker.rerank(query, candidates, top_k=self.k)
            return candidates[: self.k]

        raise ValueError(f"Unknown retrieval mode: {mode!r}")
