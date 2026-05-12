"""Unit tests for rag.retrieval.eval metric functions."""

from langchain_core.documents import Document
from rag.retrieval.eval import recall_at_k, mrr, ndcg_at_k, EvalQuery, EvalDataset


def mk(doc_id, chunk_id=""):
    return Document(page_content="x", metadata={"doc_id": doc_id, "chunk_id": str(chunk_id)})


def test_metrics():
    eq = EvalQuery(query="test", relevant_docs={"A", "B"})
    # Ranking: C(miss) A(hit) B(hit) D(miss) E(miss)
    docs = [mk("C"), mk("A"), mk("B"), mk("D"), mk("E")]
    n_rel = len(eq.relevant_docs) + len(eq.relevant_chunks)  # = 2

    r3 = recall_at_k(docs, eq.is_relevant, k=3, n_relevant=n_rel)
    print(f"Recall@3 = {r3:.4f}  (2 hits / 2 relevant = 1.0, both in top-3)")
    assert abs(r3 - 1.0) < 1e-9, r3

    r1 = recall_at_k(docs, eq.is_relevant, k=1, n_relevant=n_rel)
    print(f"Recall@1 = {r1:.4f}  (0 hits in top-1 → 0.0)")
    assert abs(r1 - 0.0) < 1e-9, r1

    m = mrr(docs, eq.is_relevant)
    print(f"MRR      = {m:.4f}  (first hit at rank 2 → expect 0.5)")
    assert abs(m - 0.5) < 1e-9, m

    n5 = ndcg_at_k(docs, eq.is_relevant, k=5)
    print(f"NDCG@5   = {n5:.4f}  (2 relevant in top-5)")
    assert n5 > 0, n5

    # Edge: no relevant docs in retrieved list
    eq2 = EvalQuery(query="nope", relevant_docs={"Z"})
    assert recall_at_k(docs, eq2.is_relevant, k=5) == 0.0
    assert mrr(docs, eq2.is_relevant) == 0.0
    assert ndcg_at_k(docs, eq2.is_relevant, k=5) == 0.0

    # Edge: chunk_id matching
    eq3 = EvalQuery(query="chunk match", relevant_chunks={"42"})
    docs3 = [mk("X", chunk_id=42), mk("Y", chunk_id=99)]
    assert mrr(docs3, eq3.is_relevant) == 1.0  # first doc matches by chunk_id

    print("Metrics: OK")


def test_dataset_load():
    dataset = EvalDataset.from_yaml("data/eval_dataset.yaml")
    assert len(dataset) > 0, "eval_dataset.yaml should not be empty"
    for q in dataset.queries:
        assert q.query, "each query must have text"
    print(f"EvalDataset: OK  ({len(dataset)} queries loaded)")


if __name__ == "__main__":
    test_metrics()
    test_dataset_load()
    print("\nAll tests passed.")
