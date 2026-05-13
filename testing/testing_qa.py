"""
Unit tests for Stage B.2 — QA Pair Generation.

Run:
    python testing/testing_qa.py [--no-live]

Tests 1-5 require no LLM (mock-based).
Test 6 is live (calls the local LLM server) and is skipped with --no-live.
"""

import sys
sys.path.insert(0, ".")

from langchain_core.documents import Document

from rag.knowledge.qa_generator import QAPair, QAGenerator, _extract_pairs


# ---------------------------------------------------------------------------
# Test 1: _extract_pairs — plain JSON array
# ---------------------------------------------------------------------------

def test_extract_pairs_plain():
    raw = '[{"question": "What is X?", "answer": "X is a thing."}]'
    pairs = _extract_pairs(raw)
    assert pairs == [{"question": "What is X?", "answer": "X is a thing."}], pairs
    print("_extract_pairs (plain JSON): OK")


# ---------------------------------------------------------------------------
# Test 2: _extract_pairs — markdown fences
# ---------------------------------------------------------------------------

def test_extract_pairs_fences():
    raw = '```json\n[{"question": "Why?", "answer": "Because."}]\n```'
    pairs = _extract_pairs(raw)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "Why?"
    assert pairs[0]["answer"]   == "Because."
    print("_extract_pairs (markdown fences): OK")


# ---------------------------------------------------------------------------
# Test 3: _extract_pairs — JSON embedded in surrounding text
# ---------------------------------------------------------------------------

def test_extract_pairs_embedded():
    raw = 'Sure! Here are the pairs:\n[{"question": "How?", "answer": "Like this."}]\nDone.'
    pairs = _extract_pairs(raw)
    assert len(pairs) == 1 and pairs[0]["question"] == "How?"
    print("_extract_pairs (embedded): OK")


# ---------------------------------------------------------------------------
# Test 4: QAPair.to_document — field mapping + stable source_id
# ---------------------------------------------------------------------------

def test_qa_pair_to_document():
    pair = QAPair(
        question="What is PSO?",
        answer="PSO is a population-based optimisation algorithm.",
        chunk_id="cid-001",
        doc_id="did-001",
    )
    doc = pair.to_document()
    assert doc.page_content                   == "What is PSO?"
    assert doc.metadata["answer"]             == "PSO is a population-based optimisation algorithm."
    assert doc.metadata["chunk_id"]           == "cid-001"
    assert doc.metadata["doc_id"]             == "did-001"
    assert doc.metadata["record_type"]        == "qa"
    assert doc.metadata["source_id"].startswith("qa:")

    # Stable: identical pair always produces identical source_id
    assert pair.to_document().metadata["source_id"] == doc.metadata["source_id"]

    # Different question → different source_id
    other = QAPair(question="Why PSO?", answer="Speed.", chunk_id="cid-001", doc_id="did-001")
    assert other.to_document().metadata["source_id"] != doc.metadata["source_id"]

    print("QAPair.to_document: OK")


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class _MockLLM:
    """Returns a fixed two-pair JSON response for any input."""

    RESPONSE = (
        '[{"question": "What is X?", "answer": "X is a concept."}, '
        '{"question": "Why use X?", "answer": "For efficiency."}]'
    )

    def invoke(self, prompt):
        class _R:
            content = _MockLLM.RESPONSE
        return _R()


# ---------------------------------------------------------------------------
# Test 5: QAGenerator.generate — mock LLM
# ---------------------------------------------------------------------------

def test_generate_mock():
    gen   = QAGenerator(_MockLLM(), n_pairs=2)
    pairs = gen.generate("X is a concept used for efficiency.", chunk_id="c1", doc_id="d1")
    assert len(pairs)          == 2,          f"expected 2, got {len(pairs)}"
    assert pairs[0].question   == "What is X?"
    assert pairs[1].answer     == "For efficiency."
    assert all(p.chunk_id == "c1" for p in pairs)
    assert all(p.doc_id   == "d1" for p in pairs)
    print(f"generate (mock): OK  {len(pairs)} pairs")


# ---------------------------------------------------------------------------
# Test 6: QAGenerator.generate_for_docs — mock LLM, two docs
# ---------------------------------------------------------------------------

def test_generate_for_docs_mock():
    gen  = QAGenerator(_MockLLM(), n_pairs=2)
    docs = [
        Document(page_content="Text A", metadata={"chunk_id": "c1", "doc_id": "d1"}),
        Document(page_content="Text B", metadata={"chunk_id": "c2", "doc_id": "d1"}),
    ]
    pairs = gen.generate_for_docs(docs)
    assert len(pairs)                    == 4,  f"expected 4, got {len(pairs)}"
    assert {p.chunk_id for p in pairs}   == {"c1", "c2"}
    assert all(p.doc_id == "d1" for p in pairs)
    print(f"generate_for_docs (mock): OK  {len(pairs)} pairs across {len(docs)} docs")


# ---------------------------------------------------------------------------
# Test 7 (live): full round-trip with real LLM + Chroma
# ---------------------------------------------------------------------------

def test_live():
    from utils.config import AppConfig
    from rag.client import LocalLlamaClient

    config = AppConfig("etc/config.yaml")
    client = LocalLlamaClient(config)

    # Use a short synthetic chunk; no real indexed documents needed
    from langchain_core.documents import Document as _Doc
    chunk = _Doc(
        page_content="Particle Swarm Optimisation (PSO) is a computational method "
                     "that optimises a problem by iteratively improving candidate "
                     "solutions with regard to a given measure of quality.",
        metadata={"chunk_id": "test-live-c1", "doc_id": "test-live-d1"},
    )

    pairs = client.qa_generator.generate([chunk.page_content], "test-live-c1", "test-live-d1")
    assert len(pairs) > 0, "live generation returned no pairs"
    print(f"  Live generate: {len(pairs)} pairs")
    for p in pairs[:2]:
        print(f"    Q: {p.question}")
        print(f"    A: {p.answer[:80]}")
    print("generate (live): OK")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_extract_pairs_plain()
    test_extract_pairs_fences()
    test_extract_pairs_embedded()
    test_qa_pair_to_document()
    test_generate_mock()
    test_generate_for_docs_mock()

    if "--no-live" in sys.argv:
        print("\nLive test skipped (--no-live)")
    else:
        print("\nRunning live test …")
        test_live()

    print("\nAll QA generation tests passed.")
