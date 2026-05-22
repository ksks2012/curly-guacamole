"""
B.1 Knowledge Extraction — tests.

Test 1: _extract_json helper (no LLM needed)
Test 2: KnowledgeExtractor.extract_one with a mock LLM
Test 3: KnowledgeExtractor.enrich on a list of Documents
Test 4: Live extraction against the real LLM server (requires running server)
"""

import json
import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Helpers from extractor (directly tested)
# ---------------------------------------------------------------------------
from rag.knowledge.extractor import (
    KnowledgeExtractor,
    _extract_json,
    KA_SUMMARY, KA_KEYWORDS, KA_ENTITIES, KA_TOPICS, KA_QUESTIONS,
)


# ---------------------------------------------------------------------------
# Test 1 — _extract_json
# ---------------------------------------------------------------------------

def test_extract_json():
    good = '{"summary": "S", "keywords": ["a", "b"]}'
    assert _extract_json(good)["summary"] == "S"

    fenced = '```json\n{"summary": "X"}\n```'
    assert _extract_json(fenced)["summary"] == "X"

    embedded = 'Here is the result:\n{"summary": "Y", "keywords": []}  \n'
    assert _extract_json(embedded)["summary"] == "Y"

    bad = "sorry I cannot help"
    assert _extract_json(bad) is None

    print("_extract_json: OK")


# ---------------------------------------------------------------------------
# Test 2 — extract_one with mock LLM
# ---------------------------------------------------------------------------

class _MockLLM:
    """Returns a fixed JSON response without calling a real server."""
    def __init__(self, payload: dict):
        import types
        self._payload = payload

    def invoke(self, prompt):
        import types
        resp = types.SimpleNamespace()
        resp.content = json.dumps(self._payload)
        return resp


def test_extract_one_mock():
    payload = {
        "summary":   "PSO is a population-based optimisation algorithm.",
        "keywords":  ["PSO", "swarm", "optimisation"],
        "entities":  ["Kennedy", "Eberhart"],
        "topics":    ["optimisation", "metaheuristics"],
        "questions": ["What is PSO?", "Who invented PSO?"],
    }
    extractor = KnowledgeExtractor(_MockLLM(payload))
    result    = extractor.extract_one("dummy text")

    assert result[KA_SUMMARY] == payload["summary"]
    assert "PSO" in result[KA_KEYWORDS]
    assert "Kennedy" in result[KA_ENTITIES]
    assert "optimisation" in result[KA_TOPICS]
    assert "What is PSO?" in result[KA_QUESTIONS]

    print(f"extract_one (mock): OK  summary={result[KA_SUMMARY][:40]!r}")


# ---------------------------------------------------------------------------
# Test 3 — enrich list of Documents with mock LLM
# ---------------------------------------------------------------------------

def test_enrich_mock():
    payload = {
        "summary":   "Gradient descent minimises a loss function.",
        "keywords":  ["gradient descent", "loss"],
        "entities":  [],
        "topics":    ["machine learning"],
        "questions": ["How does gradient descent work?"],
    }
    docs = [
        Document(page_content="Gradient descent is …", metadata={"doc_id": "ml", "chunk_id": 0}),
        Document(page_content="Adam optimizer …",     metadata={"doc_id": "ml", "chunk_id": 1}),
    ]
    extractor = KnowledgeExtractor(_MockLLM(payload))
    enriched  = extractor.enrich(docs)

    assert len(enriched) == 2
    for d in enriched:
        assert d.metadata[KA_SUMMARY]
        assert "gradient descent" in d.metadata[KA_KEYWORDS]
        assert "doc_id" in d.metadata   # original metadata preserved
    assert KnowledgeExtractor.is_enriched(enriched[0].metadata)
    print(f"enrich (mock): OK  {len(enriched)} docs enriched")


# ---------------------------------------------------------------------------
# Test 4 — live LLM extraction (skip if --no-live flag given)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_extract_one_live():
    from utils.config import AppConfig
    from rag.client import LocalLlamaClient

    print("\nLive extraction test (1 chunk)…")
    config = AppConfig()
    client = LocalLlamaClient(config)

    sample_text = (
        "Particle Swarm Optimisation (PSO) is a computational method that optimises "
        "a problem by iteratively improving candidate solutions. It was first introduced "
        "by Kennedy and Eberhart in 1995. Each particle in the swarm adjusts its position "
        "based on its own best known position and the swarm's best known position."
    )

    result = client.extractor.extract_one(sample_text)

    print(f"  {KA_SUMMARY}   : {result[KA_SUMMARY][:80]!r}")
    print(f"  {KA_KEYWORDS}  : {result[KA_KEYWORDS]}")
    print(f"  {KA_ENTITIES}  : {result[KA_ENTITIES]}")
    print(f"  {KA_TOPICS}    : {result[KA_TOPICS]}")
    print(f"  {KA_QUESTIONS} : {result[KA_QUESTIONS]}")

    assert result[KA_SUMMARY], "summary should not be empty"
    assert result[KA_KEYWORDS], "keywords should not be empty"
    print("extract_one (live): OK")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


