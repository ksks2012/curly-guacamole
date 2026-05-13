"""
QA Pair Generation — Stage B.2.

Calls an LLM to generate question-answer pairs from each text chunk and
returns them as indexable LangChain Documents (question as page_content,
answer + back-references in metadata).

Storing *questions* as page_content means that semantic search on a user
query matches against questions rather than raw chunk text.  A question
like "What is MMR retrieval?" is far more similar to a user query than the
dense technical paragraph the question was derived from.

Generated QA pairs are indexed into a dedicated Chroma collection (the
main document collection is not touched) so normal retrieval is not
affected.

Usage
-----
    from rag.knowledge.qa_generator import QAGenerator

    gen   = QAGenerator(llm)
    pairs = gen.generate_for_docs(chunks)          # batch generation
    docs  = [p.to_document() for p in pairs]       # ready for Indexer.run()

For post-hoc generation on already-indexed content use
``LocalLlamaClient.generate_qa(doc_id)``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from rag.prompt import QA_GENERATION_PROMPT
from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

log = AppLogger.get(__name__)

_FENCE_RE      = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _try_parse_list(s: str) -> list | None:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else None
    except json.JSONDecodeError:
        return None


def _extract_pairs(raw: str) -> list[dict]:
    """Parse a JSON array of {question, answer} dicts from *raw*."""
    cleaned = _FENCE_RE.sub("", raw.strip())
    if result := _try_parse_list(cleaned):
        return result
    m = _JSON_ARRAY_RE.search(cleaned)
    return _try_parse_list(m.group()) if m else []


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """A single question-answer pair derived from one chunk."""

    question: str
    answer:   str
    chunk_id: str
    doc_id:   str

    def to_document(self) -> Document:
        """Convert to a LangChain Document for vector indexing.

        The *question* is stored as ``page_content`` so semantic search on
        user queries matches questions.  The answer and source references are
        kept in metadata.

        ``source_id`` is a content-based hash so the Indexer's record manager
        can deduplicate correctly across multiple generate_qa calls.
        """
        source_id = "qa:" + hashlib.sha256(
            f"{self.chunk_id}:{self.question}".encode()
        ).hexdigest()[:16]
        return Document(
            page_content=self.question,
            metadata={
                "answer":      self.answer,
                "chunk_id":    self.chunk_id,
                "doc_id":      self.doc_id,
                "record_type": "qa",
                "source_id":   source_id,
            },
        )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class QAGenerator:
    """Generate QA pairs from text chunks via LLM.

    Args:
        llm       : ChatOpenAI (or compatible) instance.
        max_chars : Maximum characters of chunk text sent to the LLM.
                    Longer chunks are truncated.  Default: 2000.
        n_pairs   : Target number of QA pairs per chunk.  The LLM may
                    return fewer when the text is short.  Default: 5.
    """

    def __init__(
        self,
        llm: "ChatOpenAI",
        max_chars: int = 2000,
        n_pairs:   int = 5,
    ) -> None:
        self._llm       = llm
        self._max_chars = max_chars
        self._n_pairs   = n_pairs

    # ------------------------------------------------------------------
    # Single chunk
    # ------------------------------------------------------------------

    def generate(self, text: str, chunk_id: str, doc_id: str) -> list[QAPair]:
        """Generate QA pairs from *text*.

        Returns an empty list on LLM failure or unparseable response
        (never raises).
        """
        truncated = text[: self._max_chars]
        prompt    = QA_GENERATION_PROMPT.format(text=truncated, n=self._n_pairs)
        try:
            response = self._llm.invoke(prompt)
            raw      = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            log.warning("QAGenerator: LLM call failed for chunk %s: %s", chunk_id, exc)
            return []

        raw_pairs = _extract_pairs(raw)
        if not raw_pairs:
            log.warning(
                "QAGenerator: could not parse response for chunk %s: %r…",
                chunk_id, raw[:120],
            )
            return []

        result: list[QAPair] = []
        for item in raw_pairs:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question") or "").strip()
            a = str(item.get("answer")   or "").strip()
            if q and a:
                result.append(QAPair(question=q, answer=a, chunk_id=chunk_id, doc_id=doc_id))

        log.debug("QAGenerator: %d pairs from chunk %s", len(result), chunk_id)
        return result

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def generate_for_docs(self, docs: list[Document]) -> list[QAPair]:
        """Generate QA pairs for all *docs*.

        Calls :meth:`generate` once per Document.  Results from all docs
        are concatenated and returned in input order.
        """
        total     = len(docs)
        all_pairs: list[QAPair] = []
        log.info("QAGenerator.generate_for_docs: %d docs", total)
        for idx, doc in enumerate(docs):
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            doc_id   = str(doc.metadata.get("doc_id",   ""))
            log.debug(
                "  [%d/%d] chunk_id=%s  chars=%d",
                idx + 1, total, chunk_id, len(doc.page_content),
            )
            all_pairs.extend(self.generate(doc.page_content, chunk_id, doc_id))
        log.info("QAGenerator.generate_for_docs: %d total pairs", len(all_pairs))
        return all_pairs
