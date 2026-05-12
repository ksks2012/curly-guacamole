"""
Knowledge Extraction — Stage B.1.

Calls an LLM to extract structured knowledge artifacts from each chunk and
stamps the results back into the chunk's metadata.  Extraction is opt-in and
always runs synchronously (one LLM call per chunk, batched in a loop).

Extracted fields (all stored with the ``ka_`` prefix to avoid collision with
existing page-level metadata):

    ka_summary   : str  — 2-3 sentence summary of the chunk
    ka_keywords  : str  — comma-joined keyword phrases (3-8 terms)
    ka_entities  : str  — comma-joined named entities
    ka_topics    : str  — comma-joined broad topics (1-4 categories)
    ka_questions : str  — comma-joined questions the chunk answers (2-5)

All values are Chroma-safe scalars (str).  Empty string means "not extracted".

Usage
-----
    from rag.knowledge.extractor import KnowledgeExtractor

    extractor = KnowledgeExtractor(llm)
    enriched_docs = extractor.enrich(docs)   # pre-indexing path

For post-hoc enrichment of already-indexed docs, use
``LocalLlamaClient.enrich_doc(doc_id)``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from rag.prompt import KNOWLEDGE_EXTRACTION_PROMPT
from utils.logger import AppLogger

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

log = AppLogger.get(__name__)

# Chroma metadata key names for all extracted fields
KA_SUMMARY   = "ka_summary"
KA_KEYWORDS  = "ka_keywords"
KA_ENTITIES  = "ka_entities"
KA_TOPICS    = "ka_topics"
KA_QUESTIONS = "ka_questions"

# Sentinel value: extraction was attempted but failed
KA_FAILED = "__extraction_failed__"

# Default empty artifact (all fields absent or empty)
_EMPTY: dict[str, str] = {
    KA_SUMMARY:   "",
    KA_KEYWORDS:  "",
    KA_ENTITIES:  "",
    KA_TOPICS:    "",
    KA_QUESTIONS: "",
}


_FENCE_RE     = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_JSON_BLOB_RE = re.compile(r"\{.*\}", re.DOTALL)


def _join(values: list) -> str:
    """Comma-join a list of strings, ignoring blanks."""
    return ", ".join(str(v).strip() for v in values if str(v).strip())


def _try_json(s: str) -> dict | None:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_json(raw: str) -> dict | None:
    """Parse a JSON object from *raw*, tolerating markdown fences."""
    cleaned = _FENCE_RE.sub("", raw.strip())
    if result := _try_json(cleaned):
        return result
    m = _JSON_BLOB_RE.search(cleaned)
    return _try_json(m.group()) if m else None


class KnowledgeExtractor:
    """Extract structured knowledge artifacts from text chunks via LLM.

    Args:
        llm        : ChatOpenAI (or compatible) instance.  The same LLM used
                     by the RAG engine — no second model needed.
        max_chars  : Maximum characters of chunk text sent to the LLM.
                     Longer chunks are truncated (the summary still covers
                     the most important parts).  Default: 2000.
    """

    def __init__(self, llm: "ChatOpenAI", max_chars: int = 2000) -> None:
        self._llm      = llm
        self._max_chars = max_chars

    # ------------------------------------------------------------------
    # Single-chunk extraction
    # ------------------------------------------------------------------

    def extract_one(self, text: str) -> dict[str, str]:
        """Extract a knowledge artifact dict from *text*.

        Returns a dict with keys ``ka_summary``, ``ka_keywords``,
        ``ka_entities``, ``ka_topics``, ``ka_questions``.  All values are
        comma-joined strings (Chroma-safe).  On LLM or parse failure the
        dict has all-empty values (never raises).
        """
        truncated = text[: self._max_chars]
        prompt    = KNOWLEDGE_EXTRACTION_PROMPT.format(text=truncated)
        try:
            response = self._llm.invoke(prompt)
            raw      = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            log.warning("KnowledgeExtractor: LLM call failed: %s", exc)
            return dict(_EMPTY)

        obj = _extract_json(raw)
        if obj is None:
            log.warning(
                "KnowledgeExtractor: could not parse JSON response (len=%d): %r…",
                len(raw), raw[:120],
            )
            return dict(_EMPTY)

        return {
            KA_SUMMARY:   obj.get("summary") or "",
            KA_KEYWORDS:  _join(obj.get("keywords") or []),
            KA_ENTITIES:  _join(obj.get("entities") or []),
            KA_TOPICS:    _join(obj.get("topics") or []),
            KA_QUESTIONS: _join(obj.get("questions") or []),
        }

    # ------------------------------------------------------------------
    # Batch enrichment
    # ------------------------------------------------------------------

    def enrich(self, docs: list[Document]) -> list[Document]:
        """Stamp knowledge artifact fields onto each Document's metadata.

        Creates shallow copies of the Documents so the originals are not
        mutated.  Calls :meth:`extract_one` for every document in sequence.

        Args:
            docs : List of Documents to enrich.  May already have ``ka_*``
                   fields — they will be overwritten.

        Returns:
            New list of Documents with ``ka_*`` fields added to metadata.
        """
        enriched: list[Document] = []
        total = len(docs)
        log.info("KnowledgeExtractor.enrich: %d documents", total)

        for idx, doc in enumerate(docs):
            log.debug(
                "  [%d/%d] extracting — chunk_id=%s  chars=%d",
                idx + 1, total,
                doc.metadata.get("chunk_id", "?"),
                len(doc.page_content),
            )
            artifact = self.extract_one(doc.page_content)
            new_meta = {**doc.metadata, **artifact}
            enriched.append(
                Document(page_content=doc.page_content, metadata=new_meta)
            )

        log.info("KnowledgeExtractor.enrich: done (%d documents enriched)", total)
        return enriched

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def artifact_from_meta(meta: dict) -> dict[str, str]:
        """Extract only the ``ka_*`` fields from a Chroma metadata dict."""
        return {
            KA_SUMMARY:   meta.get(KA_SUMMARY,   ""),
            KA_KEYWORDS:  meta.get(KA_KEYWORDS,  ""),
            KA_ENTITIES:  meta.get(KA_ENTITIES,  ""),
            KA_TOPICS:    meta.get(KA_TOPICS,    ""),
            KA_QUESTIONS: meta.get(KA_QUESTIONS, ""),
        }

    @staticmethod
    def is_enriched(meta: dict) -> bool:
        """Return True when the metadata already contains a non-empty extraction."""
        return bool(meta.get(KA_SUMMARY) or meta.get(KA_KEYWORDS))
