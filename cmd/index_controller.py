"""
Index controller — logic layer for the document indexing tab.

Responsibilities:
  - Accept raw PDF bytes, persist to a temp file, run the chunking + indexing pipeline
  - Hold last-run result for the display layer to read
  - List indexed doc_ids from Chroma

Has NO dependency on NiceGUI.
"""

import os
import tempfile

from utils.file_processor import load_and_chunk_pdf
from utils.logger import AppLogger
from rag.client import LocalLlamaClient

log = AppLogger.get(__name__)


class IndexController:
    """Handles PDF ingestion and exposes state to the Index tab."""

    def __init__(self, client: LocalLlamaClient) -> None:
        self._client = client
        self._last_result: str = ""
        self._last_ok: bool = False

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    @property
    def last_result(self) -> str:
        return self._last_result

    @property
    def last_ok(self) -> bool:
        return self._last_ok

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def index_pdf(
        self,
        file_name: str,
        file_bytes: bytes,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        doc_id: str | None = None,
    ) -> bool:
        """Persist bytes to a temp file and run the indexing pipeline.

        Returns True on success, False on error.
        Sets last_result with a human-readable outcome message.
        """
        resolved_doc_id = (doc_id or "").strip() or file_name
        log.info(
            "index_pdf: file=%s  doc_id=%s  chunk_size=%d  overlap=%d",
            file_name, resolved_doc_id, chunk_size, chunk_overlap,
        )
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            chunks = load_and_chunk_pdf(
                tmp_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_id=resolved_doc_id,
            )
            stats = self._client.indexer.run(chunks)
            self._last_result = (
                f"{file_name} — {len(chunks)} chunks  "
                f"added={stats.get('num_added', 0)}  "
                f"updated={stats.get('num_updated', 0)}  "
                f"skipped={stats.get('num_skipped', 0)}"
            )
            self._last_ok = True
            log.info("index_pdf done: %s", self._last_result)
            return True

        except Exception as e:
            self._last_result = f"Error: {e}"
            self._last_ok = False
            log.error("index_pdf failed: %s", e, exc_info=True)
            return False

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def list_docs(self) -> list[str]:
        """Return all distinct doc_id values from Chroma."""
        try:
            return self._client.list_doc_ids()
        except Exception as e:
            log.error("list_docs failed: %s", e)
            return []
