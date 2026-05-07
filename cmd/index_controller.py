"""
Index controller — logic layer for the document indexing tab.

Responsibilities:
  - Accept raw PDF bytes, persist to a temp file, run the chunking + indexing pipeline
  - Hold last-run result for the display layer to read
  - List indexed doc_ids from Chroma

Has NO dependency on NiceGUI.
"""

import os
import re

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

    def save_pdf(
        self,
        file_name: str,
        file_bytes: bytes,
        doc_id: str | None = None,
    ) -> str:
        """Save raw PDF bytes to the configured upload directory.

        The file is named after *doc_id* (sanitised) so it is human-readable.
        Returns the absolute path of the saved file.
        Raises on I/O error.
        """
        resolved_doc_id = (doc_id or "").strip() or file_name
        upload_dir = self._client.config.upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        safe_name = re.sub(r'[^\w\-.]', '_', resolved_doc_id)
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        save_path = os.path.join(upload_dir, safe_name)
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        log.info("save_pdf: %s -> %s (%d bytes)", file_name, save_path, len(file_bytes))
        return save_path

    def embed_pdf(
        self,
        save_path: str,
        doc_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> bool:
        """Load chunks from a saved PDF and run the embedding + indexing pipeline.

        Intended to be called in a background thread (slow operation).
        Sets last_result / last_ok. Returns True on success.
        """
        file_name = os.path.basename(save_path)
        log.info(
            "embed_pdf: path=%s  doc_id=%s  chunk_size=%d  overlap=%d",
            save_path, doc_id, chunk_size, chunk_overlap,
        )
        try:
            chunks = load_and_chunk_pdf(
                save_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_id=doc_id,
            )
            stats = self._client.indexer.run(chunks)
            self._last_result = (
                f"{file_name} — {len(chunks)} chunks  "
                f"added={stats.get('num_added', 0)}  "
                f"updated={stats.get('num_updated', 0)}  "
                f"skipped={stats.get('num_skipped', 0)}"
            )
            self._last_ok = True
            log.info("embed_pdf done: %s", self._last_result)
            return True
        except Exception as e:
            self._last_result = f"Error: {e}"
            self._last_ok = False
            log.error("embed_pdf failed: %s", e, exc_info=True)
            return False

    def index_pdf(
        self,
        file_name: str,
        file_bytes: bytes,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        doc_id: str | None = None,
    ) -> bool:
        """Convenience wrapper: save bytes then embed. Returns True on success."""
        resolved_doc_id = (doc_id or "").strip() or file_name
        try:
            save_path = self.save_pdf(file_name, file_bytes, doc_id)
        except Exception as e:
            self._last_result = f"Error saving file: {e}"
            self._last_ok = False
            log.error("index_pdf save failed: %s", e, exc_info=True)
            return False
        return self.embed_pdf(save_path, resolved_doc_id, chunk_size, chunk_overlap)

    def list_docs(self) -> list[str]:
        """Return all distinct doc_id values from Chroma."""
        try:
            return self._client.list_doc_ids()
        except Exception as e:
            log.error("list_docs failed: %s", e)
            return []
