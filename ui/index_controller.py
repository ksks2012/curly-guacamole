"""
Index controller — logic layer for the document indexing tab.

Responsibilities:
  - Accept raw file bytes (PDF / Markdown / plain text), persist to upload_dir
  - Run the ingestion pipeline (parse → chunk → embed → index)
  - Hold last-run result for the display layer to read
  - List indexed doc_ids from Chroma

Has NO dependency on NiceGUI.
"""

import os
import re

from rag.ingest.document_ingester import SUPPORTED_EXTENSIONS
from ui.client_protocols import IndexClientProtocol
from utils.logger import AppLogger

log = AppLogger.get(__name__)


class IndexController:
    """Handles document ingestion (PDF / Markdown / plain text) and exposes state to the Index tab."""

    # Extensions accepted by the upload widget
    ACCEPTED_EXTENSIONS = SUPPORTED_EXTENSIONS

    def __init__(self, client: IndexClientProtocol) -> None:
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

    def save_file(
        self,
        file_name: str,
        file_bytes: bytes,
        doc_id: str | None = None,
    ) -> str:
        """Save raw file bytes to the configured upload directory.

        The saved filename preserves the original extension and uses *doc_id*
        (sanitised) as the stem so it is human-readable.
        Returns the absolute path of the saved file.
        Raises on I/O error.
        """
        resolved_doc_id = (doc_id or "").strip() or file_name
        upload_dir = self._client.config.upload_dir
        os.makedirs(upload_dir, exist_ok=True)
        # Preserve original extension; use doc_id as the stem
        _, ext = os.path.splitext(file_name)
        safe_stem = re.sub(r'[^\w\-.]', '_', resolved_doc_id)
        # Strip any extension already embedded in doc_id to avoid duplication
        if safe_stem.lower().endswith(ext.lower()) and ext:
            safe_stem = safe_stem[: -len(ext)]
        save_path = os.path.join(upload_dir, safe_stem + ext)
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        log.info("save_file: %s -> %s (%d bytes)", file_name, save_path, len(file_bytes))
        return save_path

    # Keep old name as alias for backward compatibility
    def save_pdf(self, file_name: str, file_bytes: bytes, doc_id: str | None = None) -> str:
        return self.save_file(file_name, file_bytes, doc_id)

    def embed_file(
        self,
        save_path: str,
        doc_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        title: str = "",
        tags: list[str] | None = None,
        workspace: str = "",
        importance: float = 0.0,
        strategy: str | None = None,
    ) -> bool:
        """Ingest a saved file through the full pipeline and index it.

        Supports PDF, Markdown, and plain text via DocumentIngester.
        Intended to be called in a background thread (slow operation).
        Sets last_result / last_ok. Returns True on success.
        """
        file_name = os.path.basename(save_path)
        log.info(
            "embed_file: path=%s  doc_id=%s  chunk_size=%d  overlap=%d  strategy=%s",
            save_path, doc_id, chunk_size, chunk_overlap, strategy or "auto",
        )
        try:
            chunks = self._client.ingester.ingest(
                save_path,
                doc_id=doc_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                title=title,
                tags=tags,
                workspace=workspace,
                importance=importance,
                strategy=strategy,
            )
            stats = self._client.indexer.run(chunks)
            self._client.invalidate_bm25()
            self._last_result = (
                f"{file_name} — {len(chunks)} chunks  "
                f"added={stats.get('num_added', 0)}  "
                f"updated={stats.get('num_updated', 0)}  "
                f"skipped={stats.get('num_skipped', 0)}"
            )
            self._last_ok = True
            log.info("embed_file done: %s", self._last_result)
            return True
        except Exception as e:
            self._last_result = f"Error: {e}"
            self._last_ok = False
            log.error("embed_file failed: %s", e, exc_info=True)
            return False

    # Keep old name as alias
    def embed_pdf(self, save_path: str, doc_id: str, chunk_size: int = 500, chunk_overlap: int = 100) -> bool:
        return self.embed_file(save_path, doc_id, chunk_size, chunk_overlap)

    def index_file(
        self,
        file_name: str,
        file_bytes: bytes,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        doc_id: str | None = None,
        title: str = "",
        tags: list[str] | None = None,
        workspace: str = "",
        importance: float = 0.0,
        strategy: str | None = None,
    ) -> bool:
        """Convenience wrapper: save bytes then embed. Returns True on success."""
        resolved_doc_id = (doc_id or "").strip() or file_name
        try:
            save_path = self.save_file(file_name, file_bytes, doc_id)
        except Exception as e:
            self._last_result = f"Error saving file: {e}"
            self._last_ok = False
            log.error("index_file save failed: %s", e, exc_info=True)
            return False
        return self.embed_file(
            save_path, resolved_doc_id, chunk_size, chunk_overlap,
            title=title, tags=tags, workspace=workspace, importance=importance,
            strategy=strategy,
        )

    # Keep old name as alias
    def index_pdf(self, file_name: str, file_bytes: bytes, chunk_size: int = 500,
                  chunk_overlap: int = 100, doc_id: str | None = None) -> bool:
        return self.index_file(file_name, file_bytes, chunk_size, chunk_overlap, doc_id)

    def list_docs(self) -> list[str]:
        """Return all distinct doc_id values from Chroma."""
        try:
            return self._client.list_doc_ids()
        except Exception as e:
            log.error("list_docs failed: %s", e)
            return []

    def list_docs_with_titles(self) -> list[tuple[str, str]]:
        """Return [(doc_id, display_title), ...] for all indexed documents."""
        try:
            return list(self._client.list_doc_title_map().items())
        except Exception as e:
            log.error("list_docs_with_titles failed: %s", e)
            return []

    def get_doc_info(self, doc_id: str) -> dict:
        """Return aggregated metadata for a single document.

        Queries Chroma for all chunks belonging to *doc_id* and returns a
        summary dict with doc-level fields from the first chunk plus the total
        chunk count.
        """
        try:
            result = self._client.db.get(
                where={"doc_id": {"$eq": doc_id}},
                include=["metadatas"],
            )
            metas = result.get("metadatas") or []
            if not metas:
                return {"doc_id": doc_id, "chunk_count": 0}
            first = metas[0] or {}
            return {
                "doc_id":        doc_id,
                "chunk_count":   len(metas),
                "title":         first.get("title", ""),
                "workspace":     first.get("workspace", ""),
                "tags":          first.get("tags", ""),
                "document_type": first.get("document_type", ""),
                "importance":    first.get("importance", ""),
                "source":        first.get("source", ""),
                "created_time":  first.get("created_time", ""),
                "updated_time":  first.get("updated_time", ""),
            }
        except Exception as e:
            log.error("get_doc_info failed for %r: %s", doc_id, e)
            return {"doc_id": doc_id, "chunk_count": "?"}

    def enrich_doc(self, doc_id: str, overwrite: bool = False) -> dict:
        """Run knowledge extraction on all chunks for *doc_id*.

        Intended to be called in a background thread (slow LLM calls).
        Returns dict with keys ``enriched``, ``skipped``, ``failed``.
        """
        try:
            return self._client.enrich_doc(doc_id, overwrite=overwrite)
        except Exception as e:
            log.error("enrich_doc failed for %r: %s", doc_id, e, exc_info=True)
            return {"enriched": 0, "skipped": 0, "failed": -1, "error": str(e)}
