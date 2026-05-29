"""Storage adapter for the code-block collection across multiple persist dirs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from langchain_chroma import Chroma

from rag.retrieval.collections import CODE_BLOCK_COLLECTION


@dataclass
class CodeBlockStore:
    """Opens code-block collections while hiding collection naming details."""

    embed: object
    persist_dirs: list[str]
    collection_name: str = CODE_BLOCK_COLLECTION

    def iter_databases(self) -> Iterator[tuple[str, Chroma]]:
        """Yield available code-block Chroma handles in directory priority order."""
        for persist_dir in self.persist_dirs:
            try:
                db = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embed,
                    collection_name=self.collection_name,
                )
            except Exception:
                continue
            yield str(persist_dir), db

    def first_database(self) -> tuple[str, Chroma] | tuple[None, None]:
        """Return the first available code-block DB, or ``(None, None)``."""
        for persist_dir, db in self.iter_databases():
            return persist_dir, db
        return None, None
