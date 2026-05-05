import os

from utils.file_processor import read_yaml


class AppConfig:
    """
    Loads application settings from a YAML file and exposes them as typed properties.
    Centralises all configuration defaults so no other module hard-codes them.
    """

    _DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "etc",
        "config.yaml",
    )

    def __init__(self, path: str = _DEFAULT_CONFIG_PATH):
        self._data = read_yaml(path)

    # --- Embedding server ---

    @property
    def embed_base(self) -> str:
        return self._data.get("embed_base", "http://localhost:8080/v1/")

    @property
    def embed_model(self) -> str:
        return self._data.get("embed_model", "text-embedding-ada-002")

    # --- LLM server ---

    @property
    def llm_base(self) -> str:
        return self._data.get("llm_base", "http://localhost:8080/v1/")

    @property
    def llm_model(self) -> str:
        return self._data.get("llm_model", "local-model")

    @property
    def llm_kwargs(self) -> dict:
        return self._data.get("llm_kwargs", {})

    # --- Vector store ---

    @property
    def persist_directory(self) -> str:
        return self._data.get("persist_directory", "./my_db")

    @property
    def db_url(self) -> str:
        return self._data.get("db_url", "./my_db/record_manager_cache.sql")

    @property
    def setup_rag_collection(self):
        return self._data.get("setup_rag_collection", False)

    @property
    def batch_limit(self) -> int:
        """Batch size limit used when indexing or writing to the vector store.

        Defaults to 256; override in `etc/config.yaml` with an integer value.
        """
        return int(self._data.get("batch_limit", 256))

    # --- Reranker ---

    @property
    def reranker_type(self) -> str:
        """Which reranker to use: 'cross_encoder', 'llm', or 'none'."""
        return self._data.get("reranker_type", "cross_encoder")

    @property
    def reranker_model(self) -> str:
        """Cross-encoder model name (only used when reranker_type='cross_encoder')."""
        return self._data.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- Query expansion ---

    @property
    def query_expansion_enabled(self) -> bool:
        """When True, answer_query generates extra query phrasings before retrieval."""
        return bool(self._data.get("query_expansion_enabled", False))

    @property
    def query_expansion_n(self) -> int:
        """Number of extra phrasings to generate (total candidates = n+1 including original)."""
        return int(self._data.get("query_expansion_n", 3))

    # --- Auth ---

    @property
    def api_key(self) -> str:
        return self._data.get("api_key", "sk-no-key-required")

    # --- Runtime / testing ---

    @property
    def pdf_path(self) -> str:
        return self._data.get("pdf_path", "./data/test.pdf")

    @property
    def test_query(self) -> str:
        return self._data.get("test_query", "What are the main contents of this document?")

    @property
    def test_search(self) -> str:
        return self._data.get("test_search", "test")
