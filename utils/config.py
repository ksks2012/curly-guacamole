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

    def __init__(self, path: str | None = None):
        if path is None:
            # When the package is pip-installed, __file__ resolves to site-packages
            # and _DEFAULT_CONFIG_PATH no longer exists. Fall back to CWD/etc/config.yaml
            # which is correct when running from the project root.
            path = (
                self._DEFAULT_CONFIG_PATH
                if os.path.exists(self._DEFAULT_CONFIG_PATH)
                else os.path.join(os.getcwd(), "etc", "config.yaml")
            )
        self._path = os.path.abspath(path)
        # Project root = two levels above the config file (etc/../)
        self._root = os.path.dirname(os.path.dirname(self._path))
        self._data = read_yaml(path)

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Construct an AppConfig from an explicit YAML path."""
        return cls(path=path)

    # ------------------------------------------------------------------
    # Runtime update
    # ------------------------------------------------------------------

    @property
    def raw_text(self) -> str:
        """Return the raw content of the config file as a string."""
        with open(self._path, "r", encoding="utf-8") as f:
            return f.read()

    def reload(self) -> None:
        """Re-read the config file from disk and update in-memory data.

        Properties that call ``self._data.get(...)`` on every access
        (e.g. ``query_expansion_enabled``, ``upload_dir``) become effective
        immediately.  Properties used to build LLM / Chroma objects at
        ``LocalLlamaClient.__init__`` time require a full dashboard restart.
        """
        self._data = read_yaml(self._path)

    def save(self, yaml_text: str) -> None:
        """Validate *yaml_text* as YAML, write it to disk, then call reload().

        Raises:
            ValueError: If *yaml_text* is not valid YAML.
            OSError:    If the file cannot be written.
        """
        import yaml as _yaml  # local import — only needed here
        try:
            parsed = _yaml.safe_load(yaml_text)
        except _yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Config must be a YAML mapping, not a scalar or list")
        with open(self._path, "w", encoding="utf-8") as f:
            f.write(yaml_text)
        self.reload()

    def _abspath(self, raw: str) -> str:
        """Return an absolute path, resolving relative paths against the project root."""
        if os.path.isabs(raw):
            return raw
        return os.path.normpath(os.path.join(self._root, raw))

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
    def upload_dir(self) -> str:
        """Absolute directory where uploaded PDF files are stored."""
        return self._abspath(self._data.get("upload_dir", "/tmp"))

    @property
    def persist_directory(self) -> str:
        return self._data.get("persist_directory", "./my_db")

    @property
    def db_url(self) -> str:
        """SQLAlchemy-compatible URL for the record manager database.

        If the configured value is a plain file path (no scheme), it is
        automatically prefixed with 'sqlite:///' so SQLAlchemy can parse it.
        """
        raw = self._data.get("db_url", "./my_db/record_manager_cache.sql")
        if "://" not in raw:
            raw = "sqlite:///" + raw
        return raw

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

    # --- Knowledge / Notion ---

    @property
    def raw_db_path(self) -> str:
        """Path to the SQLite raw storage database (Step 0.3)."""
        return self._abspath(self._data.get("raw_db_path", "./my_db/raw.db"))

    @property
    def notion_token(self) -> str:
        """Notion integration secret token (empty string = Notion disabled)."""
        return self._data.get("notion_token", "")

    @property
    def notion_workspace_id(self) -> str:
        """Logical workspace name used when creating a Workspace model."""
        return self._data.get("notion_workspace_id", "")

    @property
    def notion_database_id(self) -> str:
        """Notion database UUID (required by NotionClient.get_database())."""
        return self._data.get("notion_database_id", "")

    @property
    def notion_data_source_id(self) -> str:
        """Notion data source UUID.

        When set, the sync pipeline queries pages via
        POST /v1/data_sources/{id}/query directly, skipping the database
        lookup step.  Obtain it from ``NotionClient.get_database()``'s
        ``data_sources[0]["id"]`` field.
        """
        return self._data.get("notion_data_source_id", "")

    # --- Logging ---

    @property
    def log_level(self) -> str:
        """Root log level. Override in config.yaml with: log_level: DEBUG"""
        return self._data.get("log_level", "INFO")

    @property
    def log_format(self) -> str:
        """Python logging format string."""
        return self._data.get(
            "log_format",
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    @property
    def log_datefmt(self) -> str:
        """strftime date format for the asctime field."""
        return self._data.get("log_datefmt", "%H:%M:%S")

    # --- Memory (Stage C.1) ---

    @property
    def memory_db_path(self) -> str:
        """Absolute path to the conversation memory SQLite database."""
        raw = self._data.get("memory_db_path", "./my_db/memory.db")
        return self._abspath(raw)

    @property
    def memory_default_session(self) -> str:
        """Default session_id used when no session is explicitly set."""
        return self._data.get("memory_default_session", "default")

    @property
    def memory_max_recent(self) -> int:
        """Maximum Q-A turns to keep in recent_questions."""
        return int(self._data.get("memory_max_recent", 20))

    @property
    def memory_max_topics(self) -> int:
        """Maximum topic tags to keep in current_topics."""
        return int(self._data.get("memory_max_topics", 10))

    @property
    def memory_extract_topics(self) -> bool:
        """When True, call LLM to extract topic tags after each Q-A turn."""
        return bool(self._data.get("memory_extract_topics", True))

    @property
    def memory_auto_infer_project(self) -> bool:
        """When True, periodically infer active_project from recent conversation."""
        return bool(self._data.get("memory_auto_infer_project", True))
