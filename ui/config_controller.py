"""
Config controller — logic layer for the Config tab.

Responsibilities:
  - Read / write the YAML config file.
  - Reload the in-memory AppConfig so hot-reloadable fields take effect.
  - Expose FIELD_SCHEMA so the UI can render typed, labelled form fields.

Has NO dependency on NiceGUI.
"""

from __future__ import annotations

from utils.config import AppConfig
from utils.logger import AppLogger

log = AppLogger.get(__name__)

# ---------------------------------------------------------------------------
# Field schema — single source of truth for form rendering and reload logic.
#
# Each section entry:
#   "section"  : display title
#   "fields"   : list of (key, label, field_type, reload_mode, description)
#
# field_type   : "str" | "int" | "bool" | "url" | "path" | "json"
#                | "select:val1,val2,..."
# reload_mode  : "hot"     — takes effect immediately after save + reload()
#              | "restart" — requires restarting the dashboard process
# ---------------------------------------------------------------------------
FIELD_SCHEMA: list[dict] = [
    {
        "section": "LLM Server",
        "fields": [
            ("model_provider", "Model Provider",  "select:openai,openrouter", "restart", "Embedding/LLM adapter: 'openai' (local) or 'openrouter'"),
            ("embed_base",     "Embed Base URL",  "url",  "restart", "OpenAI-compatible embedding server endpoint"),
            ("embed_model",    "Embed Model",     "str",  "restart", "Model name for embeddings"),
            ("llm_base",       "LLM Base URL",    "url",  "restart", "OpenAI-compatible LLM chat endpoint"),
            ("llm_model",      "LLM Model",       "str",  "restart", "Model name for generation"),
            ("api_key",        "API Key",         "str",  "restart", "Default API key (fallback when embed/llm keys are not set)"),
            ("embed_api_key",  "Embed API Key",   "str",  "restart", "API key for the embedding server (overrides api_key)"),
            ("llm_api_key",    "LLM API Key",     "str",  "restart", "API key for the LLM server (overrides api_key)"),
            ("llm_kwargs",     "LLM Extra Args",  "json", "restart", "Extra kwargs passed to ChatOpenAI (JSON object)"),
        ],
    },
    {
        "section": "Provider / Rate Limit",
        "fields": [
            ("requests_rate_limit", "Requests / Minute", "int", "restart", "Max embedding API calls per 60 s (openrouter only; 0 = unlimited)"),
        ],
    },
    {
        "section": "Vector Store",
        "fields": [
            ("persist_directory",    "Persist Directory", "path", "restart", "Chroma vector store directory"),
            ("setup_rag_collection", "RAG Collection",    "str",  "restart", "Chroma collection name"),
            ("db_url",               "Record Manager DB", "str",  "restart", "SQLAlchemy URL for the record manager"),
            ("batch_limit",          "Batch Limit",       "int",  "restart", "Max batch size for indexing (default 256)"),
            ("upload_dir",           "Upload Directory",  "path", "hot",     "Directory where uploaded files are saved"),
        ],
    },
    {
        "section": "Reranker",
        "fields": [
            ("reranker_type",  "Reranker Type",  "select:cross_encoder,llm,none", "restart", "Which reranker to use"),
            ("reranker_model", "Reranker Model", "str",                            "restart", "Cross-encoder model name"),
        ],
    },
    {
        "section": "Query Expansion",
        "fields": [
            ("query_expansion_enabled", "Enable Expansion", "bool", "hot", "Generate alternative phrasings before retrieval"),
            ("query_expansion_n",       "Expansion Count",  "int",  "hot", "Number of extra phrasings to generate"),
        ],
    },
    {
        "section": "Knowledge / Notion",
        "fields": [
            ("raw_db_path",           "Raw DB Path",    "path", "restart", "SQLite raw storage database path"),
            ("notion_token",          "Notion Token",   "str",  "restart", "Notion integration secret token"),
            ("notion_workspace_id",   "Workspace ID",   "str",  "restart", "Logical Notion workspace name"),
            ("notion_database_id",    "Database ID",    "str",  "restart", "Notion database UUID"),
            ("notion_data_source_id", "Data Source ID", "str",  "restart", "Notion data source UUID"),
        ],
    },
    {
        "section": "Logging",
        "fields": [
            ("log_level",   "Log Level",   "select:DEBUG,INFO,WARNING,ERROR", "hot", "Root log level"),
            ("log_format",  "Log Format",  "str",                             "hot", "Python logging format string"),
            ("log_datefmt", "Date Format", "str",                             "hot", "strftime date format for timestamps"),
        ],
    },
    {
        "section": "Dev / Testing",
        "fields": [
            ("pdf_path",    "PDF Path",    "path", "hot", "Test PDF file path"),
            ("test_query",  "Test Query",  "str",  "hot", "Default test query string"),
            ("test_search", "Test Search", "str",  "hot", "Default search term"),
        ],
    },
]

# Frozensets derived from schema — kept for any external callers.
HOT_FIELDS: frozenset[str] = frozenset(
    f[0] for s in FIELD_SCHEMA for f in s["fields"] if f[3] == "hot"
)
RESTART_FIELDS: frozenset[str] = frozenset(
    f[0] for s in FIELD_SCHEMA for f in s["fields"] if f[3] == "restart"
)


class ConfigController:
    """Reads, validates, and saves AppConfig's YAML backing file."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_raw(self) -> str:
        """Return the raw YAML text of the config file."""
        return self._config.raw_text

    def get_data(self) -> dict:
        """Return a copy of the current in-memory config data."""
        return dict(self._config._data or {})

    def config_path(self) -> str:
        """Return the absolute path of the config file being managed."""
        return self._config._path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, yaml_text: str) -> tuple[bool, str]:
        """Validate *yaml_text*, write it to disk, and reload in-memory config."""
        try:
            self._config.save(yaml_text)
            log.info("Config saved and reloaded: %s", self._config._path)
            return True, "Saved."
        except ValueError as exc:
            log.warning("Config save rejected: %s", exc)
            return False, str(exc)
        except OSError as exc:
            log.error("Config write failed: %s", exc)
            return False, f"File write error: {exc}"

    def save_data(self, data: dict) -> tuple[bool, str]:
        """Merge *data* into current config, serialize to YAML, and save.

        Unknown keys not present in FIELD_SCHEMA are preserved unchanged.
        """
        import yaml as _yaml
        merged = dict(self._config._data or {})
        merged.update(data)
        try:
            text = _yaml.dump(
                merged, allow_unicode=True, default_flow_style=False, sort_keys=False
            )
        except Exception as exc:
            return False, f"Serialization error: {exc}"
        return self.save(text)

    # ------------------------------------------------------------------
    # Diff helpers
    # ------------------------------------------------------------------

    def changed_keys(self, yaml_text: str) -> tuple[list[str], list[str]]:
        """Compare *yaml_text* against current in-memory data."""
        import yaml as _yaml
        try:
            new_data: dict = _yaml.safe_load(yaml_text) or {}
        except Exception:
            return [], []
        return self.changed_keys_from_data(new_data)

    def changed_keys_from_data(self, new_data: dict) -> tuple[list[str], list[str]]:
        """Compare a data dict against current in-memory config.

        Returns:
            (hot_changed, restart_changed) — lists of changed key names.
        """
        old = self._config._data or {}
        changed = [
            k for k in set(new_data) | set(old)
            if new_data.get(k) != old.get(k)
        ]
        hot     = [k for k in changed if k in HOT_FIELDS]
        restart = [k for k in changed if k in RESTART_FIELDS]
        return hot, restart
