"""
LLM configuration smoke test.

Verifies that the configured embedding server and LLM endpoint respond
correctly.  Designed to work with local llama.cpp / Ollama servers AND
cloud providers such as OpenRouter (https://openrouter.ai/api/v1).

OpenRouter setup (etc/config.yaml):
    embed_base:   "http://localhost:8080/v1/"   # local embedding server
    llm_base:     "https://openrouter.ai/api/v1"
    embed_model:  "text-embedding-ada-002"      # or whatever your server exposes
    llm_model:    "anthropic/claude-3-haiku"    # or any OpenRouter model slug
    api_key:      "sk-no-key-required"          # used as fallback
    llm_api_key:  "sk-or-..."                   # OpenRouter key
    # embed_api_key is omitted → falls back to api_key

Tests:
    1  AppConfig loads and exposes embed / LLM settings correctly
    2  Embedding server reachability  (SKIP if unreachable)
    3  Embed a short text → returns a non-empty float vector
    4  LLM server reachability        (SKIP if unreachable)
    5  Single-turn chat completion    → returns a non-empty string response

Usage:
    python testing/testing_llm_config.py
"""

import socket
import sys
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.embeddings import OpenRouterEmbeddings
from utils.config import AppConfig
from utils.logger import AppLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _server_up(base_url: str, timeout: float = 3.0) -> bool:
    """Return True if the TCP port in *base_url* is reachable."""
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _skip(msg: str) -> None:
    print(f"  [SKIP] {msg}")


def _fail(msg: str, exc: BaseException | None = None) -> None:
    print(f"  [FAIL] {msg}")
    if exc:
        print(f"         {type(exc).__name__}: {exc}")
    sys.exit(1)


def _check(condition: bool, msg: str) -> None:
    if condition:
        _ok(msg)
    else:
        _fail(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_config(config: AppConfig) -> None:
    _section("Test 1: AppConfig settings")
    print(f"  model_provider = {config.model_provider!r}")
    print(f"  embed_base    = {config.embed_base!r}")
    print(f"  embed_model   = {config.embed_model!r}")
    print(f"  embed_api_key = {'(set)' if config.embed_api_key != 'sk-no-key-required' else '(default)'}")
    print(f"  llm_base      = {config.llm_base!r}")
    print(f"  llm_model     = {config.llm_model!r}")
    print(f"  llm_api_key   = {'(set)' if config.llm_api_key != 'sk-no-key-required' else '(default)'}")
    _check(bool(config.embed_base), "embed_base is configured")
    _check(bool(config.llm_base),   "llm_base is configured")
    _check(bool(config.embed_model), "embed_model is configured")
    _check(bool(config.llm_model),   "llm_model is configured")
    _check(config.model_provider in ("openai", "openrouter"), "model_provider is a known value")


def test_embeddings(config: AppConfig) -> None:
    _section("Test 2–3: Embeddings")

    if not _server_up(config.embed_base):
        _skip(f"embed server not reachable at {config.embed_base!r} — skipping tests 2–3")
        return

    _ok(f"embed server reachable: {config.embed_base!r}")

    if config.model_provider == "openrouter":
        embed = OpenRouterEmbeddings(
            model=config.embed_model,
            api_key=config.embed_api_key,
            base_url=config.embed_base,
        )
        print(f"  using OpenRouterEmbeddings")
    else:
        embed = OpenAIEmbeddings(
            openai_api_key=config.embed_api_key,
            openai_api_base=config.embed_base,
            model=config.embed_model,
        )
        print(f"  using OpenAIEmbeddings")

    try:
        vector = embed.embed_query("Hello, this is a connectivity test.")
    except Exception as exc:
        _fail("embed_query() raised an exception", exc)
        return  # unreachable, keeps type-checker happy

    _check(isinstance(vector, list) and len(vector) > 0, "embed_query returns a list")
    _check(all(isinstance(v, float) for v in vector[:5]), "vector elements are floats")
    print(f"  vector dim = {len(vector)}, first 5 values = {[round(v, 4) for v in vector[:5]]}")
    _ok(f"embed_query succeeded (dim={len(vector)})")


def test_llm(config: AppConfig) -> None:
    _section("Test 4–5: LLM chat completion")

    if not _server_up(config.llm_base):
        _skip(f"LLM server not reachable at {config.llm_base!r} — skipping tests 4–5")
        return

    _ok(f"LLM server reachable: {config.llm_base!r}")

    llm = ChatOpenAI(
        base_url=config.llm_base,
        api_key=config.llm_api_key,
        model=config.llm_model,
        **config.llm_kwargs,
    )

    prompt = "Reply with exactly one sentence: what is 2 + 2?"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
    except Exception as exc:
        _fail("llm.invoke() raised an exception", exc)
        return

    content = response.content if hasattr(response, "content") else str(response)
    _check(isinstance(content, str) and len(content) > 0, "LLM returned a non-empty response")
    print(f"  prompt   : {prompt!r}")
    print(f"  response : {content!r}")
    _ok("LLM chat completion succeeded")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    AppLogger.setup(level="WARNING")
    config = AppConfig()

    test_config(config)
    test_embeddings(config)
    test_llm(config)

    print("\nAll checks completed.")


if __name__ == "__main__":
    main()
