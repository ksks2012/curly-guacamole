"""
OpenRouter integration smoke test.

Verifies that the OpenRouter API responds correctly for both the embedding
endpoint and the chat completion endpoint, using credentials from the
active AppConfig (etc/config.yaml).

Required config (etc/config.yaml):
    model_provider: "openrouter"
    embed_base:     "https://openrouter.ai/api/v1/embeddings"
    embed_model:    "text-embedding-3-small"   # or any OpenRouter embed model
    llm_base:       "https://openrouter.ai/api/v1"
    llm_model:      "anthropic/claude-3-haiku" # or any OpenRouter chat model
    llm_api_key:    "sk-or-..."
    # embed_api_key falls back to llm_api_key when omitted

Tests:
    1  Config declares model_provider = "openrouter"
    2  OpenRouter HTTPS endpoint is reachable
    3  embed_query() returns a non-empty float vector
    4  embed_documents() returns one vector per input text
    5  ChatOpenAI.invoke() via OpenRouter returns a non-empty string

Usage:
    python testing/testing_openrouter.py
"""

import socket
import sys
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from rag.embeddings import OpenRouterEmbeddings
from utils.config import AppConfig
from utils.logger import AppLogger

OPENROUTER_HOST = "openrouter.ai"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _host_up(host: str, port: int = 443, timeout: float = 5.0) -> bool:
    try:
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

def test_provider_config(config: AppConfig) -> None:
    _section("Test 1: model_provider is openrouter")
    print(f"  model_provider = {config.model_provider!r}")
    print(f"  embed_base     = {config.embed_base!r}")
    print(f"  embed_model    = {config.embed_model!r}")
    print(f"  llm_base       = {config.llm_base!r}")
    print(f"  llm_model      = {config.llm_model!r}")
    _check(config.model_provider == "openrouter", "model_provider == 'openrouter'")


def test_reachability(config: AppConfig) -> bool:
    """Return True if OpenRouter is reachable; tests are skipped otherwise."""
    _section("Test 2: OpenRouter reachability")
    up = _host_up(OPENROUTER_HOST)
    if up:
        _ok(f"{OPENROUTER_HOST}:443 is reachable")
    else:
        _skip(f"{OPENROUTER_HOST}:443 not reachable — skipping tests 3-5")
    return up


def test_embed_query(embed: OpenRouterEmbeddings) -> None:
    _section("Test 3: embed_query()")
    try:
        vector = embed.embed_query("OpenRouter embedding connectivity test")
    except Exception as exc:
        _fail("embed_query() raised an exception", exc)
        return

    _check(isinstance(vector, list) and len(vector) > 0, "returns a non-empty list")
    _check(all(isinstance(v, float) for v in vector[:5]), "elements are floats")
    print(f"  dim={len(vector)}, first 5: {[round(v, 6) for v in vector[:5]]}")
    _ok(f"embed_query succeeded (dim={len(vector)})")


def test_embed_documents(embed: OpenRouterEmbeddings) -> None:
    _section("Test 4: embed_documents() — batch input")
    texts = ["first sentence", "second sentence", "third sentence"]
    try:
        vectors = embed.embed_documents(texts)
    except Exception as exc:
        _fail("embed_documents() raised an exception", exc)
        return

    _check(len(vectors) == len(texts), f"returns {len(texts)} vectors")
    _check(all(len(v) == len(vectors[0]) for v in vectors), "all vectors have equal dimension")
    print(f"  batch size={len(vectors)}, dim={len(vectors[0])}")
    _ok("embed_documents succeeded")


def test_llm_chat(config: AppConfig) -> None:
    _section("Test 5: ChatOpenAI via OpenRouter")
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
    _check(isinstance(content, str) and len(content) > 0, "response is a non-empty string")
    print(f"  prompt   : {prompt!r}")
    print(f"  response : {content!r}")
    _ok("LLM chat completion succeeded")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    AppLogger.setup(level="WARNING")
    config = AppConfig()

    test_provider_config(config)

    if not test_reachability(config):
        print("\nAll reachable checks skipped (no network).")
        return

    embed = OpenRouterEmbeddings(
        model=config.embed_model,
        api_key=config.embed_api_key,
        base_url=config.embed_base,
    )

    test_embed_query(embed)
    test_embed_documents(embed)
    test_llm_chat(config)

    print("\nAll checks completed.")


if __name__ == "__main__":
    main()

