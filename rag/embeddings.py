"""Custom embedding implementations for non-OpenAI providers."""

import threading
import time
from collections import deque
from typing import List

import httpx
from langchain_core.embeddings import Embeddings


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Blocks the calling thread until a request slot is available within the
    rolling one-minute window.  Pass ``requests_per_minute=0`` to disable.
    """

    def __init__(self, requests_per_minute: int = 0) -> None:
        self._rpm = requests_per_minute
        self._window = 60.0  # seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request slot is available, then consume it."""
        if self._rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop timestamps outside the rolling window
                while self._timestamps and self._timestamps[0] <= now - self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._rpm:
                    self._timestamps.append(now)
                    return
                # Wait until the oldest slot falls out of the window
                wait = self._timestamps[0] + self._window - now
            time.sleep(max(wait, 0.05))


class OpenRouterEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by the OpenRouter embeddings API.

    OpenRouter exposes an OpenAI-compatible ``/v1/embeddings`` endpoint, but
    ``langchain_openai.OpenAIEmbeddings`` tries to count tokens with tiktoken
    before sending the request, which causes errors for models whose tokeniser
    is not bundled locally.  This lightweight wrapper sends the request
    directly via ``httpx``, bypassing that check.

    Args:
        model:                OpenRouter embedding model slug,
                              e.g. ``"text-embedding-3-small"``.
        api_key:              OpenRouter API key (``sk-or-...``).
        base_url:             Full URL of the embeddings endpoint.
                              Defaults to ``"https://openrouter.ai/api/v1/embeddings"``.
        requests_per_minute:  Maximum API calls per 60-second rolling window.
                              ``0`` (default) means unlimited.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1/embeddings",
        requests_per_minute: int = 0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.url = base_url
        self._limiter = RateLimiter(requests_per_minute)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._limiter.acquire()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": texts}
        with httpx.Client() as client:
            response = client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

