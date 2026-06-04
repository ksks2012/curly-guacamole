"""Custom embedding implementations for non-OpenAI providers."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import httpx
from langchain_core.embeddings import Embeddings

from rag.rate_limiter import RateLimiter  # noqa: F401 — re-exported for back-compat


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
        base_url:             Base URL of the OpenAI-compatible API server,
                              e.g. ``"https://openrouter.ai/api/v1"``.
                              The ``/embeddings`` path is appended automatically,
                              matching the behaviour of ``OpenAIEmbeddings``.
                              Defaults to ``"https://openrouter.ai/api/v1"``.
        requests_per_minute:  Maximum API calls per 60-second rolling window.
                              ``0`` (default) means unlimited.
        batch_size:           Number of texts per individual HTTP request when
                              ``embed_documents`` splits a large input.
                              Defaults to 64.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        requests_per_minute: int = 0,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.api_key = api_key
        # Normalise base_url the same way OpenAIEmbeddings does: strip trailing
        # slash, then append /embeddings.  This means embed_base in config.yaml
        # always means the API root (e.g. https://openrouter.ai/api/v1) for
        # both providers — no manual /embeddings suffix required.
        self.url = base_url.rstrip("/") + "/embeddings"
        self._batch_size = batch_size
        self._limiter = RateLimiter(requests_per_minute)
        # Derive worker count from rate limit so we can saturate it with
        # overlapping I/O without spawning an excessive number of threads.
        # rpm=0 (unlimited) → 4 workers as a sensible default.
        # rpm>0             → rpm // 5, clamped to [1, 8].
        if requests_per_minute > 0:
            self._max_workers: int = min(max(1, requests_per_minute // 5), 8)
        else:
            self._max_workers = 4

    def _post_batch(self, texts: List[str]) -> List[List[float]]:
        """Acquire a rate-limit slot then POST one embedding batch.
        
        Note: empty or whitespace-only strings are replaced with a single space
        to avoid API validation errors.
        """
        # Replace empty/whitespace-only strings with a single space (API requirement)
        texts = [t if t.strip() else " " for t in texts]
        
        self._limiter.acquire()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client() as client:
            response = client.post(
                self.url,
                headers=headers,
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            body = response.json()
            if "data" not in body:
                raise ValueError(
                    f"Unexpected embedding API response (missing 'data'): {body}"
                )
            return [item["embedding"] for item in body["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Split into sub-batches so we can dispatch them in parallel.
        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]

        # Fast path: single sub-batch — no thread overhead.
        if len(batches) == 1:
            return self._post_batch(batches[0])

        # Parallel path: dispatch all sub-batches concurrently.
        # RateLimiter.acquire() inside _post_batch is thread-safe and ensures
        # we never exceed requests_per_minute across all workers.
        results: List[Optional[List[List[float]]]] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(self._post_batch, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()  # re-raises any exception

        return [vec for batch_vecs in results for vec in batch_vecs]  # type: ignore[arg-type]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

