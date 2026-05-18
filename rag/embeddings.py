"""Custom embedding implementations for non-OpenAI providers."""

from typing import List

import httpx
from langchain_core.embeddings import Embeddings


class OpenRouterEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by the OpenRouter embeddings API.

    OpenRouter exposes an OpenAI-compatible ``/v1/embeddings`` endpoint, but
    ``langchain_openai.OpenAIEmbeddings`` tries to count tokens with tiktoken
    before sending the request, which causes errors for models whose tokeniser
    is not bundled locally.  This lightweight wrapper sends the request
    directly via ``httpx``, bypassing that check.

    Args:
        model:   OpenRouter embedding model slug, e.g. ``"text-embedding-3-small"``.
        api_key: OpenRouter API key (``sk-or-...``).
        base_url: Base URL of the embeddings endpoint.
                  Defaults to ``"https://openrouter.ai/api/v1/embeddings"``.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1/embeddings",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": texts}
        with httpx.Client() as client:
            response = client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
