from __future__ import annotations

import httpx
import numpy as np


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None,
        dimensions: int,
        timeout: float = 60,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        response = await self._client.post(
            "/embeddings", json={"model": self.model, "input": texts}
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
        if vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {vectors.shape[1]}"
            )
        return vectors

    async def embed_query(self, text: str) -> np.ndarray:
        return (await self.embed_documents([text]))[0]

    async def close(self) -> None:
        await self._client.aclose()
