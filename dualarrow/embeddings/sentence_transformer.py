from __future__ import annotations

import asyncio

import numpy as np

from dualarrow.errors import AdapterUnavailableError


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model: str, dimensions: int | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Install the local-ml extra: pip install 'dualarrow-rag[local-ml]'"
            ) from exc
        self._model = SentenceTransformer(model)
        inferred = self._model.get_sentence_embedding_dimension()
        self.dimensions = dimensions or int(inferred or 0)
        if not self.dimensions:
            raise ValueError("Unable to determine embedding dimensions")

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        result = await asyncio.to_thread(
            self._model.encode,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(result, dtype=np.float32)

    async def embed_query(self, text: str) -> np.ndarray:
        return (await self.embed_documents([text]))[0]
