from __future__ import annotations

import hashlib
import math

import numpy as np

from dualarrow.utils import tokenize


class HashEmbeddingProvider:
    """Deterministic zero-network embedding for local demos and tests.

    It is not intended to replace a trained embedding model in accuracy-sensitive deployments.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            weight = 1.0 + math.log1p(len(token))
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.stack([self._embed(text) for text in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)
