from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    dimensions: int

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        ...

    async def embed_query(self, text: str) -> np.ndarray:
        ...
