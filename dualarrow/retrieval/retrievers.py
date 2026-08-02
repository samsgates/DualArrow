from __future__ import annotations

from typing import Protocol

from dualarrow.embeddings.base import EmbeddingProvider
from dualarrow.models import EvidenceCandidate
from dualarrow.stores.base import KnowledgeStore


class Retriever(Protocol):
    name: str

    async def retrieve(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        ...


class ExactRetriever:
    name = "exact"

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    async def retrieve(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        return await self.store.exact_search(query, top_k, filters)


class BM25Retriever:
    name = "bm25"

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    async def retrieve(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        return await self.store.lexical_search(query, top_k, filters)


class DenseRetriever:
    name = "dense"

    def __init__(self, store: KnowledgeStore, embeddings: EmbeddingProvider) -> None:
        self.store = store
        self.embeddings = embeddings

    async def retrieve(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        vector = await self.embeddings.embed_query(query)
        return await self.store.dense_search(vector, top_k, filters)


class GraphRetriever:
    name = "graph"

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    async def retrieve(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        return await self.store.graph_search(query, top_k, filters)
