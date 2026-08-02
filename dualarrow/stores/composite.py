from __future__ import annotations

import asyncio

import numpy as np

from dualarrow.models import EvidenceCandidate, KnowledgeUnit, SourceDocument
from dualarrow.stores.arrowmd import ArrowMDStore
from dualarrow.stores.external import GraphIndexAdapter, LexicalIndexAdapter, VectorIndexAdapter


class CompositeKnowledgeStore:
    """Canonical ArrowMD store plus optional specialized external indexes."""

    def __init__(
        self,
        canonical: ArrowMDStore,
        *,
        vector: VectorIndexAdapter | None = None,
        lexical: LexicalIndexAdapter | None = None,
        graph: GraphIndexAdapter | None = None,
    ) -> None:
        self.canonical = canonical
        self.vector = vector
        self.lexical = lexical
        self.graph = graph
        self.capabilities = set(canonical.capabilities)

    async def initialize(self) -> None:
        adapters = [self.canonical, self.vector, self.lexical, self.graph]
        await asyncio.gather(*(adapter.initialize() for adapter in adapters if adapter is not None))

    async def close(self) -> None:
        adapters = [self.canonical, self.vector, self.lexical, self.graph]
        await asyncio.gather(*(adapter.close() for adapter in adapters if adapter is not None))

    async def replace_document(
        self,
        document: SourceDocument,
        units: list[KnowledgeUnit],
        embeddings: np.ndarray,
        links: list[str],
    ) -> None:
        tasks = []
        if self.vector:
            tasks.append(self.vector.replace_document(document, units, embeddings))
        if self.lexical:
            tasks.append(self.lexical.replace_document(document, units))
        if self.graph:
            tasks.append(self.graph.replace_document(document, units, links))
        if tasks:
            await asyncio.gather(*tasks)
        # Commit the canonical checksum last. A failed external update will therefore
        # be retried on the next incremental ingestion run.
        await self.canonical.replace_document(document, units, embeddings, links)

    async def delete_documents_not_in(self, document_ids: set[str]) -> int:
        existing = await self.canonical.list_document_ids()
        removed = existing - document_ids
        tasks = []
        for adapter in (self.vector, self.lexical, self.graph):
            if adapter:
                for document_id in removed:
                    tasks.append(adapter.delete_document(document_id))
        if tasks:
            await asyncio.gather(*tasks)
        return await self.canonical.delete_documents_not_in(document_ids)

    async def document_checksum(self, document_id: str) -> str | None:
        return await self.canonical.document_checksum(document_id)

    async def list_document_ids(self) -> set[str]:
        return await self.canonical.list_document_ids()

    async def exact_search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        return await self.canonical.exact_search(query, top_k, filters)

    async def lexical_search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        if self.lexical:
            return await self.lexical.search(query, top_k, filters)
        return await self.canonical.lexical_search(query, top_k, filters)

    async def dense_search(self, vector: np.ndarray, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        if self.vector:
            return await self.vector.search(vector, top_k, filters)
        return await self.canonical.dense_search(vector, top_k, filters)

    async def graph_search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        if self.graph:
            return await self.graph.search(query, top_k, filters)
        return await self.canonical.graph_search(query, top_k, filters)

    async def get_units(self, unit_ids: list[str]) -> list[KnowledgeUnit]:
        return await self.canonical.get_units(unit_ids)
