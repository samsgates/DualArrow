from __future__ import annotations

from enum import StrEnum
from typing import Protocol

import numpy as np

from dualarrow.models import EvidenceCandidate, KnowledgeUnit, SourceDocument


class StoreCapability(StrEnum):
    DOCUMENTS = "documents"
    DENSE_VECTOR = "dense_vector"
    SPARSE_VECTOR = "sparse_vector"
    MULTI_VECTOR = "multi_vector"
    BM25 = "bm25"
    FULL_TEXT = "full_text"
    METADATA_FILTER = "metadata_filter"
    GRAPH_TRAVERSAL = "graph_traversal"
    RDF = "rdf"
    SPARQL = "sparql"
    CYPHER = "cypher"
    TIME_SERIES = "time_series"
    TRANSACTIONS = "transactions"
    ACCESS_CONTROL = "access_control"


class KnowledgeStore(Protocol):
    capabilities: set[StoreCapability]

    async def initialize(self) -> None: ...
    async def close(self) -> None: ...

    async def replace_document(
        self,
        document: SourceDocument,
        units: list[KnowledgeUnit],
        embeddings: np.ndarray,
        links: list[str],
    ) -> None: ...

    async def delete_documents_not_in(self, document_ids: set[str]) -> int: ...
    async def document_checksum(self, document_id: str) -> str | None: ...
    async def list_document_ids(self) -> set[str]: ...
    async def exact_search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]: ...
    async def lexical_search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]: ...
    async def dense_search(self, vector: np.ndarray, top_k: int, filters: dict) -> list[EvidenceCandidate]: ...
    async def graph_search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]: ...
    async def get_units(self, unit_ids: list[str]) -> list[KnowledgeUnit]: ...
