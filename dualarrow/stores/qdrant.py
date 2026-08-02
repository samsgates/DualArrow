from __future__ import annotations

import uuid

import numpy as np

from dualarrow.errors import AdapterUnavailableError
from dualarrow.models import EvidenceCandidate, KnowledgeUnit, SourceDocument
from dualarrow.utils import stable_id


class QdrantVectorAdapter:
    def __init__(
        self,
        *,
        url: str,
        collection: str,
        dimensions: int,
        api_key: str | None = None,
        prefer_grpc: bool = False,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client import models
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Install the Qdrant extra: pip install 'dualarrow-rag[qdrant]'"
            ) from exc
        self.models = models
        self.client = AsyncQdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        self.collection = collection
        self.dimensions = dimensions

    async def initialize(self) -> None:
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=self.models.VectorParams(
                    size=self.dimensions, distance=self.models.Distance.COSINE
                ),
            )

    async def close(self) -> None:
        await self.client.close()

    @staticmethod
    def _point_id(unit_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"dualarrow:{unit_id}"))

    async def replace_document(
        self, document: SourceDocument, units: list[KnowledgeUnit], embeddings: np.ndarray
    ) -> None:
        await self.delete_document(document.document_id)
        points = []
        for unit, vector in zip(units, embeddings, strict=True):
            payload = unit.model_dump(mode="json")
            payload["content_hash"] = unit.checksum
            points.append(
                self.models.PointStruct(
                    id=self._point_id(unit.unit_id),
                    vector=np.asarray(vector, dtype=np.float32).tolist(),
                    payload=payload,
                )
            )
        if points:
            await self.client.upsert(
                collection_name=self.collection, points=points, wait=True
            )


    async def delete_document(self, document_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection,
            points_selector=self.models.FilterSelector(
                filter=self.models.Filter(
                    must=[
                        self.models.FieldCondition(
                            key="document_id",
                            match=self.models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def delete_documents_not_in(self, document_ids: set[str]) -> int:
        # Cross-index deletion is intentionally driven by canonical document events.
        # A full orphan sweep is provider-specific and exposed through maintenance tooling.
        return 0

    async def search(self, vector: np.ndarray, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        conditions = []
        for key, value in filters.items():
            conditions.append(
                self.models.FieldCondition(key=f"metadata.{key}", match=self.models.MatchValue(value=value))
            )
        result = await self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            query_filter=self.models.Filter(must=conditions) if conditions else None,
            limit=top_k,
            with_payload=True,
        )
        evidence = []
        for point in result.points:
            payload = point.payload or {}
            evidence.append(
                EvidenceCandidate(
                    evidence_id=stable_id(str(payload.get("unit_id")), prefix="ev_"),
                    unit_id=str(payload.get("unit_id")),
                    document_id=str(payload.get("document_id")),
                    content=str(payload.get("content", "")),
                    source_uri=str(payload.get("source_uri", "")),
                    title=str(payload.get("title", "")),
                    heading_path=list(payload.get("heading_path") or []),
                    line_start=payload.get("line_start"),
                    line_end=payload.get("line_end"),
                    retrieval_methods=["dense"],
                    scores={"dense": float(point.score)},
                    normalized_score=float(point.score),
                    authority_score=float(payload.get("authority", 0.75)),
                    freshness_score=1.0,
                    provenance_score=1.0,
                    content_hash=str(payload.get("content_hash", payload.get("checksum", ""))),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return evidence
