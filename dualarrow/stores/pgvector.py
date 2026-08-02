from __future__ import annotations


import numpy as np

from dualarrow.errors import AdapterUnavailableError
from dualarrow.models import EvidenceCandidate, KnowledgeUnit, SourceDocument
from dualarrow.utils import stable_id


class PgVectorAdapter:
    def __init__(self, *, connection: str, table: str, dimensions: int) -> None:
        try:
            from pgvector.sqlalchemy import Vector
            from sqlalchemy import JSON, Float, Integer, String, Text, delete, select
            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
            from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Install the Postgres extra: pip install 'dualarrow-rag[postgres]'"
            ) from exc

        class Base(DeclarativeBase):
            pass

        class VectorUnit(Base):
            __tablename__ = table
            unit_id: Mapped[str] = mapped_column(String(128), primary_key=True)
            document_id: Mapped[str] = mapped_column(String(128), index=True)
            view_id: Mapped[str] = mapped_column(String(128), index=True)
            content: Mapped[str] = mapped_column(Text)
            source_uri: Mapped[str] = mapped_column(Text)
            checksum: Mapped[str] = mapped_column(String(64))
            title: Mapped[str] = mapped_column(Text)
            heading_path: Mapped[list] = mapped_column(JSON)
            line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
            line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
            authority: Mapped[float] = mapped_column(Float)
            metadata_json: Mapped[dict] = mapped_column(JSON)
            embedding: Mapped[list[float]] = mapped_column(Vector(dimensions))

        self.Base = Base
        self.VectorUnit = VectorUnit
        self.select = select
        self.delete = delete
        self.engine = create_async_engine(connection, pool_pre_ping=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
            await connection.run_sync(self.Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def replace_document(
        self, document: SourceDocument, units: list[KnowledgeUnit], embeddings: np.ndarray
    ) -> None:
        await self.delete_document(document.document_id)
        async with self.sessions.begin() as session:
            session.add_all([
                self.VectorUnit(
                    unit_id=unit.unit_id,
                    document_id=unit.document_id,
                    view_id=unit.view_id,
                    content=unit.content,
                    source_uri=unit.source_uri,
                    checksum=unit.checksum,
                    title=unit.title,
                    heading_path=unit.heading_path,
                    line_start=unit.line_start,
                    line_end=unit.line_end,
                    authority=unit.authority,
                    metadata_json=unit.metadata,
                    embedding=np.asarray(vector, dtype=np.float32).tolist(),
                )
                for unit, vector in zip(units, embeddings, strict=True)
            ])


    async def delete_document(self, document_id: str) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                self.delete(self.VectorUnit).where(
                    self.VectorUnit.document_id == document_id
                )
            )

    async def delete_documents_not_in(self, document_ids: set[str]) -> int:
        if not document_ids:
            async with self.sessions.begin() as session:
                result = await session.execute(self.delete(self.VectorUnit))
                return int(result.rowcount or 0)
        async with self.sessions.begin() as session:
            result = await session.execute(
                self.delete(self.VectorUnit).where(
                    self.VectorUnit.document_id.not_in(document_ids)
                )
            )
            return int(result.rowcount or 0)

    async def search(self, vector: np.ndarray, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        distance = self.VectorUnit.embedding.cosine_distance(
            np.asarray(vector, dtype=np.float32).tolist()
        )
        statement = self.select(self.VectorUnit, distance.label("distance"))
        for key, value in filters.items():
            statement = statement.where(self.VectorUnit.metadata_json[key].as_string() == str(value))
        statement = statement.order_by(distance).limit(top_k)
        async with self.sessions() as session:
            rows = (await session.execute(statement)).all()
        output = []
        for row in rows:
            item = row[0]
            score = max(0.0, 1.0 - float(row.distance))
            output.append(
                EvidenceCandidate(
                    evidence_id=stable_id(item.unit_id, prefix="ev_"),
                    unit_id=item.unit_id,
                    document_id=item.document_id,
                    content=item.content,
                    source_uri=item.source_uri,
                    title=item.title,
                    heading_path=list(item.heading_path or []),
                    line_start=item.line_start,
                    line_end=item.line_end,
                    retrieval_methods=["dense"],
                    scores={"dense": score},
                    normalized_score=score,
                    authority_score=item.authority,
                    freshness_score=1.0,
                    provenance_score=1.0,
                    content_hash=item.checksum,
                    metadata=dict(item.metadata_json or {}),
                )
            )
        return output
