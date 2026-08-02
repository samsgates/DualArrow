from __future__ import annotations

from dualarrow.errors import AdapterUnavailableError
from dualarrow.models import EvidenceCandidate, KnowledgeUnit, SourceDocument
from dualarrow.utils import stable_id, tokenize


class Neo4jGraphAdapter:
    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Install the Neo4j extra: pip install 'dualarrow-rag[neo4j]'"
            ) from exc
        self.driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        self.database = database

    async def initialize(self) -> None:
        await self.driver.verify_connectivity()
        async with self.driver.session(database=self.database) as session:
            await session.run(
                "CREATE CONSTRAINT dualarrow_doc IF NOT EXISTS "
                "FOR (d:DualArrowDocument) REQUIRE d.document_id IS UNIQUE"
            )
            await session.run(
                "CREATE CONSTRAINT dualarrow_unit IF NOT EXISTS "
                "FOR (u:DualArrowUnit) REQUIRE u.unit_id IS UNIQUE"
            )

    async def close(self) -> None:
        await self.driver.close()

    async def replace_document(
        self, document: SourceDocument, units: list[KnowledgeUnit], links: list[str]
    ) -> None:
        async with self.driver.session(database=self.database) as session:
            await session.execute_write(self._replace_tx, document, units, links)

    @staticmethod
    async def _replace_tx(tx, document: SourceDocument, units: list[KnowledgeUnit], links: list[str]):
        await tx.run(
            "MATCH (d:DualArrowDocument {document_id:$id}) DETACH DELETE d",
            id=document.document_id,
        )
        await tx.run(
            "CREATE (d:DualArrowDocument {document_id:$id,title:$title,source_uri:$uri,"
            "authority:$authority,checksum:$checksum,metadata:$metadata})",
            id=document.document_id,
            title=document.title,
            uri=document.source_uri,
            authority=document.authority,
            checksum=document.checksum,
            metadata=document.metadata,
        )
        if units:
            await tx.run(
                """
                MATCH (d:DualArrowDocument {document_id:$document_id})
                UNWIND $units AS item
                CREATE (u:DualArrowUnit {
                    unit_id:item.unit_id, document_id:item.document_id, content:item.content,
                    title:item.title, source_uri:item.source_uri, checksum:item.checksum,
                    heading_path:item.heading_path, line_start:item.line_start,
                    line_end:item.line_end, authority:item.authority, metadata:item.metadata
                })
                CREATE (d)-[:CONTAINS]->(u)
                """,
                document_id=document.document_id,
                units=[unit.model_dump(mode="json") for unit in units],
            )
        if links:
            await tx.run(
                """
                MATCH (d:DualArrowDocument {document_id:$document_id})
                UNWIND $links AS target
                MERGE (t:DualArrowTarget {key:target})
                MERGE (d)-[:LINKS_TO]->(t)
                """,
                document_id=document.document_id,
                links=links,
            )


    async def delete_document(self, document_id: str) -> None:
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                "MATCH (d:DualArrowDocument {document_id:$id}) DETACH DELETE d",
                id=document_id,
            )
            await result.consume()

    async def delete_documents_not_in(self, document_ids: set[str]) -> int:
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                "MATCH (d:DualArrowDocument) WHERE NOT d.document_id IN $ids "
                "WITH collect(d) AS docs, count(d) AS count FOREACH (d IN docs | DETACH DELETE d) "
                "RETURN count",
                ids=list(document_ids),
            )
            record = await result.single()
            return int(record["count"] if record else 0)

    async def search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        tokens = [token for token in tokenize(query) if len(token) > 2][:12]
        if not tokens:
            return []
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (d:DualArrowDocument)-[:CONTAINS]->(u:DualArrowUnit)
                WITH d,u, reduce(score=0.0, token IN $tokens |
                    score + CASE WHEN toLower(u.content) CONTAINS token THEN 1.0 ELSE 0.0 END
                          + CASE WHEN toLower(u.title) CONTAINS token THEN 1.5 ELSE 0.0 END) AS direct
                WHERE direct > 0
                OPTIONAL MATCH (d)-[:LINKS_TO]->(target:DualArrowTarget)<-[:LINKS_TO]-(related:DualArrowDocument)
                OPTIONAL MATCH (related)-[:CONTAINS]->(ru:DualArrowUnit)
                WITH u, direct, collect(DISTINCT ru)[0..2] AS related_units
                UNWIND ([u] + related_units) AS candidate
                WITH candidate, max(direct) AS score
                WHERE candidate IS NOT NULL
                RETURN candidate, score ORDER BY score DESC LIMIT $limit
                """,
                tokens=tokens,
                limit=top_k,
            )
            rows = await result.data()
        output = []
        for row in rows:
            item = dict(row["candidate"])
            score = float(row["score"])
            output.append(
                EvidenceCandidate(
                    evidence_id=stable_id(item["unit_id"], prefix="ev_"),
                    unit_id=item["unit_id"],
                    document_id=item["document_id"],
                    content=item["content"],
                    source_uri=item["source_uri"],
                    title=item.get("title", ""),
                    heading_path=list(item.get("heading_path") or []),
                    line_start=item.get("line_start"),
                    line_end=item.get("line_end"),
                    retrieval_methods=["graph"],
                    scores={"graph": score},
                    normalized_score=score,
                    authority_score=float(item.get("authority", 0.75)),
                    freshness_score=1.0,
                    provenance_score=1.0,
                    content_hash=item.get("checksum", ""),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        return output
