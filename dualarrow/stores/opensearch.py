from __future__ import annotations

from urllib.parse import urlparse

from dualarrow.errors import AdapterUnavailableError
from dualarrow.models import EvidenceCandidate, KnowledgeUnit, SourceDocument
from dualarrow.utils import stable_id


class OpenSearchLexicalAdapter:
    def __init__(
        self,
        *,
        url: str,
        index: str,
        username: str | None = None,
        password: str | None = None,
        verify_certs: bool = True,
    ) -> None:
        try:
            from opensearchpy import AsyncOpenSearch
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Install the OpenSearch extra: pip install 'dualarrow-rag[opensearch]'"
            ) from exc
        parsed = urlparse(url)
        self.client = AsyncOpenSearch(
            hosts=[{
                "host": parsed.hostname or "localhost",
                "port": parsed.port or (443 if parsed.scheme == "https" else 9200),
                "scheme": parsed.scheme or "http",
            }],
            http_auth=(username, password) if username else None,
            verify_certs=verify_certs,
            use_ssl=parsed.scheme == "https",
        )
        self.index = index

    async def initialize(self) -> None:
        if not await self.client.indices.exists(index=self.index):
            await self.client.indices.create(
                index=self.index,
                body={
                    "mappings": {
                        "properties": {
                            "unit_id": {"type": "keyword"},
                            "document_id": {"type": "keyword"},
                            "view_id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "heading_path": {"type": "text"},
                            "content": {"type": "text"},
                            "source_uri": {"type": "keyword"},
                            "authority": {"type": "float"},
                            "metadata": {"type": "object", "dynamic": True},
                        }
                    }
                },
            )

    async def close(self) -> None:
        await self.client.close()

    async def replace_document(self, document: SourceDocument, units: list[KnowledgeUnit]) -> None:
        await self.delete_document(document.document_id)
        if not units:
            return
        body: list[dict] = []
        for unit in units:
            body.append({"index": {"_index": self.index, "_id": unit.unit_id}})
            payload = unit.model_dump(mode="json")
            payload["content_hash"] = unit.checksum
            payload["heading_path"] = " / ".join(unit.heading_path)
            body.append(payload)
        await self.client.bulk(body=body, refresh=True)


    async def delete_document(self, document_id: str) -> None:
        await self.client.delete_by_query(
            index=self.index,
            body={"query": {"term": {"document_id": document_id}}},
            conflicts="proceed",
            refresh=True,
        )

    async def delete_documents_not_in(self, document_ids: set[str]) -> int:
        return 0

    async def search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        filter_clauses = [{"term": {f"metadata.{key}.keyword": value}} for key, value in filters.items()]
        response = await self.client.search(
            index=self.index,
            size=top_k,
            body={
                "query": {
                    "bool": {
                        "must": [{
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "heading_path^2", "content"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        }],
                        "filter": filter_clauses,
                    }
                }
            },
        )
        output = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            heading = source.get("heading_path", "")
            output.append(
                EvidenceCandidate(
                    evidence_id=stable_id(source["unit_id"], prefix="ev_"),
                    unit_id=source["unit_id"],
                    document_id=source["document_id"],
                    content=source["content"],
                    source_uri=source["source_uri"],
                    title=source.get("title", ""),
                    heading_path=heading.split(" / ") if heading else [],
                    line_start=source.get("line_start"),
                    line_end=source.get("line_end"),
                    retrieval_methods=["bm25"],
                    scores={"bm25": float(hit["_score"] or 0)},
                    normalized_score=float(hit["_score"] or 0),
                    authority_score=float(source.get("authority", 0.75)),
                    freshness_score=1.0,
                    provenance_score=1.0,
                    content_hash=source.get("content_hash", source.get("checksum", "")),
                    metadata=source.get("metadata", {}),
                )
            )
        return output
