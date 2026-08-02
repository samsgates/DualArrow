from __future__ import annotations

import json
from urllib.parse import quote

import httpx

from dualarrow.models import EvidenceCandidate, KnowledgeUnit, SourceDocument
from dualarrow.utils import stable_id, tokenize


class GraphDBAdapter:
    """GraphDB adapter using the standard SPARQL query and update endpoints."""

    NS = "https://dualarrow.dev/schema/"
    RES = "https://dualarrow.dev/resource/"

    def __init__(
        self,
        *,
        query_endpoint: str,
        update_endpoint: str,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 60,
    ) -> None:
        self.query_endpoint = query_endpoint
        self.update_endpoint = update_endpoint
        self.client = httpx.AsyncClient(
            timeout=timeout,
            auth=(username, password) if username else None,
        )

    async def initialize(self) -> None:
        response = await self.client.post(
            self.query_endpoint,
            data={"query": "ASK { ?s ?p ?o }"},
            headers={"Accept": "application/sparql-results+json"},
        )
        response.raise_for_status()

    async def close(self) -> None:
        await self.client.aclose()

    async def _update(self, update: str) -> None:
        response = await self.client.post(
            self.update_endpoint,
            data={"update": update},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()

    @staticmethod
    def _literal(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    async def replace_document(
        self, document: SourceDocument, units: list[KnowledgeUnit], links: list[str]
    ) -> None:
        doc_uri = f"<{self.RES}document/{quote(document.document_id, safe='')}>"
        await self.delete_document(document.document_id)
        triples = [
            f"{doc_uri} a <{self.NS}Document> ;",
            f"  <{self.NS}documentId> {self._literal(document.document_id)} ;",
            f"  <{self.NS}title> {self._literal(document.title)} ;",
            f"  <{self.NS}sourceUri> {self._literal(document.source_uri)} .",
        ]
        for unit in units:
            unit_uri = f"<{self.RES}unit/{quote(unit.unit_id, safe='')}>"
            triples.extend([
                f"{unit_uri} a <{self.NS}KnowledgeUnit> ;",
                f"  <{self.NS}unitId> {self._literal(unit.unit_id)} ;",
                f"  <{self.NS}documentId> {self._literal(unit.document_id)} ;",
                f"  <{self.NS}content> {self._literal(unit.content)} ;",
                f"  <{self.NS}title> {self._literal(unit.title)} ;",
                f"  <{self.NS}sourceUri> {self._literal(unit.source_uri)} ;",
                f"  <{self.NS}checksum> {self._literal(unit.checksum)} ;",
                f"  <{self.NS}authority> {unit.authority} .",
                f"{doc_uri} <{self.NS}contains> {unit_uri} .",
            ])
        for target in links:
            triples.append(
                f"{doc_uri} <{self.NS}linksTo> {self._literal(target)} ."
            )
        await self._update("INSERT DATA {\n" + "\n".join(triples) + "\n}")


    async def delete_document(self, document_id: str) -> None:
        doc_uri = f"<{self.RES}document/{quote(document_id, safe='')}>"
        update = f"""
        DELETE {{
          {doc_uri} ?docP ?docO .
          ?unit ?unitP ?unitO .
        }}
        WHERE {{
          OPTIONAL {{ {doc_uri} ?docP ?docO . }}
          OPTIONAL {{
            {doc_uri} <{self.NS}contains> ?unit .
            ?unit ?unitP ?unitO .
          }}
        }}
        """
        await self._update(update)

    async def delete_documents_not_in(self, document_ids: set[str]) -> int:
        return 0

    async def search(self, query: str, top_k: int, filters: dict) -> list[EvidenceCandidate]:
        tokens = [token for token in tokenize(query) if len(token) > 2][:8]
        if not tokens:
            return []
        conditions = " || ".join(
            f"CONTAINS(LCASE(STR(?content)), {self._literal(token)})" for token in tokens
        )
        sparql = f"""
        SELECT ?unitId ?documentId ?content ?title ?sourceUri ?checksum ?authority WHERE {{
          ?u a <{self.NS}KnowledgeUnit> ;
             <{self.NS}unitId> ?unitId ;
             <{self.NS}documentId> ?documentId ;
             <{self.NS}content> ?content ;
             <{self.NS}title> ?title ;
             <{self.NS}sourceUri> ?sourceUri ;
             <{self.NS}checksum> ?checksum ;
             <{self.NS}authority> ?authority .
          FILTER ({conditions})
        }} LIMIT {int(top_k)}
        """
        response = await self.client.post(
            self.query_endpoint,
            data={"query": sparql},
            headers={"Accept": "application/sparql-results+json"},
        )
        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]
        output = []
        for row in bindings:
            content = row["content"]["value"]
            score = sum(1.0 for token in tokens if token in content.casefold())
            unit_id = row["unitId"]["value"]
            output.append(
                EvidenceCandidate(
                    evidence_id=stable_id(unit_id, prefix="ev_"),
                    unit_id=unit_id,
                    document_id=row["documentId"]["value"],
                    content=content,
                    source_uri=row["sourceUri"]["value"],
                    title=row["title"]["value"],
                    retrieval_methods=["graph"],
                    scores={"graph": score},
                    normalized_score=score,
                    authority_score=float(row["authority"]["value"]),
                    freshness_score=1.0,
                    provenance_score=1.0,
                    content_hash=row["checksum"]["value"],
                )
            )
        return output
