from __future__ import annotations

from dualarrow.ingestion.chunking import _unit


def first_paragraph_view(parsed, view_id: str, options: dict):
    maximum = int(options.get("maximum_characters", 1000))
    paragraph = parsed.document.content.split("\n\n", 1)[0][:maximum]
    return [
        _unit(
            parsed,
            view_id=view_id,
            content=paragraph,
            heading_path=[parsed.document.title],
            line_start=1,
            line_end=max(1, paragraph.count("\n") + 1),
            metadata={"plugin": "first_paragraph"},
        )
    ]


class AuthorityBM25Retriever:
    name = "authority_bm25"

    def __init__(self, store, minimum_authority: float) -> None:
        self.store = store
        self.minimum_authority = minimum_authority

    async def retrieve(self, query: str, top_k: int, filters: dict):
        candidates = await self.store.lexical_search(query, top_k * 2, filters)
        candidates = [
            candidate
            for candidate in candidates
            if candidate.authority_score >= self.minimum_authority
        ]
        for candidate in candidates:
            candidate.retrieval_methods = ["authority_bm25"]
        return candidates[:top_k]


def create_authority_retriever(store, embeddings, options: dict):
    return AuthorityBM25Retriever(
        store,
        minimum_authority=float(options.get("minimum_authority", 0.9)),
    )
