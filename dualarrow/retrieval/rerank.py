from __future__ import annotations

import asyncio

from dualarrow.errors import AdapterUnavailableError
from dualarrow.models import EvidenceCandidate
from dualarrow.utils import clamp, tokenize


class HeuristicReranker:
    def rerank(
        self, query: str, candidates: list[EvidenceCandidate], limit: int
    ) -> list[EvidenceCandidate]:
        query_tokens = set(tokenize(query))
        for item in candidates:
            candidate_tokens = set(tokenize(item.title + " " + item.content))
            overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
            exact_bonus = 0.15 if query.casefold() in item.content.casefold() else 0.0
            score = (
                0.48 * item.normalized_score
                + 0.26 * overlap
                + 0.12 * item.authority_score
                + 0.07 * item.freshness_score
                + 0.07 * item.provenance_score
                + exact_bonus
            )
            item.scores["reranker"] = score
            item.normalized_score = clamp(score)
        return sorted(candidates, key=lambda item: item.normalized_score, reverse=True)[:limit]


class CrossEncoderReranker:
    def __init__(self, model: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Install the local-ml extra for cross-encoder reranking"
            ) from exc
        self.model = CrossEncoder(model)

    async def rerank(
        self, query: str, candidates: list[EvidenceCandidate], limit: int
    ) -> list[EvidenceCandidate]:
        if not candidates:
            return []
        pairs = [(query, candidate.content) for candidate in candidates]
        raw = await asyncio.to_thread(self.model.predict, pairs)
        values = [float(value) for value in raw]
        minimum = min(values)
        maximum = max(values)
        span = maximum - minimum
        for candidate, value in zip(candidates, values, strict=True):
            normalized = (value - minimum) / span if span else 0.5
            candidate.scores["cross_encoder"] = value
            candidate.normalized_score = clamp(
                0.75 * normalized + 0.15 * candidate.authority_score + 0.10 * candidate.freshness_score
            )
        return sorted(candidates, key=lambda item: item.normalized_score, reverse=True)[:limit]
