from __future__ import annotations

from collections import defaultdict

from dualarrow.models import EvidenceCandidate
from dualarrow.utils import clamp, unique_preserve_order


def weighted_rrf(
    results: list[tuple[str, float, list[EvidenceCandidate]]],
    *,
    rrf_k: int = 60,
    limit: int = 100,
) -> list[EvidenceCandidate]:
    fused: dict[str, EvidenceCandidate] = {}
    scores: defaultdict[str, float] = defaultdict(float)
    for method, weight, candidates in results:
        for rank, candidate in enumerate(candidates, start=1):
            key = candidate.unit_id
            scores[key] += weight / (rrf_k + rank)
            if key not in fused:
                fused[key] = candidate.model_copy(deep=True)
            else:
                current = fused[key]
                current.retrieval_methods = unique_preserve_order(
                    [*current.retrieval_methods, *candidate.retrieval_methods, method]
                )
                current.scores.update(candidate.scores)
                current.authority_score = max(current.authority_score, candidate.authority_score)
                current.freshness_score = max(current.freshness_score, candidate.freshness_score)
                current.provenance_score = max(current.provenance_score, candidate.provenance_score)
    if not scores:
        return []
    max_score = max(scores.values()) or 1.0
    for key, candidate in fused.items():
        candidate.normalized_score = clamp(scores[key] / max_score)
        candidate.scores["fusion"] = scores[key]
    return sorted(
        fused.values(),
        key=lambda item: (
            item.normalized_score,
            item.authority_score,
            item.freshness_score,
        ),
        reverse=True,
    )[:limit]
